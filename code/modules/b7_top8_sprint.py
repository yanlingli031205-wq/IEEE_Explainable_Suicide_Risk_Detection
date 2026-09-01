"""B7 Top-8 sprint utilities for the Lenormand IEEE BigData Cup system.

This module is deliberately CPU-only.  It does not train or load a language
model.  It consumes the frozen B4 three-fold Qwen logits and the frozen B4-E2
test candidate table, then provides:

* strict grouped outer-OOF evaluation of an adjusted-classify-and-count (ACC)
  Factor decoder;
* fold-margin and accepted ACC deployment to cached test probabilities;
* exactly reproducible B4-E2 evidence set decoding at nearby thresholds; and
* official-format, verbatim-evidence, component-isolation, and SHA256 audits.

The important safety property is that an ACC candidate is deployable only
when its *held-out* OOF gate passes.  Evidence threshold variants are labelled
as public-leaderboard probes rather than as newly validated models.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


B7_RUNTIME_REVISION = "2026-08-27.top8-sprint-v1"

FACTOR_LABELS = (
    "cognitive deficits",
    "coping strategy",
    "dysfunctional family",
    "emotion dysregulation",
    "exposure to others' suicide",
    "hopelessness",
    "interpersonal difficulty",
    "interpersonal violence",
    "low self-esteem",
    "low socio-economic status",
    "meaning in life",
    "mental health issues",
    "physical health/characteristic",
    "poor school performance",
    "poor social support",
    "prior self-harm or suicidal thought/attempt",
    "psychological capital",
    "sense of responsibility",
    "sexual orientation related issues",
    "social support",
    "stressful life event",
    "substance use",
    "suicide means (with access)",
    "traumatic experience",
)
OFFICIAL_COLUMNS = ("row_id", "risk_level", "evidence", "factors")
OFFICIAL_RISKS = ("Indicator", "Ideation", "Behavior", "Attempt")


@dataclass(frozen=True)
class FactorQuantificationConfig:
    q38_weight: float = 0.75
    acc_shrinks: tuple[float, ...] = (0.25, 0.50)
    expansion_min: float = 0.75
    expansion_max: float = 1.50
    acc_min_separation: float = 0.05
    threshold_kappa_tail: float = 0.0
    threshold_kappa_mid: float = 2.0
    threshold_kappa_head: float = 2.0
    gate_macro_f1: float = 0.006
    gate_tail_macro_f1: float = 0.0
    gate_folds_improved: int = 2
    gate_worst_fold_delta: float = -0.005


@dataclass(frozen=True)
class EvidenceProbeConfig:
    probability_threshold: float
    event_gap_chars: int = 40
    top_k_indicator: int = 0
    top_k_ideation: int = 2
    top_k_behavior: int = 3
    top_k_attempt: int = 2

    def top_k(self, risk: str) -> int:
        return int(
            {
                "indicator": self.top_k_indicator,
                "ideation": self.top_k_ideation,
                "behavior": self.top_k_behavior,
                "attempt": self.top_k_attempt,
            }.get(str(risk).strip().casefold(), self.top_k_behavior)
        )


def _json_default(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def json_dump(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def sha256(path: str | Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-5, 1 - 1e-5)
    return np.log(probability / (1.0 - probability))


def _average_precision(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.int8)
    score = np.asarray(score, dtype=float)
    positives = int(target.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    ranked = target[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float((precision * ranked).sum() / positives)


def per_label_f1(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    tp = np.logical_and(target, prediction).sum(axis=0).astype(float)
    fp = np.logical_and(~target, prediction).sum(axis=0).astype(float)
    fn = np.logical_and(target, ~prediction).sum(axis=0).astype(float)
    denominator = 2 * tp + fp + fn
    return np.divide(
        2 * tp,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0,
    )


def factor_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
    strata: np.ndarray | None = None,
) -> dict[str, float]:
    target = np.asarray(target, dtype=np.int8)
    prediction = np.asarray(prediction, dtype=np.int8)
    support = target.sum(axis=0)
    if strata is None:
        strata = np.where(support < 60, "tail", np.where(support < 200, "mid", "head"))
    label_f1 = per_label_f1(target, prediction)
    aps = [
        _average_precision(target[:, label], probability[:, label])
        for label in range(target.shape[1])
        if np.unique(target[:, label]).size == 2
    ]
    target_bool, prediction_bool = target.astype(bool), prediction.astype(bool)
    tp = np.logical_and(target_bool, prediction_bool).sum()
    fp = np.logical_and(~target_bool, prediction_bool).sum()
    fn = np.logical_and(target_bool, ~prediction_bool).sum()
    return {
        "macro_ap": float(np.mean(aps)),
        "macro_f1": float(label_f1.mean()),
        "micro_f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "tail_macro_f1": float(label_f1[strata == "tail"].mean()),
        "mid_macro_f1": float(label_f1[strata == "mid"].mean()),
        "head_macro_f1": float(label_f1[strata == "head"].mean()),
        "mean_predicted_labels": float(prediction.sum(axis=1).mean()),
    }


def fit_thresholds(
    probability: np.ndarray,
    target: np.ndarray,
    config: FactorQuantificationConfig,
) -> np.ndarray:
    """Reproduce B4's support-adaptive per-label threshold rule."""

    probability = np.asarray(probability, dtype=np.float32)
    target = np.asarray(target, dtype=bool)
    grid = np.linspace(0.02, 0.98, 97, dtype=np.float32)
    prediction = probability[:, :, None] >= grid[None, None, :]
    truth = target[:, :, None]
    tp = np.logical_and(prediction, truth).sum(axis=0)
    fp = np.logical_and(prediction, ~truth).sum(axis=0)
    fn = np.logical_and(~prediction, truth).sum(axis=0)
    denominator = 2 * tp + fp + fn
    scores = np.divide(
        2 * tp,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0,
    )
    global_threshold = float(grid[int(np.argmax(scores.mean(axis=0)))])
    support = target.sum(axis=0)
    thresholds = np.full(target.shape[1], global_threshold, dtype=np.float32)
    for label in range(target.shape[1]):
        if support[label] == 0 or support[label] == len(target):
            continue
        local = float(grid[int(np.argmax(scores[label]))])
        if support[label] < 60:
            kappa = config.threshold_kappa_tail
        elif support[label] < 200:
            kappa = config.threshold_kappa_mid
        else:
            kappa = config.threshold_kappa_head
        weight = support[label] / (support[label] + kappa) if kappa > 0 else 1.0
        thresholds[label] = weight * local + (1.0 - weight) * global_threshold
    return thresholds


def _top_k_mask(score: np.ndarray, count: int) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    count = int(np.clip(count, 0, len(score)))
    output = np.zeros(len(score), dtype=np.int8)
    if count:
        output[np.argsort(-score, kind="mergesort")[:count]] = 1
    return output


def _acc_prior(
    train_score: np.ndarray,
    train_target: np.ndarray,
    target_score: np.ndarray,
    threshold: float,
    config: FactorQuantificationConfig,
) -> tuple[float, dict[str, float]]:
    train_target = np.asarray(train_target, dtype=np.int8)
    train_positive = train_score >= threshold
    tpr = float(train_positive[train_target == 1].mean()) if train_target.any() else 0.0
    negative = train_target == 0
    fpr = float(train_positive[negative].mean()) if negative.any() else 0.0
    observed_rate = float((target_score >= threshold).mean())
    separation = tpr - fpr
    train_prior = float(train_target.mean())
    if separation < config.acc_min_separation:
        estimate = train_prior
        fallback = 1.0
    else:
        estimate = float(np.clip((observed_rate - fpr) / separation, 0.0, 1.0))
        fallback = 0.0
    return estimate, {
        "train_prior": train_prior,
        "tpr": tpr,
        "fpr": fpr,
        "separation": separation,
        "observed_positive_rate": observed_rate,
        "raw_acc_prior": estimate,
        "fallback_to_train_prior": fallback,
    }


def _quantified_prediction(
    train_score: np.ndarray,
    train_target: np.ndarray,
    target_score: np.ndarray,
    thresholds: np.ndarray,
    shrink: float,
    config: FactorQuantificationConfig,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    output = np.zeros((len(target_score), train_target.shape[1]), dtype=np.int8)
    diagnostics: list[dict[str, float]] = []
    for label in range(train_target.shape[1]):
        raw_prior, row = _acc_prior(
            train_score[:, label],
            train_target[:, label],
            target_score[:, label],
            float(thresholds[label]),
            config,
        )
        train_prior = row["train_prior"]
        adjusted_prior = (1.0 - shrink) * train_prior + shrink * raw_prior
        train_prediction_rate = float(
            (train_score[:, label] >= thresholds[label]).mean()
        )
        expansion = float(
            np.clip(
                train_prediction_rate / max(train_prior, 1e-6),
                config.expansion_min,
                config.expansion_max,
            )
        )
        count = int(round(len(target_score) * adjusted_prior * expansion))
        output[:, label] = _top_k_mask(target_score[:, label], count)
        diagnostics.append(
            {
                "label_index": float(label),
                **row,
                "shrink": float(shrink),
                "adjusted_prior": float(adjusted_prior),
                "f1_expansion": expansion,
                "predicted_count": float(output[:, label].sum()),
            }
        )
    return output, diagnostics


def run_factor_quantification_oof(
    row_ids: Sequence[str],
    folds: np.ndarray,
    target: np.ndarray,
    probability: np.ndarray,
    config: FactorQuantificationConfig,
    output_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate all ACC decisions on unseen grouped outer folds."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    row_ids = np.asarray(row_ids).astype(str)
    folds = np.asarray(folds, dtype=int)
    target = np.asarray(target, dtype=np.int8)
    probability = np.asarray(probability, dtype=np.float32)
    if probability.shape != target.shape or len(row_ids) != len(target):
        raise ValueError("Factor OOF row/shape mismatch")
    if not np.isfinite(probability).all():
        raise ValueError("Factor OOF contains non-finite probabilities")

    systems = ["BASELINE_THRESHOLD"] + [
        f"ACC_SHRINK_{int(round(shrink * 100)):03d}" for shrink in config.acc_shrinks
    ]
    predictions = {name: np.zeros_like(target, dtype=np.int8) for name in systems}
    thresholds_by_fold: dict[int, np.ndarray] = {}
    prior_rows: list[dict[str, Any]] = []
    outer_values = sorted(np.unique(folds).tolist())
    for outer_fold in outer_values:
        train = folds != outer_fold
        valid = folds == outer_fold
        thresholds = fit_thresholds(probability[train], target[train], config)
        thresholds_by_fold[int(outer_fold)] = thresholds
        predictions["BASELINE_THRESHOLD"][valid] = (
            probability[valid] >= thresholds[None, :]
        )
        for shrink in config.acc_shrinks:
            name = f"ACC_SHRINK_{int(round(shrink * 100)):03d}"
            fold_prediction, diagnostics = _quantified_prediction(
                probability[train],
                target[train],
                probability[valid],
                thresholds,
                shrink,
                config,
            )
            predictions[name][valid] = fold_prediction
            for row in diagnostics:
                label = int(row.pop("label_index"))
                prior_rows.append(
                    {
                        "outer_fold": int(outer_fold),
                        "system": name,
                        "label": FACTOR_LABELS[label],
                        **row,
                    }
                )

    support = target.sum(axis=0)
    strata = np.where(support < 60, "tail", np.where(support < 200, "mid", "head"))
    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    for name in systems:
        summary_rows.append(
            {"system": name, **factor_metrics(target, probability, predictions[name], strata)}
        )
        label_f1 = per_label_f1(target, predictions[name])
        for label, label_name in enumerate(FACTOR_LABELS):
            label_rows.append(
                {
                    "system": name,
                    "label": label_name,
                    "stratum": strata[label],
                    "support": int(support[label]),
                    "f1": float(label_f1[label]),
                    "ap": _average_precision(target[:, label], probability[:, label]),
                    "predicted_count": int(predictions[name][:, label].sum()),
                }
            )
        for outer_fold in outer_values:
            valid = folds == outer_fold
            fold_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "system": name,
                    **factor_metrics(
                        target[valid], probability[valid], predictions[name][valid], strata
                    ),
                }
            )

    summary = pd.DataFrame(summary_rows)
    fold_frame = pd.DataFrame(fold_rows)
    per_label = pd.DataFrame(label_rows)
    baseline = summary.set_index("system").loc["BASELINE_THRESHOLD"]
    baseline_folds = fold_frame[fold_frame.system == "BASELINE_THRESHOLD"].set_index(
        "outer_fold"
    )
    gate_rows: list[dict[str, Any]] = []
    for name in systems[1:]:
        row = summary.set_index("system").loc[name]
        candidate_folds = fold_frame[fold_frame.system == name].set_index("outer_fold")
        fold_delta = candidate_folds.macro_f1 - baseline_folds.macro_f1
        delta_macro = float(row.macro_f1 - baseline.macro_f1)
        delta_tail = float(row.tail_macro_f1 - baseline.tail_macro_f1)
        passed = bool(
            delta_macro >= config.gate_macro_f1
            and delta_tail >= config.gate_tail_macro_f1
            and int((fold_delta > 0).sum()) >= config.gate_folds_improved
            and float(fold_delta.min()) >= config.gate_worst_fold_delta
        )
        gate_rows.append(
            {
                "system": name,
                "delta_macro_f1": delta_macro,
                "delta_tail_macro_f1": delta_tail,
                "folds_improved": int((fold_delta > 0).sum()),
                "worst_fold_delta_macro_f1": float(fold_delta.min()),
                "passed": passed,
            }
        )
    gate = pd.DataFrame(gate_rows).sort_values(
        ["passed", "delta_macro_f1"], ascending=[False, False]
    )
    accepted = gate[gate.passed]
    selected = str(accepted.iloc[0].system) if len(accepted) else None

    summary.to_csv(output_dir / "B7_FACTOR_OOF_SUMMARY.csv", index=False)
    fold_frame.to_csv(output_dir / "B7_FACTOR_OUTER_FOLDS.csv", index=False)
    per_label.to_csv(output_dir / "B7_FACTOR_PER_LABEL.csv", index=False)
    pd.DataFrame(prior_rows).to_csv(output_dir / "B7_FACTOR_PRIOR_AUDIT.csv", index=False)
    gate.to_csv(output_dir / "B7_FACTOR_GATE.csv", index=False)
    np.savez_compressed(
        output_dir / "B7_FACTOR_OOF.npz",
        row_ids=row_ids,
        folds=folds,
        targets=target,
        probabilities=probability,
        **{f"prediction__{name}": value for name, value in predictions.items()},
    )
    decision = {
        "runtime_revision": B7_RUNTIME_REVISION,
        "status": "complete",
        "selected_factor_system": selected,
        "factor_gate_passed": selected is not None,
        "baseline": summary.set_index("system").loc["BASELINE_THRESHOLD"].to_dict(),
        "gate_results": gate.to_dict(orient="records"),
        "gate_rule": {
            "delta_macro_f1_min": config.gate_macro_f1,
            "delta_tail_macro_f1_min": config.gate_tail_macro_f1,
            "folds_improved_min": config.gate_folds_improved,
            "worst_fold_delta_min": config.gate_worst_fold_delta,
        },
        "config": asdict(config),
        "note": (
            "ACC only changes the decision set; rank-based Macro-AP is intentionally "
            "unchanged. Failure of this gate means no label-shift Factor candidate is "
            "licensed for final deployment."
        ),
    }
    json_dump(decision, output_dir / "B7_FACTOR_DECISION.json")
    state: dict[str, np.ndarray] = {
        "baseline_prediction": predictions["BASELINE_THRESHOLD"],
        **{name: value for name, value in predictions.items()},
    }
    for fold, thresholds in thresholds_by_fold.items():
        state[f"thresholds_fold_{fold}"] = thresholds
    return decision, state


def deploy_factor_decoders(
    oof_probability: np.ndarray,
    target: np.ndarray,
    test_fold_probabilities: Sequence[np.ndarray],
    state: Mapping[str, np.ndarray],
    factor_decision: Mapping[str, Any],
    config: FactorQuantificationConfig,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Create current, fold-margin, and (if licensed) ACC test decisions."""

    if len(test_fold_probabilities) != 3:
        raise ValueError("B7 deployment requires exactly three test probability folds")
    test_folds = [np.asarray(value, dtype=float) for value in test_fold_probabilities]
    if len({value.shape for value in test_folds}) != 1:
        raise ValueError("test fold probability shapes differ")
    all_thresholds = fit_thresholds(oof_probability, target, config)
    mean_probability = np.mean(np.stack(test_folds), axis=0)
    current = (mean_probability >= all_thresholds[None, :]).astype(np.int8)

    fold_margins = []
    for fold, probability in enumerate(test_folds):
        key = f"thresholds_fold_{fold}"
        if key not in state:
            raise KeyError(f"missing {key} from strict OOF state")
        fold_margins.append(logit(probability) - logit(state[key])[None, :])
    mean_margin = np.mean(np.stack(fold_margins), axis=0)
    fold_margin = (mean_margin >= 0).astype(np.int8)
    systems: dict[str, np.ndarray] = {
        "CURRENT_RECREATED": current,
        "FOLD_MARGIN_PROBE": fold_margin,
    }

    diagnostics: list[dict[str, Any]] = []
    selected = factor_decision.get("selected_factor_system")
    if selected:
        shrink = int(str(selected).rsplit("_", 1)[-1]) / 100.0
        quantified, rows = _quantified_prediction(
            np.asarray(oof_probability, dtype=float),
            np.asarray(target, dtype=np.int8),
            mean_probability,
            all_thresholds,
            shrink,
            config,
        )
        systems[str(selected)] = quantified
        for row in rows:
            label = int(row.pop("label_index"))
            diagnostics.append({"label": FACTOR_LABELS[label], **row})
    return systems, pd.DataFrame(diagnostics)


def factor_frame(row_ids: Sequence[str], prediction: np.ndarray) -> pd.DataFrame:
    prediction = np.asarray(prediction, dtype=np.int8)
    if prediction.shape != (len(row_ids), len(FACTOR_LABELS)):
        raise ValueError(f"invalid factor prediction shape: {prediction.shape}")
    lists = [
        [FACTOR_LABELS[label] for label in np.flatnonzero(row)] for row in prediction
    ]
    return pd.DataFrame(
        {"row_id": np.asarray(row_ids).astype(str), "factors": [repr(x) for x in lists]}
    )


def _normalize(value: Any) -> str:
    # Exact B4-E/B4-E2 normalization; keeping this byte-for-byte equivalent is
    # required for the default 0.40 decoder reproduction gate.
    return " ".join(str(value or "").strip().casefold().split())


def _near_same_event(
    left: int,
    right: int,
    selected: Sequence[tuple[int, int]],
    gap: int,
) -> bool:
    for other_left, other_right in selected:
        if right < other_left:
            distance = other_left - right
        elif other_right < left:
            distance = left - other_right
        else:
            distance = 0
        if distance <= gap:
            return True
    return False


def serialize_evidence(items: Sequence[str]) -> str:
    pieces: list[str] = []
    seen: set[str] = set()
    for item in items:
        for piece in str(item).split(";"):
            piece = piece.strip()
            normalized = _normalize(piece)
            if piece and normalized not in seen:
                pieces.append(piece)
                seen.add(normalized)
    return "; ".join(pieces)


def decode_evidence_candidates(
    candidate_table: pd.DataFrame,
    risk_frame: pd.DataFrame,
    config: EvidenceProbeConfig,
) -> dict[str, list[str]]:
    required = {"row_id", "candidate", "left", "right", "meta_probability", "margin"}
    missing = required - set(candidate_table.columns)
    if missing:
        raise KeyError(f"candidate meta table missing columns: {sorted(missing)}")
    if not {"row_id", "risk_level"}.issubset(risk_frame.columns):
        raise KeyError("risk frame requires row_id and risk_level")
    table = candidate_table.copy()
    table["row_id"] = table.row_id.astype(str)
    table["meta_probability"] = pd.to_numeric(table.meta_probability, errors="coerce")
    if "candidate_tokens" not in table.columns:
        table["candidate_tokens"] = table.candidate.map(
            lambda value: len(_normalize(value).split())
        )
    risk_by_row = dict(zip(risk_frame.row_id.astype(str), risk_frame.risk_level))
    groups = {row_id: group for row_id, group in table.groupby("row_id", sort=False)}
    output: dict[str, list[str]] = {}
    for row_id in risk_frame.row_id.astype(str):
        top_k = config.top_k(risk_by_row[row_id])
        if top_k <= 0 or row_id not in groups:
            output[row_id] = []
            continue
        candidates = groups[row_id]
        candidates = candidates[
            np.isfinite(candidates.meta_probability)
            & (candidates.meta_probability >= config.probability_threshold)
        ].sort_values(
            ["meta_probability", "candidate_tokens", "margin"],
            ascending=[False, True, False],
            kind="mergesort",
        )
        selected: list[str] = []
        selected_spans: list[tuple[int, int]] = []
        selected_literals: set[str] = set()
        for item in candidates.itertuples():
            literal = _normalize(item.candidate)
            if not literal or literal in selected_literals:
                continue
            left, right = int(item.left), int(item.right)
            if _near_same_event(left, right, selected_spans, config.event_gap_chars):
                continue
            selected.append(str(item.candidate).strip())
            selected_spans.append((left, right))
            selected_literals.add(literal)
            if len(selected) >= top_k:
                break
        output[row_id] = selected
    return output


def evidence_frame(
    risk_frame: pd.DataFrame,
    evidence_by_row: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    frame = risk_frame[["row_id", "risk_level"]].copy()
    frame["row_id"] = frame.row_id.astype(str)
    frame["evidence"] = [
        serialize_evidence(evidence_by_row.get(row_id, [])) for row_id in frame.row_id
    ]
    return frame


def _text_column(frame: pd.DataFrame) -> str:
    normalized = {re.sub(r"[^a-z0-9]", "", str(c).casefold()): c for c in frame.columns}
    for candidate in ("post", "text"):
        if candidate in normalized:
            return str(normalized[candidate])
    raise KeyError(f"could not find post/text in {frame.columns.tolist()}")


def audit_submission(
    submission: pd.DataFrame,
    test_frame: pd.DataFrame,
    original: pd.DataFrame | None = None,
) -> dict[str, Any]:
    submission = submission.copy().fillna("")
    test_frame = test_frame.copy().fillna("")
    if submission.columns.tolist() != list(OFFICIAL_COLUMNS):
        raise AssertionError(f"invalid submission columns: {submission.columns.tolist()}")
    expected_rows = test_frame.row_id.astype(str).tolist()
    if submission.row_id.astype(str).tolist() != expected_rows:
        raise AssertionError("submission row order differs from leaderboard.xlsx")
    if submission.row_id.astype(str).duplicated().any():
        raise AssertionError("duplicate row_id")
    if not set(submission.risk_level).issubset(OFFICIAL_RISKS):
        raise AssertionError("invalid risk label")

    allowed = set(FACTOR_LABELS)
    factor_lists: list[list[str]] = []
    for raw in submission.factors:
        parsed = ast.literal_eval(str(raw))
        if not isinstance(parsed, list) or not set(parsed).issubset(allowed):
            raise AssertionError(f"invalid factor list: {raw}")
        if len(parsed) != len(set(parsed)):
            raise AssertionError(f"duplicate factor: {raw}")
        factor_lists.append(parsed)

    text_col = _text_column(test_frame)
    text_by_row = dict(zip(test_frame.row_id.astype(str), test_frame[text_col].astype(str)))
    evidence_counts: list[int] = []
    verbatim_failures: list[dict[str, str]] = []
    indicator_nonempty = 0
    for row in submission.itertuples(index=False):
        phrases = [piece.strip() for piece in str(row.evidence).split(";") if piece.strip()]
        evidence_counts.append(len(phrases))
        if row.risk_level == "Indicator" and phrases:
            indicator_nonempty += 1
        for phrase in phrases:
            if phrase not in text_by_row[str(row.row_id)]:
                verbatim_failures.append({"row_id": str(row.row_id), "phrase": phrase})
    if indicator_nonempty or verbatim_failures:
        raise AssertionError(
            f"submission evidence invalid: indicator_nonempty={indicator_nonempty}, "
            f"verbatim_failures={len(verbatim_failures)}"
        )

    audit: dict[str, Any] = {
        "runtime_revision": B7_RUNTIME_REVISION,
        "rows": int(len(submission)),
        "mean_evidence_phrases": float(np.mean(evidence_counts)),
        "mean_factors": float(np.mean([len(value) for value in factor_lists])),
        "indicator_nonempty_evidence": int(indicator_nonempty),
        "verbatim_failures": int(len(verbatim_failures)),
        "status": "READY_TO_UPLOAD",
    }
    if original is not None:
        original = original.copy().fillna("")
        if original.row_id.astype(str).tolist() != expected_rows:
            raise AssertionError("original submission rows differ")
        for column in OFFICIAL_COLUMNS[1:]:
            changed = submission[column].astype(str).to_numpy() != original[column].astype(str).to_numpy()
            audit[f"changed_{column}_rows"] = int(changed.sum())
        audit["changed_any_rows"] = int(
            np.any(
                submission[list(OFFICIAL_COLUMNS[1:])].astype(str).to_numpy()
                != original[list(OFFICIAL_COLUMNS[1:])].astype(str).to_numpy(),
                axis=1,
            ).sum()
        )
    return audit


def write_submission_variant(
    name: str,
    original: pd.DataFrame,
    test_frame: pd.DataFrame,
    output_dir: str | Path,
    task1_frame: pd.DataFrame | None = None,
    factor_prediction_frame: pd.DataFrame | None = None,
    declared_component: str = "unknown",
) -> dict[str, Any]:
    output_dir = Path(output_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = original.copy().fillna("")
    frame["row_id"] = frame.row_id.astype(str)
    if task1_frame is not None:
        task1 = task1_frame.copy().fillna("")
        task1["row_id"] = task1.row_id.astype(str)
        if task1.row_id.tolist() != frame.row_id.tolist():
            raise AssertionError(f"{name}: Task1 row mismatch")
        frame["risk_level"] = task1.risk_level.astype(str)
        frame["evidence"] = task1.evidence.astype(str)
    if factor_prediction_frame is not None:
        factors = factor_prediction_frame.copy().fillna("")
        factors["row_id"] = factors.row_id.astype(str)
        if factors.row_id.tolist() != frame.row_id.tolist():
            raise AssertionError(f"{name}: Factor row mismatch")
        frame["factors"] = factors.factors.astype(str)
    frame = frame[list(OFFICIAL_COLUMNS)]
    audit = audit_submission(frame, test_frame, original)
    path = output_dir / "Lenormand.csv"
    frame.to_csv(path, index=False)
    reread = pd.read_csv(path, dtype=str, keep_default_na=False)
    audit_submission(reread, test_frame, original)
    audit.update(
        {
            "candidate": name,
            "declared_component": declared_component,
            "path": str(path),
            "sha256": sha256(path),
        }
    )
    json_dump(audit, output_dir / "AUDIT.json")
    return audit


def load_probability_npz(
    path: str | Path,
    expected_row_ids: Sequence[str],
    preferred_keys: Sequence[str] = ("verifier_logits", "logits", "probabilities"),
) -> np.ndarray:
    path = Path(path)
    saved = np.load(path, allow_pickle=True)
    if saved["row_ids"].astype(str).tolist() != np.asarray(expected_row_ids).astype(str).tolist():
        raise AssertionError(f"row mismatch: {path}")
    for key in preferred_keys:
        if key in saved.files:
            matrix = np.asarray(saved[key], dtype=np.float32)
            probability = matrix if "prob" in key else sigmoid(matrix)
            if not np.isfinite(probability).all():
                raise ValueError(f"non-finite probability: {path}")
            return probability.astype(np.float32)
    raise KeyError(f"none of {preferred_keys} in {path}: {saved.files}")
