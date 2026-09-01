"""B4-E: official-metric-aware evidence set calibration.

This module consumes the cached Qwen3.8 candidate margins produced by
``b4_task1_q38.py``.  It deliberately performs the cheap, falsifiable part of
the Evidence upgrade before another 27B training run:

* score with the published symmetric, one-to-one, per-post Phrase-F1;
* collapse overlapping/nearby candidates into evidence events;
* calibrate score threshold and risk-conditional output cardinality; and
* evaluate the calibration by inner user-grouped cross-fitting.

No validation Gold is used to score its own row: every cross-fitted prediction
is produced by a configuration selected on the other inner folds.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

import b1_experiments as b1
import qwen38_dual_task_experiments as q38


B4E_RUNTIME_REVISION = "2026-08-24.official-one-to-one-event-set-v1"
RISK_ORDER = ("Indicator", "Ideation", "Behavior", "Attempt")


def _normalize(value: Any) -> str:
    return " ".join(str(value).strip().casefold().split())


def _parse_phrase_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, np.ndarray)):
        raw = list(value)
    elif value is None or (isinstance(value, float) and math.isnan(value)):
        raw = []
    else:
        text = str(value).strip()
        if not text or text.casefold() == "none":
            raw = []
        else:
            try:
                decoded = json.loads(text)
                raw = decoded if isinstance(decoded, list) else [decoded]
            except (json.JSONDecodeError, TypeError):
                raw = text.split(";")
    return [str(item).strip() for item in raw if _normalize(item) not in {"", "none"}]


@dataclass(frozen=True)
class EvidenceSetConfig:
    """A small, predeclared Evidence set decoder configuration."""

    margin_threshold: float = 0.0
    event_gap_chars: int = 12
    length_penalty_per_token: float = 0.05
    top_k_indicator: int = 0
    top_k_ideation: int = 1
    top_k_behavior: int = 2
    top_k_attempt: int = 2

    def top_k(self, risk: str) -> int:
        values = {
            "indicator": self.top_k_indicator,
            "ideation": self.top_k_ideation,
            "behavior": self.top_k_behavior,
            "attempt": self.top_k_attempt,
        }
        return int(values.get(str(risk).strip().casefold(), self.top_k_behavior))

    @property
    def key(self) -> str:
        return (
            f"thr={self.margin_threshold:g}|gap={self.event_gap_chars}|"
            f"len={self.length_penalty_per_token:g}|"
            f"k={self.top_k_indicator}{self.top_k_ideation}"
            f"{self.top_k_behavior}{self.top_k_attempt}"
        )


def default_config_grid() -> list[EvidenceSetConfig]:
    """Predeclared compact grid; do not expand it after viewing fold results."""

    cardinalities = (
        (0, 1, 1, 1),
        (0, 1, 2, 1),
        (0, 1, 2, 2),
        (0, 2, 2, 2),
        (0, 2, 3, 2),
        (0, 2, 3, 3),
    )
    return [
        EvidenceSetConfig(
            margin_threshold=threshold,
            event_gap_chars=gap,
            length_penalty_per_token=penalty,
            top_k_indicator=ks[0],
            top_k_ideation=ks[1],
            top_k_behavior=ks[2],
            top_k_attempt=ks[3],
        )
        for threshold in (-1.0, 0.0, 0.5, 1.0, 1.5, 2.0)
        for gap in (0, 8, 20, 40)
        for penalty in (0.0, 0.05, 0.10)
        for ks in cardinalities
    ]


def _row_id_from_pair_id(value: Any) -> str | None:
    match = re.search(r"::([^:]+)::", str(value))
    return match.group(1) if match else None


def prepare_artifacts(
    audit: pd.DataFrame,
    validation: pd.DataFrame,
    bundle: b1.DataBundle,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and align cached candidate margins with the training bundle."""

    required_audit = {"query_row_idx", "left", "right", "candidate", "margin"}
    required_validation = {
        "row_id", "blended_risk", "predicted_evidence", "gold_evidence",
    }
    missing_audit = required_audit - set(audit.columns)
    missing_validation = required_validation - set(validation.columns)
    if missing_audit or missing_validation:
        raise KeyError(
            f"missing audit={sorted(missing_audit)}, validation={sorted(missing_validation)}"
        )

    row_id_by_index = {idx: str(row_id) for idx, row_id in enumerate(bundle.row_ids)}
    user_by_row_id = {
        str(row_id): str(user_id)
        for row_id, user_id in zip(bundle.row_ids, bundle.user_ids)
    }
    audit = audit.copy()
    if "row_id" not in audit.columns:
        if "pair_id" in audit.columns:
            audit["row_id"] = audit.pair_id.map(_row_id_from_pair_id)
        else:
            audit["row_id"] = audit.query_row_idx.astype(int).map(row_id_by_index)
    unresolved = audit.row_id.isna()
    if unresolved.any():
        audit.loc[unresolved, "row_id"] = (
            audit.loc[unresolved, "query_row_idx"].astype(int).map(row_id_by_index)
        )
    audit["row_id"] = audit.row_id.astype(str)
    audit["left"] = audit.left.astype(int)
    audit["right"] = audit.right.astype(int)
    audit["margin"] = pd.to_numeric(audit.margin, errors="coerce")
    audit = audit[np.isfinite(audit.margin)].copy()
    audit["candidate"] = audit.candidate.astype(str)
    audit["normalized_candidate"] = audit.candidate.map(_normalize)
    audit["candidate_tokens"] = audit.normalized_candidate.str.split().map(len)
    audit = audit[audit.normalized_candidate.ne("")].copy()

    validation = validation.copy()
    validation["row_id"] = validation.row_id.astype(str)
    validation["user_id"] = validation.row_id.map(user_by_row_id)
    if validation.user_id.isna().any():
        missing = validation.loc[validation.user_id.isna(), "row_id"].head(10).tolist()
        raise AssertionError(f"validation row_ids not found in bundle: {missing}")
    validation["baseline_evidence"] = validation.predicted_evidence.map(_parse_phrase_list)
    validation["gold_evidence_list"] = validation.gold_evidence.map(_parse_phrase_list)
    validation["gold_count"] = validation.gold_evidence_list.map(len).astype(int)

    valid_ids = set(validation.row_id)
    audit = audit[audit.row_id.isin(valid_ids)].copy()
    missing_candidate_rows = valid_ids - set(audit.row_id)
    if missing_candidate_rows:
        # Rows without a candidate are legal and simply decode to an empty set.
        print(f"[B4-E] {len(missing_candidate_rows)} validation rows have no cached candidates")
    return audit.reset_index(drop=True), validation.reset_index(drop=True)


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


def select_row(
    candidates: pd.DataFrame,
    predicted_risk: str,
    config: EvidenceSetConfig,
) -> list[str]:
    """Decode one evidence set with literal and event-level duplicate control."""

    top_k = config.top_k(predicted_risk)
    if top_k <= 0 or candidates.empty:
        return []
    accepted = candidates[candidates.margin >= config.margin_threshold].copy()
    if accepted.empty:
        return []
    accepted["decoder_score"] = (
        accepted.margin
        - config.length_penalty_per_token * accepted.candidate_tokens.astype(float)
    )
    accepted = accepted.sort_values(
        ["decoder_score", "margin", "candidate_tokens"],
        ascending=[False, False, True],
        kind="mergesort",
    )

    output: list[str] = []
    selected_spans: list[tuple[int, int]] = []
    selected_literals: set[str] = set()
    for item in accepted.itertuples():
        literal = str(item.normalized_candidate)
        if literal in selected_literals:
            continue
        left, right = int(item.left), int(item.right)
        if _near_same_event(left, right, selected_spans, config.event_gap_chars):
            continue
        output.append(str(item.candidate).strip())
        selected_spans.append((left, right))
        selected_literals.add(literal)
        if len(output) >= top_k:
            break
    return output


def decode_config(
    audit: pd.DataFrame,
    validation: pd.DataFrame,
    config: EvidenceSetConfig,
    row_ids: Iterable[str] | None = None,
    candidate_groups: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, list[str]]:
    wanted = set(validation.row_id if row_ids is None else map(str, row_ids))
    risk_by_row = dict(zip(validation.row_id, validation.blended_risk))
    if candidate_groups is None:
        candidate_groups = {
            row_id: group for row_id, group in audit.groupby("row_id", sort=False)
        }
    empty = audit.iloc[:0]
    return {
        row_id: select_row(candidate_groups.get(row_id, empty), risk_by_row[row_id], config)
        for row_id in validation.row_id
        if row_id in wanted
    }


def score_predictions(
    validation: pd.DataFrame,
    predictions: Mapping[str, Sequence[str]],
    row_ids: Iterable[str] | None = None,
) -> dict[str, float]:
    wanted = set(validation.row_id if row_ids is None else map(str, row_ids))
    frame = validation[validation.row_id.isin(wanted)]
    predicted = [list(predictions.get(row_id, [])) for row_id in frame.row_id]
    gold = frame.gold_evidence_list.tolist()
    metrics = q38.official_like_phrase_f1(predicted, gold)
    metrics["mean_predicted_phrases"] = float(np.mean([len(row) for row in predicted]))
    metrics["rows"] = int(len(frame))
    return metrics


def baseline_score(validation: pd.DataFrame) -> dict[str, float]:
    predictions = dict(zip(validation.row_id, validation.baseline_evidence))
    return score_predictions(validation, predictions)


def make_inner_splits(
    validation: pd.DataFrame,
    n_splits: int = 3,
    seed: int = 20260824,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """User-grouped inner partitions, stratified by risk and Gold count."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    indices = np.arange(len(validation))
    # Risk alone is deliberately used as the stratum.  Crossing it with exact
    # Gold counts creates singleton classes (e.g. one post with eight spans),
    # which makes stratification unstable and leaks needless label detail into
    # the partition generator.
    strata = validation.blended_risk.astype(str)
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(indices, strata, groups=validation.user_id))
    except ValueError:
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(indices, groups=validation.user_id))
    for train_idx, valid_idx in splits:
        train_users = set(validation.iloc[train_idx].user_id)
        valid_users = set(validation.iloc[valid_idx].user_id)
        if train_users & valid_users:
            raise AssertionError("inner user leakage detected")
    return splits


def _config_complexity(config: EvidenceSetConfig) -> tuple[float, ...]:
    return (
        config.top_k_indicator
        + config.top_k_ideation
        + config.top_k_behavior
        + config.top_k_attempt,
        -config.margin_threshold,
        config.event_gap_chars,
        config.length_penalty_per_token,
    )


def tune_config(
    audit: pd.DataFrame,
    validation: pd.DataFrame,
    row_ids: Sequence[str],
    grid: Sequence[EvidenceSetConfig],
    prediction_cache: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
) -> tuple[EvidenceSetConfig, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for config in grid:
        predictions = (
            prediction_cache[config.key]
            if prediction_cache is not None
            else decode_config(audit, validation, config, row_ids=row_ids)
        )
        metrics = score_predictions(validation, predictions, row_ids=row_ids)
        records.append({"config_key": config.key, **asdict(config), **metrics})
    table = pd.DataFrame(records)
    if table.empty:
        raise RuntimeError("empty calibration grid")
    # F1 is primary.  Ties prefer fewer predictions and then the simpler grid point.
    table = table.sort_values(
        ["f1", "mean_predicted_phrases", "margin_threshold"],
        ascending=[False, True, False],
        kind="mergesort",
    ).reset_index(drop=True)
    best_row = table.iloc[0]
    best = EvidenceSetConfig(
        margin_threshold=float(best_row.margin_threshold),
        event_gap_chars=int(best_row.event_gap_chars),
        length_penalty_per_token=float(best_row.length_penalty_per_token),
        top_k_indicator=int(best_row.top_k_indicator),
        top_k_ideation=int(best_row.top_k_ideation),
        top_k_behavior=int(best_row.top_k_behavior),
        top_k_attempt=int(best_row.top_k_attempt),
    )
    return best, table


def crossfit_calibration(
    audit: pd.DataFrame,
    validation: pd.DataFrame,
    output_dir: str | Path,
    grid: Sequence[EvidenceSetConfig] | None = None,
    n_splits: int = 3,
    seed: int = 20260824,
) -> dict[str, Any]:
    """Strict inner group-cross-fitted selection and deploy-config fitting."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = list(grid or default_config_grid())
    splits = make_inner_splits(validation, n_splits=n_splits, seed=seed)
    candidate_groups = {
        row_id: group for row_id, group in audit.groupby("row_id", sort=False)
    }
    print(f"[B4-E] precomputing {len(grid)} metric-aligned decoders")
    prediction_cache = {
        config.key: decode_config(
            audit, validation, config, candidate_groups=candidate_groups
        )
        for config in grid
    }
    crossfit_predictions: dict[str, list[str]] = {}
    selection_records: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []

    baseline_predictions = dict(zip(validation.row_id, validation.baseline_evidence))
    for inner_fold, (train_idx, valid_idx) in enumerate(splits):
        train_ids = validation.iloc[train_idx].row_id.tolist()
        valid_ids = validation.iloc[valid_idx].row_id.tolist()
        best, table = tune_config(
            audit, validation, train_ids, grid, prediction_cache=prediction_cache
        )
        table.insert(0, "inner_fold", inner_fold)
        table.to_csv(output_dir / f"inner_{inner_fold}_grid.csv", index=False)
        cached_predictions = prediction_cache[best.key]
        predictions = {row_id: cached_predictions[row_id] for row_id in valid_ids}
        crossfit_predictions.update(predictions)
        challenger = score_predictions(validation, predictions, valid_ids)
        baseline = score_predictions(validation, baseline_predictions, valid_ids)
        fold_records.append({
            "inner_fold": inner_fold,
            "selected_config": best.key,
            "challenger_f1": challenger["f1"],
            "baseline_f1": baseline["f1"],
            "delta_f1": challenger["f1"] - baseline["f1"],
            "challenger_precision": challenger["precision"],
            "challenger_recall": challenger["recall"],
            "rows": challenger["rows"],
        })
        selection_records.append({"inner_fold": inner_fold, **asdict(best), "config_key": best.key})
        print(
            f"[B4-E] inner={inner_fold} {best.key} "
            f"F1={challenger['f1']:.4f} delta={challenger['f1']-baseline['f1']:+.4f}"
        )

    if set(crossfit_predictions) != set(validation.row_id):
        missing = set(validation.row_id) - set(crossfit_predictions)
        raise AssertionError(f"crossfit predictions incomplete: {len(missing)} missing")
    crossfit_metrics = score_predictions(validation, crossfit_predictions)
    baseline_metrics = baseline_score(validation)

    deploy_config, deploy_grid = tune_config(
        audit, validation, validation.row_id.tolist(), grid,
        prediction_cache=prediction_cache,
    )
    deploy_grid.to_csv(output_dir / "deploy_grid_all_validation.csv", index=False)
    deploy_predictions = prediction_cache[deploy_config.key]
    deploy_metrics_in_sample = score_predictions(validation, deploy_predictions)

    fold_table = pd.DataFrame(fold_records)
    fold_table.to_csv(output_dir / "inner_fold_results.csv", index=False)
    pd.DataFrame(selection_records).to_csv(
        output_dir / "inner_selected_configs.csv", index=False
    )
    prediction_frame = validation[["row_id", "user_id", "blended_risk"]].copy()
    prediction_frame["gold_evidence"] = validation.gold_evidence_list.map(
        lambda value: json.dumps(value, ensure_ascii=False)
    )
    prediction_frame["baseline_evidence"] = validation.baseline_evidence.map(
        lambda value: json.dumps(value, ensure_ascii=False)
    )
    prediction_frame["crossfit_evidence"] = prediction_frame.row_id.map(
        lambda row_id: json.dumps(crossfit_predictions[row_id], ensure_ascii=False)
    )
    prediction_frame["deploy_config_evidence_in_sample"] = prediction_frame.row_id.map(
        lambda row_id: json.dumps(deploy_predictions[row_id], ensure_ascii=False)
    )
    prediction_frame.to_csv(output_dir / "crossfit_predictions.csv", index=False)

    delta = float(crossfit_metrics["f1"] - baseline_metrics["f1"])
    folds_better = int((fold_table.delta_f1 > 0).sum())
    accepted = bool(delta >= 0.010 and folds_better >= 2)
    decision = {
        "runtime_revision": B4E_RUNTIME_REVISION,
        "status": "complete",
        "accepted": accepted,
        "recommended_next_step": (
            "FREEZE_EVENT_SET_DECODER_AND_CONFIRM_OUTER_FOLDS_1_2"
            if accepted
            else "DO_NOT_RETRAIN_27B_YET_REVISE_EVENT_OBSERVER"
        ),
        "official_rule": {
            "symmetric_containment": True,
            "predicted_token_length_max_gold_multiple": 3,
            "one_to_one_matching": True,
            "per_post_macro_average": True,
        },
        "baseline": baseline_metrics,
        "crossfit": crossfit_metrics,
        "delta_crossfit_f1": delta,
        "inner_folds_better_than_baseline": folds_better,
        "inner_folds": fold_records,
        "deploy_config": asdict(deploy_config),
        "deploy_config_key": deploy_config.key,
        "deploy_metrics_in_sample_diagnostic_only": deploy_metrics_in_sample,
        "acceptance_rule": {
            "crossfit_delta_f1_min": 0.010,
            "inner_folds_better_min": 2,
        },
        "warning": (
            "The deploy-config metric is in-sample and diagnostic only. Cross-fitted "
            "Phrase-F1 is primary. Confirm the frozen decoder on outer folds 1 and 2."
        ),
    }
    (output_dir / "B4E_DECISION.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return decision


def boundary_invariance_variants(
    phrase: str,
    max_drop: int = 3,
    min_remaining_tokens: int = 2,
) -> list[str]:
    """Metric-aligned *positive* subspans for an optional later SFT run.

    The published scorer explicitly accepts faithful subspans.  These variants
    must therefore never be labelled as hard negatives.  Adjacent text outside
    Gold remains the appropriate source of boundary hard negatives.
    """

    phrase = str(phrase)
    token_spans = list(re.finditer(r"\S+", phrase))
    variants: list[str] = []
    for drop in range(1, max_drop + 1):
        if len(token_spans) - drop >= min_remaining_tokens:
            variants.append(phrase[token_spans[drop].start():].strip())
            variants.append(phrase[:token_spans[-drop - 1].end()].strip())
    seen: set[str] = set()
    return [
        item for item in variants
        if not (_normalize(item) in seen or seen.add(_normalize(item)))
    ]
