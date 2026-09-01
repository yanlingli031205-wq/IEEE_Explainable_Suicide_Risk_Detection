"""B4-E2: candidate-level meta calibration and adaptive evidence event decoding.

The 27B Task-1 model already proposes exact, verbatim evidence candidates and
assigns a Yes/No margin to each candidate.  B4-E2 does not train or reload the
27B model.  Instead, it learns a deliberately small calibration model that
answers a narrower question:

    Given the frozen 27B margin and observable candidate metadata, is this
    candidate likely to match one of the official Gold evidence phrases?

Candidate probabilities are converted into a variable-size evidence set by a
fixed probability threshold, literal de-duplication, spatial event suppression,
and risk-conditional safety caps.  All development metrics are produced by
user-grouped cross-fitting: the Gold evidence for a held-out post is never used
to fit the calibrator that predicts that post.

This module intentionally compares against the strong risk-consistent baseline
(``Indicator -> []``), rather than the weaker historical top-3 decoder.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import b1_experiments as b1
import b4e_evidence_set as b4e
import qwen38_dual_task_experiments as q38


B4E2_RUNTIME_REVISION = "2026-08-24.candidate-meta-event-decoder-v2"

CATEGORICAL_FEATURES = (
    "source",
    "blended_risk",
    "document_risk",
    "conditioned_risk",
)
NUMERIC_FEATURES = (
    "margin",
    "selection_score",
    "candidate_tokens",
    "candidate_chars",
    "row_candidate_count",
    "margin_rank",
    "margin_from_top",
    "baseline_count",
    "text_tokens",
    "text_chars",
    "left_fraction",
    "span_fraction",
)
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


@dataclass(frozen=True)
class CandidateMetaConfig:
    """Frozen B4-E2 model and set-decoder hyperparameters.

    The 0.40 probability threshold is intentionally fixed before any outer
    Fold-1/Fold-2 confirmation.  Re-tuning it on those folds invalidates them as
    confirmation sets.
    """

    logistic_c: float = 0.1
    probability_threshold: float = 0.40
    event_gap_chars: int = 40
    top_k_indicator: int = 0
    top_k_ideation: int = 2
    top_k_behavior: int = 3
    top_k_attempt: int = 2
    max_iter: int = 1000
    seed: int = 20260824

    def top_k(self, risk: str) -> int:
        return int({
            "indicator": self.top_k_indicator,
            "ideation": self.top_k_ideation,
            "behavior": self.top_k_behavior,
            "attempt": self.top_k_attempt,
        }.get(str(risk).strip().casefold(), self.top_k_behavior))


def _official_candidate_match(candidate: str, gold: str) -> bool:
    prediction = b4e._normalize(candidate)
    target = b4e._normalize(gold)
    if not prediction or not target:
        return False
    if len(prediction.split()) > 3 * max(1, len(target.split())):
        return False
    return prediction in target or target in prediction


def strong_baseline_predictions(
    validation: pd.DataFrame,
) -> dict[str, list[str]]:
    """Historical predictions plus the risk/evidence consistency constraint."""

    required = {"row_id", "blended_risk", "baseline_evidence"}
    missing = required - set(validation.columns)
    if missing:
        raise KeyError(f"strong baseline missing columns: {sorted(missing)}")
    return {
        str(row.row_id): (
            [] if str(row.blended_risk).strip().casefold() == "indicator"
            else list(row.baseline_evidence)
        )
        for row in validation.itertuples()
    }


def _text_maps(bundle: b1.DataBundle) -> tuple[dict[str, int], dict[str, int]]:
    token_map: dict[str, int] = {}
    char_map: dict[str, int] = {}
    for row_id, text in zip(bundle.row_ids, bundle.texts):
        row_id = str(row_id)
        text = str(text or "")
        token_map[row_id] = len(text.split())
        char_map[row_id] = len(text)
    return token_map, char_map


def build_candidate_meta_table(
    audit: pd.DataFrame,
    validation: pd.DataFrame,
    bundle: b1.DataBundle,
    include_targets: bool = True,
) -> pd.DataFrame:
    """Create candidate-level, test-time-observable calibration features.

    ``candidate_target`` is used only while fitting/evaluating on training
    folds.  Every other column is available at test time.
    """

    required_validation = {
        "row_id", "blended_risk", "document_risk", "conditioned_risk",
        "baseline_evidence",
    }
    missing_validation = required_validation - set(validation.columns)
    if missing_validation:
        raise KeyError(
            "candidate meta validation missing columns: "
            + repr(sorted(missing_validation))
        )

    table = audit.copy()
    if "selection_score" not in table.columns:
        table["selection_score"] = (
            table.margin.astype(float)
            - 0.002 * table.candidate_tokens.astype(float)
        )
    table["selection_score"] = pd.to_numeric(
        table.selection_score, errors="coerce"
    ).fillna(table.margin.astype(float))
    table["source"] = table.get("source", "unknown").fillna("unknown").astype(str)
    table["candidate_chars"] = (
        table.right.astype(int) - table.left.astype(int)
    ).clip(lower=1)
    table["row_candidate_count"] = (
        table.groupby("row_id").candidate.transform("size").astype(float)
    )
    table["margin_rank"] = table.groupby("row_id").margin.rank(
        method="first", ascending=False
    )
    table["margin_from_top"] = (
        table.groupby("row_id").margin.transform("max") - table.margin
    )

    validation_by_row = validation.set_index("row_id", drop=False)
    for column in ("blended_risk", "document_risk", "conditioned_risk"):
        table[column] = table.row_id.map(validation_by_row[column]).astype(str)
    baseline_counts = validation.baseline_evidence.map(len).astype(float)
    table["baseline_count"] = table.row_id.map(
        dict(zip(validation.row_id, baseline_counts))
    ).fillna(0.0)
    table["risk_agreement"] = (
        table.document_risk.eq(table.conditioned_risk).astype(float)
    )

    token_map, char_map = _text_maps(bundle)
    table["text_tokens"] = table.row_id.map(token_map).fillna(0).astype(float)
    table["text_chars"] = table.row_id.map(char_map).fillna(0).astype(float)
    safe_chars = table.text_chars.clip(lower=1.0)
    table["left_fraction"] = table.left.astype(float) / safe_chars
    table["span_fraction"] = table.candidate_chars.astype(float) / safe_chars

    if table[list(MODEL_FEATURES)].isna().any().any():
        bad = table[list(MODEL_FEATURES)].columns[
            table[list(MODEL_FEATURES)].isna().any()
        ].tolist()
        raise AssertionError(f"NaN candidate meta features: {bad}")

    if include_targets:
        if "gold_evidence_list" not in validation.columns:
            raise KeyError("gold_evidence_list is required when include_targets=True")
        gold_by_row = dict(zip(validation.row_id, validation.gold_evidence_list))
        table["candidate_target"] = [
            int(any(
                _official_candidate_match(candidate, gold)
                for gold in gold_by_row.get(str(row_id), [])
            ))
            for row_id, candidate in zip(table.row_id, table.candidate)
        ]
    return table.reset_index(drop=True)


def build_meta_pipeline(config: CandidateMetaConfig) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(CATEGORICAL_FEATURES),
            ),
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
        ],
        remainder="drop",
    )
    classifier = LogisticRegression(
        C=config.logistic_c,
        class_weight=None,
        max_iter=config.max_iter,
        random_state=config.seed,
        solver="lbfgs",
    )
    return Pipeline([
        ("preprocess", preprocess),
        ("classifier", classifier),
    ])


def fit_meta_model(
    candidate_table: pd.DataFrame,
    train_row_ids: Sequence[str],
    config: CandidateMetaConfig,
) -> Pipeline:
    wanted = set(map(str, train_row_ids))
    train = candidate_table[candidate_table.row_id.isin(wanted)]
    if train.empty:
        raise ValueError("no candidate rows available for meta-calibrator fitting")
    if "candidate_target" not in train.columns:
        raise KeyError("candidate_target is required for fitting")
    if train.candidate_target.nunique() < 2:
        raise ValueError("candidate meta train split contains only one target class")
    model = build_meta_pipeline(config)
    model.fit(train[list(MODEL_FEATURES)], train.candidate_target.astype(int))
    return model


def predict_candidate_probabilities(
    model: Pipeline,
    candidate_table: pd.DataFrame,
    row_ids: Sequence[str],
) -> pd.Series:
    wanted = set(map(str, row_ids))
    mask = candidate_table.row_id.isin(wanted)
    probabilities = pd.Series(np.nan, index=candidate_table.index, dtype=float)
    if mask.any():
        probabilities.loc[mask] = model.predict_proba(
            candidate_table.loc[mask, list(MODEL_FEATURES)]
        )[:, 1]
    return probabilities


def decode_meta_probabilities(
    candidate_table: pd.DataFrame,
    validation: pd.DataFrame,
    probabilities: pd.Series | np.ndarray,
    config: CandidateMetaConfig,
    row_ids: Sequence[str] | None = None,
) -> dict[str, list[str]]:
    """Convert candidate probabilities into a variable-cardinality event set."""

    table = candidate_table.copy()
    table["meta_probability"] = np.asarray(probabilities, dtype=float)
    wanted = set(
        validation.row_id.astype(str)
        if row_ids is None else map(str, row_ids)
    )
    risk_by_row = dict(zip(validation.row_id.astype(str), validation.blended_risk))
    groups = {
        str(row_id): group
        for row_id, group in table[table.row_id.isin(wanted)].groupby(
            "row_id", sort=False
        )
    }

    output: dict[str, list[str]] = {}
    for row_id in validation.row_id.astype(str):
        if row_id not in wanted:
            continue
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
            literal = b4e._normalize(item.candidate)
            if not literal or literal in selected_literals:
                continue
            left, right = int(item.left), int(item.right)
            if b4e._near_same_event(
                left, right, selected_spans, config.event_gap_chars
            ):
                continue
            selected.append(str(item.candidate).strip())
            selected_spans.append((left, right))
            selected_literals.add(literal)
            if len(selected) >= top_k:
                break
        output[row_id] = selected
    return output


def _stratified_metrics(
    validation: pd.DataFrame,
    systems: Mapping[str, Mapping[str, Sequence[str]]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    strata: list[tuple[str, pd.Series]] = [
        (f"risk={risk}", validation.blended_risk.eq(risk))
        for risk in b4e.RISK_ORDER
    ]
    strata.extend([
        ("gold_count=0", validation.gold_count.eq(0)),
        ("gold_count=1", validation.gold_count.eq(1)),
        ("gold_count=2", validation.gold_count.eq(2)),
        ("gold_count>=3", validation.gold_count.ge(3)),
    ])
    for stratum, mask in strata:
        ids = validation.loc[mask, "row_id"].astype(str).tolist()
        for system, predictions in systems.items():
            metrics = b4e.score_predictions(validation, predictions, ids)
            rows.append({"stratum": stratum, "system": system, **metrics})
    return pd.DataFrame(rows)


def crossfit_candidate_meta_decoder(
    audit: pd.DataFrame,
    validation: pd.DataFrame,
    bundle: b1.DataBundle,
    output_dir: str | Path,
    config: CandidateMetaConfig | None = None,
    n_splits: int = 3,
    split_seed: int = 20260824,
) -> dict[str, Any]:
    """Strict user-grouped cross-fit plus a deploy model fitted on all rows."""

    config = config or CandidateMetaConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_table = build_candidate_meta_table(
        audit, validation, bundle, include_targets=True
    )
    splits = b4e.make_inner_splits(
        validation, n_splits=n_splits, seed=split_seed
    )
    baseline_predictions = strong_baseline_predictions(validation)
    crossfit_predictions: dict[str, list[str]] = {}
    fold_records: list[dict[str, Any]] = []
    oof_probabilities = pd.Series(
        np.nan, index=candidate_table.index, dtype=float
    )

    for inner_fold, (train_idx, valid_idx) in enumerate(splits):
        train_ids = validation.iloc[train_idx].row_id.astype(str).tolist()
        valid_ids = validation.iloc[valid_idx].row_id.astype(str).tolist()
        model = fit_meta_model(candidate_table, train_ids, config)
        probabilities = predict_candidate_probabilities(
            model, candidate_table, valid_ids
        )
        valid_mask = candidate_table.row_id.isin(valid_ids)
        oof_probabilities.loc[valid_mask] = probabilities.loc[valid_mask]
        predictions = decode_meta_probabilities(
            candidate_table,
            validation,
            probabilities,
            config,
            row_ids=valid_ids,
        )
        crossfit_predictions.update(predictions)
        challenger = b4e.score_predictions(validation, predictions, valid_ids)
        baseline = b4e.score_predictions(
            validation, baseline_predictions, valid_ids
        )
        record = {
            "inner_fold": inner_fold,
            "rows": len(valid_ids),
            "candidate_rows": int(valid_mask.sum()),
            "challenger_f1": challenger["f1"],
            "baseline_f1": baseline["f1"],
            "delta_f1": challenger["f1"] - baseline["f1"],
            "challenger_precision": challenger["precision"],
            "challenger_recall": challenger["recall"],
            "baseline_precision": baseline["precision"],
            "baseline_recall": baseline["recall"],
        }
        fold_records.append(record)
        print(
            f"[B4-E2] inner={inner_fold} F1={challenger['f1']:.4f} "
            f"strong-baseline={baseline['f1']:.4f} "
            f"delta={record['delta_f1']:+.4f}"
        )

    if set(crossfit_predictions) != set(validation.row_id.astype(str)):
        missing = set(validation.row_id.astype(str)) - set(crossfit_predictions)
        raise AssertionError(f"incomplete meta crossfit predictions: {len(missing)}")
    if oof_probabilities.isna().any():
        raise AssertionError("candidate meta OOF probabilities are incomplete")

    baseline_metrics = b4e.score_predictions(
        validation, baseline_predictions
    )
    crossfit_metrics = b4e.score_predictions(
        validation, crossfit_predictions
    )
    fold_table = pd.DataFrame(fold_records)
    fold_table.to_csv(output_dir / "B4E2_INNER_FOLD_RESULTS.csv", index=False)

    prediction_frame = validation[
        ["row_id", "user_id", "blended_risk"]
    ].copy()
    prediction_frame["gold_evidence"] = validation.gold_evidence_list.map(
        lambda value: json.dumps(value, ensure_ascii=False)
    )
    for name, predictions in (
        ("strong_baseline_evidence", baseline_predictions),
        ("crossfit_meta_evidence", crossfit_predictions),
    ):
        prediction_frame[name] = prediction_frame.row_id.map(
            lambda row_id, p=predictions: json.dumps(
                list(p[str(row_id)]), ensure_ascii=False
            )
        )
    prediction_frame.to_csv(
        output_dir / "B4E2_CROSSFIT_PREDICTIONS.csv", index=False
    )

    candidate_oof = candidate_table[
        ["row_id", "candidate", "left", "right", "margin", "candidate_target"]
    ].copy()
    candidate_oof["meta_oof_probability"] = oof_probabilities
    candidate_oof.to_csv(
        output_dir / "B4E2_CANDIDATE_OOF_AUDIT.csv", index=False
    )

    strata = _stratified_metrics(
        validation,
        {
            "INDICATOR_EMPTY_STRONG_BASELINE": baseline_predictions,
            "B4E2_CANDIDATE_META": crossfit_predictions,
        },
    )
    strata.to_csv(output_dir / "B4E2_STRATIFIED_METRICS.csv", index=False)

    # Fold-0 Gold is allowed to fit this deploy calibrator.  Its own fitted
    # metric is diagnostic only; the artifact is for untouched outer folds.
    deploy_model = fit_meta_model(
        candidate_table, validation.row_id.astype(str).tolist(), config
    )
    joblib.dump(deploy_model, output_dir / "B4E2_FOLD0_META_CALIBRATOR.joblib")
    (output_dir / "B4E2_FROZEN_CONFIG.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    delta = float(crossfit_metrics["f1"] - baseline_metrics["f1"])
    folds_better = int((fold_table.delta_f1 > 0).sum())
    accepted = bool(
        crossfit_metrics["f1"] >= 0.747
        and delta >= 0.015
        and folds_better >= 2
    )
    decision = {
        "runtime_revision": B4E2_RUNTIME_REVISION,
        "status": "complete",
        "accepted_for_outer_confirmation": accepted,
        "recommended_next_step": (
            "FREEZE_META_DECODER_AND_CONFIRM_OUTER_FOLDS_1_2"
            if accepted else
            "KEEP_INDICATOR_EMPTY_BASELINE_AND_REVISE_META_DECODER"
        ),
        "strong_baseline": baseline_metrics,
        "candidate_meta_crossfit": crossfit_metrics,
        "delta_crossfit_f1_vs_strong_baseline": delta,
        "inner_folds_better_than_strong_baseline": folds_better,
        "inner_folds": fold_records,
        "frozen_config": asdict(config),
        "acceptance_rule": {
            "crossfit_evidence_f1_min": 0.747,
            "delta_vs_indicator_empty_strong_baseline_min": 0.015,
            "inner_folds_better_min": 2,
        },
        "scientific_interpretation": (
            "The candidate meta-calibrator estimates event relevance; its fixed "
            "probability threshold acts as an adaptive evidence-count observer."
        ),
        "warning": (
            "Fold-0 was used for model development. Do not retune threshold, "
            "features, event gap, or risk caps after viewing outer Fold-1/Fold-2."
        ),
    }
    (output_dir / "B4E2_DECISION.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return decision


def apply_frozen_meta_decoder(
    model_path: str | Path,
    config_path: str | Path,
    audit: pd.DataFrame,
    validation: pd.DataFrame,
    bundle: b1.DataBundle,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    """Apply a frozen Fold-0 calibrator to an untouched fold or test set."""

    model = joblib.load(model_path)
    config = CandidateMetaConfig(**json.loads(Path(config_path).read_text()))
    candidate_table = build_candidate_meta_table(
        audit, validation, bundle, include_targets=False
    )
    probabilities = pd.Series(
        model.predict_proba(candidate_table[list(MODEL_FEATURES)])[:, 1],
        index=candidate_table.index,
    )
    predictions = decode_meta_probabilities(
        candidate_table, validation, probabilities, config
    )
    candidate_table = candidate_table.copy()
    candidate_table["meta_probability"] = probabilities
    return predictions, candidate_table


def _risk_ids(values: Sequence[Any]) -> np.ndarray:
    output: list[int] = []
    for value in values:
        key = str(value).strip().casefold()
        if key not in b1.RISK_TO_ID:
            raise ValueError(f"unknown risk label in outer confirmation: {value!r}")
        output.append(int(b1.RISK_TO_ID[key]))
    return np.asarray(output, dtype=np.int64)


def confirm_frozen_meta_decoder(
    fold_artifacts: Mapping[int, tuple[str | Path, str | Path]],
    bundle: b1.DataBundle,
    folds: np.ndarray,
    model_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    factor_macro_f1: float = 0.691954,
    rank8_reference: float = 0.7615,
    save_candidate_audits: bool = True,
) -> dict[str, Any]:
    """Apply one frozen Fold-0 calibrator to predeclared OOF fold artifacts.

    ``fold_artifacts`` maps an outer fold id to
    ``(evidence_candidate_audit.csv, validation_predictions.csv)``.  The base
    Task-1 adapter that produced each pair must exclude that fold.  This
    function never searches a threshold, feature, event gap, or risk cap.

    Methodological boundary: the Fold-0 development candidate margins came
    from a base adapter trained on Folds 1/2.  Consequently, this is useful
    cross-fitted competition confirmation but not a fully nested external-test
    estimate.  The saved decision states that limitation explicitly.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = np.asarray(folds, dtype=int)
    if len(folds) != len(bundle.row_ids):
        raise ValueError("fold vector and bundle length differ")
    row_to_index = {
        str(row_id): index for index, row_id in enumerate(bundle.row_ids)
    }

    fold_records: list[dict[str, Any]] = []
    validation_frames: list[pd.DataFrame] = []
    meta_predictions_all: dict[str, list[str]] = {}
    baseline_predictions_all: dict[str, list[str]] = {}
    prediction_frames: list[pd.DataFrame] = []

    for fold in sorted(fold_artifacts):
        audit_path, prediction_path = map(Path, fold_artifacts[fold])
        if not audit_path.exists() or not prediction_path.exists():
            raise FileNotFoundError(
                f"outer fold {fold} artifacts missing: {audit_path}, {prediction_path}"
            )
        audit_raw = pd.read_csv(audit_path)
        validation_raw = pd.read_csv(prediction_path)
        audit, validation = b4e.prepare_artifacts(
            audit_raw, validation_raw, bundle
        )
        row_indices = np.asarray(
            [row_to_index[str(row_id)] for row_id in validation.row_id],
            dtype=int,
        )
        if not np.all(folds[row_indices] == int(fold)):
            raise AssertionError(
                f"artifact rows do not belong exclusively to outer fold {fold}"
            )
        expected_ids = set(bundle.row_ids[folds == int(fold)].astype(str))
        actual_ids = set(validation.row_id.astype(str))
        if actual_ids != expected_ids:
            raise AssertionError(
                f"outer fold {fold} coverage mismatch: "
                f"missing={len(expected_ids-actual_ids)}, extra={len(actual_ids-expected_ids)}"
            )

        meta_predictions, candidate_meta = apply_frozen_meta_decoder(
            model_path, config_path, audit, validation, bundle
        )
        baseline_predictions = strong_baseline_predictions(validation)
        meta_metrics = b4e.score_predictions(validation, meta_predictions)
        baseline_metrics = b4e.score_predictions(
            validation, baseline_predictions
        )
        gold_risk = _risk_ids(validation.gold_risk)
        predicted_risk = _risk_ids(validation.blended_risk)
        risk_weighted_f1 = float(
            f1_score(
                gold_risk, predicted_risk,
                average="weighted", zero_division=0,
            )
        )
        fold_record = {
            "outer_fold": int(fold),
            "rows": int(len(validation)),
            "risk_weighted_f1": risk_weighted_f1,
            "strong_baseline_evidence_f1": baseline_metrics["f1"],
            "meta_evidence_f1": meta_metrics["f1"],
            "delta_evidence_f1": meta_metrics["f1"] - baseline_metrics["f1"],
            "meta_evidence_precision": meta_metrics["precision"],
            "meta_evidence_recall": meta_metrics["recall"],
            "meta_mean_predicted_phrases": meta_metrics["mean_predicted_phrases"],
        }
        fold_records.append(fold_record)
        print(
            f"[B4-E2 confirm] outer={fold} Evidence F1={meta_metrics['f1']:.4f} "
            f"strong-baseline={baseline_metrics['f1']:.4f} "
            f"delta={fold_record['delta_evidence_f1']:+.4f}"
        )

        if save_candidate_audits:
            candidate_meta.to_csv(
                output_dir / f"fold_{fold}_candidate_meta_audit.csv", index=False
            )
        prediction_frame = validation[
            ["row_id", "user_id", "gold_risk", "blended_risk"]
        ].copy()
        prediction_frame["outer_fold"] = int(fold)
        prediction_frame["gold_evidence"] = validation.gold_evidence_list.map(
            lambda value: json.dumps(value, ensure_ascii=False)
        )
        prediction_frame["strong_baseline_evidence"] = prediction_frame.row_id.map(
            lambda row_id: json.dumps(
                baseline_predictions[str(row_id)], ensure_ascii=False
            )
        )
        prediction_frame["meta_evidence"] = prediction_frame.row_id.map(
            lambda row_id: json.dumps(
                meta_predictions[str(row_id)], ensure_ascii=False
            )
        )
        prediction_frames.append(prediction_frame)
        validation_frames.append(validation)
        meta_predictions_all.update(meta_predictions)
        baseline_predictions_all.update(baseline_predictions)

    if not fold_records:
        raise ValueError("no outer fold artifacts were supplied")
    combined_validation = pd.concat(validation_frames, ignore_index=True)
    combined_meta = b4e.score_predictions(
        combined_validation, meta_predictions_all
    )
    combined_baseline = b4e.score_predictions(
        combined_validation, baseline_predictions_all
    )
    combined_gold_risk = _risk_ids(combined_validation.gold_risk)
    combined_predicted_risk = _risk_ids(combined_validation.blended_risk)
    combined_risk_wf1 = float(
        f1_score(
            combined_gold_risk, combined_predicted_risk,
            average="weighted", zero_division=0,
        )
    )
    composite = float(
        0.4 * combined_risk_wf1
        + 0.3 * combined_meta["f1"]
        + 0.3 * float(factor_macro_f1)
    )
    baseline_composite = float(
        0.4 * combined_risk_wf1
        + 0.3 * combined_baseline["f1"]
        + 0.3 * float(factor_macro_f1)
    )
    fold_table = pd.DataFrame(fold_records)
    fold_table.to_csv(output_dir / "B4E2_OUTER_FOLD_RESULTS.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output_dir / "B4E2_OUTER_PREDICTIONS.csv", index=False
    )

    pooled_delta = float(combined_meta["f1"] - combined_baseline["f1"])
    minimum_fold_delta = float(fold_table.delta_evidence_f1.min())
    folds_better = int((fold_table.delta_evidence_f1 > 0).sum())
    decoder_confirmed = bool(
        combined_meta["f1"] >= 0.745
        and pooled_delta >= 0.010
        and minimum_fold_delta >= -0.010
        and folds_better >= 1
    )
    top8_projection_passed = bool(composite >= float(rank8_reference))
    accepted = bool(decoder_confirmed and top8_projection_passed)
    decision = {
        "runtime_revision": B4E2_RUNTIME_REVISION,
        "status": "complete",
        "confirmed_folds": sorted(map(int, fold_artifacts)),
        "candidate_audits_saved": bool(save_candidate_audits),
        "frozen_model_path": str(model_path),
        "frozen_config": json.loads(Path(config_path).read_text(encoding="utf-8")),
        "outer_fold_results": fold_records,
        "pooled": {
            "rows": int(len(combined_validation)),
            "risk_weighted_f1": combined_risk_wf1,
            "strong_baseline_evidence": combined_baseline,
            "meta_evidence": combined_meta,
            "delta_evidence_f1": pooled_delta,
            "factor_macro_f1_reference": float(factor_macro_f1),
            "strong_baseline_composite_projection": baseline_composite,
            "meta_composite_projection": composite,
            "rank8_reference": float(rank8_reference),
        },
        "decoder_confirmed": decoder_confirmed,
        "top8_projection_passed": top8_projection_passed,
        "accepted_for_final_test": accepted,
        "recommended_next_step": (
            "REFIT_FIXED_META_ON_ALL_THREE_FOLD_OOF_AND_BUILD_TEST_SUBMISSION"
            if accepted else
            "KEEP_STRONG_BASELINE_OR_REVISIT_MULTI_EVENT_RECALL"
        ),
        "pre_registered_confirmation_rule": {
            "pooled_meta_evidence_f1_min": 0.745,
            "pooled_delta_vs_strong_baseline_min": 0.010,
            "minimum_single_fold_delta_min": -0.010,
            "folds_better_min": 1,
            "composite_projection_min": float(rank8_reference),
        },
        "methodological_warning": (
            "This is row-level OOF and valid for competition model selection, but "
            "not a fully nested external-test estimate: the Fold-0 development "
            "candidate generator was trained on Folds 1/2. Do not describe these "
            "folds as an untouched external test in a paper."
        ),
    }
    (output_dir / "B4E2_OUTER_CONFIRMATION_DECISION.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return decision
