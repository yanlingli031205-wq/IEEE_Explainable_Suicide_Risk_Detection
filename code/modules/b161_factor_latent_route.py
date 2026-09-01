"""B16.1: frozen seven-label latent-readout confirmation for Task 2.

The route was fixed after the B16 Fold-0 screen.  Folds 1 and 2 are used only
for confirmation: no layer, C, label, alpha, or threshold rule is selected on
their held-out labels.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

import b1_experiments as b1
import b4p_anchor_verifier as b4
import b16_factor_latent_readout as b16


B161_RUNTIME_REVISION = "2026-08-31.factor-latent-seven-label-confirmation-v3"

ROUTE_LABELS = (
    "psychological capital",
    "social support",
    "poor school performance",
    "sense of responsibility",
    "traumatic experience",
    "low socio-economic status",
    "interpersonal difficulty",
)


@dataclass
class RouteConfirmationConfig:
    selected_layer: int = 63
    c_value: float = 0.001
    route_labels: tuple[str, ...] = ROUTE_LABELS
    inner_splits: int = 3
    extraction_batch_size: int = 2
    extraction_chunk_size: int = 128
    seed: int = 42
    pooled_macro_f1_gain_min: float = 0.006
    pooled_macro_ap_gain_min: float = 0.006
    pooled_tail_gain_floor: float = -0.005
    each_fold_macro_f1_gain_strictly_positive: bool = True


def _json_default(value: Any) -> Any:
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


def _route_ids(config: RouteConfirmationConfig) -> np.ndarray:
    unknown = set(config.route_labels) - set(b1.FACTOR_LABELS)
    if unknown:
        raise ValueError(f"Unknown route labels: {sorted(unknown)}")
    ids = np.asarray([b1.FACTOR_LABELS.index(label) for label in config.route_labels], dtype=int)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate B16.1 route labels")
    # ``build_prompt_table`` always emits labels in the official taxonomy
    # order.  The research-facing route tuple is grouped for readability and
    # is not in that order, so canonicalize before reshaping pair scores.
    return np.sort(ids)


def build_reduced_prompt_frame(
    bundle: b1.DataBundle,
    folds: np.ndarray,
    semantic_cache: b4.SemanticCache,
    verifier_config: b4.B4PConfig,
    fold: int,
    route_ids: np.ndarray,
) -> pd.DataFrame:
    """Keep all outer-train labels but only routed outer-valid labels."""
    frame = b16.build_all_prompts(bundle, folds, semantic_cache, verifier_config, fold)
    is_valid = np.asarray(folds, dtype=int)[frame.query_row_idx.astype(int)] == int(fold)
    keep = ~is_valid | frame.label_idx.astype(int).isin(set(map(int, route_ids)))
    reduced = frame.loc[keep].reset_index(drop=True)
    train_rows = int((np.asarray(folds) != int(fold)).sum())
    valid_rows = int((np.asarray(folds) == int(fold)).sum())
    expected = train_rows * len(b1.FACTOR_LABELS) + valid_rows * len(route_ids)
    if len(reduced) != expected:
        raise AssertionError(f"Expected {expected} reduced prompts, got {len(reduced)}")
    return reduced


def _pair_features(
    hidden: np.ndarray,
    margins: np.ndarray,
    label_ids: np.ndarray,
) -> np.ndarray:
    hidden = np.asarray(hidden[:, 0, :], dtype=np.float32)
    margins = np.asarray(margins, dtype=np.float32).reshape(-1, 1)
    one_hot = np.eye(len(b1.FACTOR_LABELS), dtype=np.float32)[np.asarray(label_ids, dtype=int)]
    return np.concatenate([hidden, margins, one_hot], axis=1)


def _canonical_pair_indices(
    query_ids: np.ndarray,
    label_ids: np.ndarray,
    row_ids: np.ndarray,
    requested_label_ids: np.ndarray,
) -> np.ndarray:
    """Return pair indices in explicit row-major taxonomy order.

    Prompt generation, pandas filtering, and cached chunk concatenation are not
    required to preserve the same incidental order.  Downstream reshaping does,
    so align by the stable (query row, label) key instead of asserting whatever
    order the prompt frame happened to use.
    """
    query_ids = np.asarray(query_ids, dtype=int)
    label_ids = np.asarray(label_ids, dtype=int)
    row_ids = np.asarray(row_ids, dtype=int)
    requested_label_ids = np.asarray(requested_label_ids, dtype=int)
    if query_ids.shape != label_ids.shape:
        raise ValueError("B16.1 query/label key arrays have different shapes")

    pair_lookup: dict[tuple[int, int], int] = {}
    for pair_index, (query_id, label_id) in enumerate(zip(query_ids, label_ids)):
        key = (int(query_id), int(label_id))
        if key in pair_lookup:
            raise AssertionError(f"B16.1 duplicate prompt pair: {key}")
        pair_lookup[key] = int(pair_index)

    requested = [
        (int(row_id), int(label_id))
        for row_id in row_ids
        for label_id in requested_label_ids
    ]
    missing = [key for key in requested if key not in pair_lookup]
    if missing:
        raise AssertionError(
            f"B16.1 missing {len(missing)} prompt pairs; first={missing[:5]}"
        )
    return np.asarray([pair_lookup[key] for key in requested], dtype=int)


def _score_route(model: Any, features: np.ndarray, rows: int, labels: int) -> np.ndarray:
    values = np.asarray(model.decision_function(features), dtype=np.float32)
    return values.reshape(int(rows), int(labels))


def _fixed_crossfit_scores(
    x_train: np.ndarray,
    targets: np.ndarray,
    train_rows: np.ndarray,
    user_ids: np.ndarray,
    config: RouteConfirmationConfig,
) -> np.ndarray:
    train_rows = np.asarray(train_rows, dtype=int)
    groups = np.asarray(user_ids).astype(str)[train_rows]
    splits = min(int(config.inner_splits), len(np.unique(groups)))
    splitter = GroupKFold(n_splits=splits)
    output = np.full((len(train_rows), len(b1.FACTOR_LABELS)), np.nan, dtype=np.float32)
    labels = len(b1.FACTOR_LABELS)
    for split_id, (inner_train, inner_valid) in enumerate(
        splitter.split(train_rows, groups=groups)
    ):
        row_train = train_rows[inner_train]
        pair_train = np.concatenate(
            [np.arange(index * labels, (index + 1) * labels) for index in inner_train]
        )
        pair_valid = np.concatenate(
            [np.arange(index * labels, (index + 1) * labels) for index in inner_valid]
        )
        probe = b16._make_probe(config.c_value, config.seed + split_id)
        b16._fit_probe(probe, x_train[pair_train], targets, row_train)
        output[inner_valid] = b16._score_matrix(
            probe, x_train[pair_valid], len(inner_valid)
        )
    if not np.isfinite(output).all():
        raise ValueError("Incomplete fixed B16.1 inner OOF scores")
    return output


def _metric_ap(target: np.ndarray, probability: np.ndarray) -> float:
    return float(
        np.mean(
            [
                average_precision_score(target[:, label], probability[:, label])
                for label in range(target.shape[1])
                if np.unique(target[:, label]).size == 2
            ]
        )
    )


def run_confirmation_fold(
    bundle: b1.DataBundle,
    anchor: dict[str, np.ndarray],
    semantic_cache: b4.SemanticCache,
    verifier_config: b4.B4PConfig,
    adapter_path: str | Path,
    output_dir: str | Path,
    fold: int,
    config: RouteConfirmationConfig,
) -> dict[str, Any]:
    if int(fold) not in {1, 2}:
        raise ValueError("B16.1 confirmation is reserved for untouched Folds 1 and 2")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    route_ids = _route_ids(config)
    folds, targets = anchor["folds"], anchor["targets"]
    train_rows = np.flatnonzero(folds != int(fold))
    valid_rows = np.flatnonzero(folds == int(fold))

    frame = build_reduced_prompt_frame(
        bundle, folds, semantic_cache, verifier_config, fold, route_ids
    )
    frame.drop(columns=["prompt"]).to_csv(output_dir / "B161_PROMPT_AUDIT.csv", index=False)
    extraction_config = b16.FactorLatentConfig(
        fold=int(fold),
        selected_layers=(int(config.selected_layer),),
        c_grid=(float(config.c_value),),
        blend_alphas=(1.0,),
        inner_splits=int(config.inner_splits),
        extraction_batch_size=int(config.extraction_batch_size),
        extraction_chunk_size=int(config.extraction_chunk_size),
        seed=int(config.seed),
    )
    extraction = b16.extract_factor_hidden(
        frame, adapter_path, verifier_config, output_dir, extraction_config
    )
    margins = np.asarray(extraction["margins"], dtype=np.float32)
    hidden = np.asarray(extraction["hidden"], dtype=np.float16)
    query = frame.query_row_idx.astype(int).to_numpy()
    label = frame.label_idx.astype(int).to_numpy()
    x_all = _pair_features(hidden, margins, label)
    train_pair_indices = _canonical_pair_indices(
        query,
        label,
        train_rows,
        np.arange(len(b1.FACTOR_LABELS), dtype=int),
    )
    valid_pair_indices = _canonical_pair_indices(
        query,
        label,
        valid_rows,
        route_ids,
    )
    x_train = x_all[train_pair_indices]
    x_valid = x_all[valid_pair_indices]

    valid_margin = margins[valid_pair_indices].reshape(len(valid_rows), len(route_ids))
    reproduction_mae = float(
        np.mean(
            np.abs(
                b4.sigmoid(valid_margin)
                - anchor["q38_probability"][valid_rows][:, route_ids]
            )
        )
    )
    if reproduction_mae > 0.01:
        raise AssertionError(f"B16.1 Q38 margin reproduction failed: {reproduction_mae:.5f}")

    inner_scores = _fixed_crossfit_scores(
        x_train, targets, train_rows, bundle.user_ids, config
    )
    probe = b16._make_probe(config.c_value, config.seed + int(fold))
    b16._fit_probe(probe, x_train, targets, train_rows)
    joblib.dump(probe, output_dir / "B161_PROBE.joblib")
    valid_route_scores = _score_route(
        probe, x_valid, len(valid_rows), len(route_ids)
    )

    base_train = anchor["anchor_probability"][train_rows]
    base_valid = anchor["anchor_probability"][valid_rows]
    routed_train, routed_valid = base_train.copy(), base_valid.copy()
    routed_train[:, route_ids] = b4.sigmoid(inner_scores[:, route_ids])
    routed_valid[:, route_ids] = b4.sigmoid(valid_route_scores)
    threshold_config = b4.B4PConfig(
        threshold_kappa_tail=0.0,
        threshold_kappa_mid=2.0,
        threshold_kappa_head=2.0,
        seed=config.seed,
    )
    base_thresholds = b4.fit_factor_thresholds(
        base_train, targets[train_rows], threshold_config
    )
    route_thresholds = b4.fit_factor_thresholds(
        routed_train, targets[train_rows], threshold_config
    )
    base_prediction = (base_valid >= base_thresholds).astype(np.int8)
    route_prediction = (routed_valid >= route_thresholds).astype(np.int8)
    base_metrics, base_table = b4.factor_metric_bundle(
        targets[valid_rows], base_valid, base_prediction
    )
    route_metrics, route_table = b4.factor_metric_bundle(
        targets[valid_rows], routed_valid, route_prediction
    )
    base_table.insert(0, "system", "Q14_25_Q38_75")
    route_table.insert(0, "system", "B161_SEVEN_LABEL_ROUTE")
    pd.concat([base_table, route_table], ignore_index=True).to_csv(
        output_dir / "B161_PER_LABEL.csv", index=False
    )
    summary = pd.DataFrame(
        [
            {"system": "Q14_25_Q38_75", **base_metrics},
            {"system": "B161_SEVEN_LABEL_ROUTE", **route_metrics},
        ]
    )
    summary.to_csv(output_dir / "B161_FOLD_SUMMARY.csv", index=False)
    np.savez_compressed(
        output_dir / "B161_FOLD_OUTPUTS.npz",
        row_ids=bundle.row_ids,
        folds=folds,
        valid_rows=valid_rows,
        route_ids=route_ids,
        targets=targets,
        base_probability=base_valid,
        base_prediction=base_prediction,
        route_probability=routed_valid,
        route_prediction=route_prediction,
        inner_probe_scores=inner_scores,
        valid_route_scores=valid_route_scores,
        base_thresholds=base_thresholds,
        route_thresholds=route_thresholds,
    )
    decision = {
        "runtime_revision": B161_RUNTIME_REVISION,
        "fold": int(fold),
        "rows": int(len(valid_rows)),
        "selected_layer": int(config.selected_layer),
        "c_value": float(config.c_value),
        "route_labels": list(config.route_labels),
        "q38_margin_reproduction_mae": reproduction_mae,
        "baseline": base_metrics,
        "route": route_metrics,
        "deltas": {
            "macro_f1": route_metrics["macro_f1"] - base_metrics["macro_f1"],
            "macro_ap": route_metrics["macro_ap"] - base_metrics["macro_ap"],
            "tail_macro_f1": route_metrics["tail_macro_f1"] - base_metrics["tail_macro_f1"],
        },
        "status": "FOLD_COMPLETE",
    }
    json_dump(decision, output_dir / "B161_FOLD_DECISION.json")
    return decision


def aggregate_confirmation(
    bundle: b1.DataBundle,
    output_root: str | Path,
    config: RouteConfirmationConfig,
) -> dict[str, Any]:
    output_root = Path(output_root)
    targets = bundle.factor_binary.astype(np.int8)
    base_probability = np.full_like(targets, np.nan, dtype=np.float32)
    route_probability = np.full_like(targets, np.nan, dtype=np.float32)
    base_prediction = np.zeros_like(targets, dtype=np.int8)
    route_prediction = np.zeros_like(targets, dtype=np.int8)
    confirmation = np.zeros(len(targets), dtype=bool)
    fold_decisions: list[dict[str, Any]] = []
    for fold in (1, 2):
        directory = output_root / f"fold_{fold}"
        decision_path = directory / "B161_FOLD_DECISION.json"
        output_path = directory / "B161_FOLD_OUTPUTS.npz"
        if not decision_path.exists() or not output_path.exists():
            raise FileNotFoundError(f"B16.1 Fold {fold} is incomplete")
        fold_decisions.append(json.loads(decision_path.read_text(encoding="utf-8")))
        saved = np.load(output_path, allow_pickle=True)
        if saved["row_ids"].astype(str).tolist() != bundle.row_ids.astype(str).tolist():
            raise AssertionError(f"B16.1 row mismatch in Fold {fold}")
        valid = saved["valid_rows"].astype(int)
        confirmation[valid] = True
        base_probability[valid] = saved["base_probability"]
        route_probability[valid] = saved["route_probability"]
        base_prediction[valid] = saved["base_prediction"]
        route_prediction[valid] = saved["route_prediction"]
    rows = np.flatnonzero(confirmation)
    base_metrics, _ = b4.factor_metric_bundle(
        targets[rows], base_probability[rows], base_prediction[rows]
    )
    route_metrics, _ = b4.factor_metric_bundle(
        targets[rows], route_probability[rows], route_prediction[rows]
    )
    deltas = {
        "macro_f1": route_metrics["macro_f1"] - base_metrics["macro_f1"],
        "macro_ap": route_metrics["macro_ap"] - base_metrics["macro_ap"],
        "tail_macro_f1": route_metrics["tail_macro_f1"] - base_metrics["tail_macro_f1"],
    }
    each_positive = all(
        float(item["deltas"]["macro_f1"]) > 0 for item in fold_decisions
    )
    passed = bool(
        each_positive
        and deltas["macro_f1"] >= config.pooled_macro_f1_gain_min
        and deltas["macro_ap"] >= config.pooled_macro_ap_gain_min
        and deltas["tail_macro_f1"] >= config.pooled_tail_gain_floor
    )
    summary = pd.DataFrame(
        [
            {"scope": "POOLED_FOLDS_1_2", "system": "Q14_25_Q38_75", **base_metrics},
            {"scope": "POOLED_FOLDS_1_2", "system": "B161_SEVEN_LABEL_ROUTE", **route_metrics},
            *[
                {
                    "scope": f"FOLD_{item['fold']}",
                    "system": "B161_SEVEN_LABEL_ROUTE",
                    **item["route"],
                    "delta_macro_f1": item["deltas"]["macro_f1"],
                    "delta_macro_ap": item["deltas"]["macro_ap"],
                }
                for item in fold_decisions
            ],
        ]
    )
    summary.to_csv(output_root / "B161_CONFIRMATION_SUMMARY.csv", index=False)
    decision = {
        "runtime_revision": B161_RUNTIME_REVISION,
        "status": "complete",
        "passed": passed,
        "decision": "BUILD_B161_TEST_ROUTE" if passed else "STOP_B161",
        "route_labels": list(config.route_labels),
        "selected_layer": config.selected_layer,
        "c_value": config.c_value,
        "confirmation_folds": [1, 2],
        "each_fold_positive": each_positive,
        "baseline": base_metrics,
        "route": route_metrics,
        "deltas": deltas,
        "fold_decisions": fold_decisions,
        "acceptance_rule": {
            "pooled_macro_f1_gain_min": config.pooled_macro_f1_gain_min,
            "pooled_macro_ap_gain_min": config.pooled_macro_ap_gain_min,
            "pooled_tail_gain_floor": config.pooled_tail_gain_floor,
            "each_fold_macro_f1_gain_strictly_positive": True,
        },
        "evaluation": "Untouched grouped outer Folds 1/2; route/layer/C fixed after Fold 0.",
    }
    json_dump(decision, output_root / "B161_CONFIRMATION_DECISION.json")
    return decision
