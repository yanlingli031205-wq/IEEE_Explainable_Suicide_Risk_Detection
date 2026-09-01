"""Three-fold Qwen3.8 Full64 completion and Factor OOF decision utilities.

The expensive model work lives in ``qwen38_dual_task_experiments``.  This
module is intentionally CPU-only: it validates and assembles the three fold
artifacts, evaluates Qwen3-14B and Qwen3.8-27B on the identical grouped folds,
and builds a regularized non-negative OOF stack.

Fold 0 was used as a screening fold.  Consequently, the decision report keeps
the two untouched continuation folds (1 and 2) visible instead of presenting a
single pooled number as if all three folds were confirmatory.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

import b1_experiments as b1
import b4p_anchor_verifier as b4


Q38_OOF_REVISION = "2026-08-23.full64-three-fold-oof-v2"


def _sha256(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _require_same_rows(actual: np.ndarray, expected: np.ndarray, source: Path) -> None:
    if np.asarray(actual).astype(str).tolist() != np.asarray(expected).astype(str).tolist():
        raise AssertionError(f"Row order mismatch in {source}")


def load_q14_oof(
    path: str | Path,
    bundle: b1.DataBundle,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the existing 14B OOF reference with strict alignment checks."""
    path = Path(path)
    saved = np.load(path, allow_pickle=True)
    _require_same_rows(saved["row_ids"], bundle.row_ids, path)
    folds = saved["folds"].astype(int)
    targets = saved["targets"].astype(np.int8)
    logits_key = "verifier_logits" if "verifier_logits" in saved.files else "logits"
    logits = saved[logits_key].astype(np.float32)
    expected_shape = bundle.factor_binary.shape
    if logits.shape != expected_shape or targets.shape != expected_shape:
        raise ValueError(f"14B OOF shape mismatch: logits={logits.shape}, targets={targets.shape}")
    if not np.array_equal(targets, bundle.factor_binary.astype(np.int8)):
        raise AssertionError("14B OOF targets differ from the loaded training data")
    if not np.isfinite(logits).all():
        raise ValueError("14B OOF contains non-finite logits")
    return logits, folds, targets


def q38_fold_path(fold_root: str | Path, fold: int) -> Path:
    return Path(fold_root) / f"fold_{fold}" / "qwen38_factor_fold_logits.npz"


def load_q38_full_oof(
    fold_root: str | Path,
    bundle: b1.DataBundle,
    expected_folds: np.ndarray,
    n_splits: int = 3,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Assemble only the held-out rows from each Qwen3.8 fold artifact.

    Taking only ``folds == fold`` is deliberate.  The NPZ matrix is full sized
    and all non-held-out rows should be ignored even if a stale cache happened
    to put values there.
    """
    expected_folds = np.asarray(expected_folds, dtype=int)
    output = np.full(bundle.factor_binary.shape, np.nan, dtype=np.float32)
    records: list[dict[str, Any]] = []
    for fold in range(n_splits):
        path = q38_fold_path(fold_root, fold)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing Qwen3.8 Fold {fold}: {path}. Run/resume that fold before OOF assembly."
            )
        saved = np.load(path, allow_pickle=True)
        _require_same_rows(saved["row_ids"], bundle.row_ids, path)
        saved_folds = saved["folds"].astype(int)
        if not np.array_equal(saved_folds, expected_folds):
            raise AssertionError(f"Fold assignment mismatch in {path}")
        if "targets" in saved.files and not np.array_equal(
            saved["targets"].astype(np.int8), bundle.factor_binary.astype(np.int8)
        ):
            raise AssertionError(f"Target mismatch in {path}")
        matrix = saved["logits"].astype(np.float32)
        if matrix.shape != bundle.factor_binary.shape:
            raise ValueError(f"Unexpected Qwen3.8 matrix shape in {path}: {matrix.shape}")
        valid = expected_folds == fold
        if not np.isfinite(matrix[valid]).all():
            missing = int((~np.isfinite(matrix[valid])).sum())
            raise ValueError(f"Fold {fold} has {missing} missing held-out logits")
        output[valid] = matrix[valid]
        metrics_path = path.with_name("qwen38_factor_fold_metrics.json")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        records.append(
            {
                "fold": fold,
                "rows": int(valid.sum()),
                "artifact": str(path),
                "sha256": _sha256(path),
                **{f"reported::{key}": value for key, value in metrics.items() if isinstance(value, (int, float))},
            }
        )
    if not np.isfinite(output).all():
        raise AssertionError("Assembled Qwen3.8 OOF is incomplete")
    return output, pd.DataFrame(records)


def _per_label_ap(targets: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    result = np.full(targets.shape[1], np.nan, dtype=np.float64)
    for label in range(targets.shape[1]):
        if np.unique(targets[:, label]).size == 2:
            result[label] = average_precision_score(targets[:, label], probabilities[:, label])
    return result


def _fold_zero_margin_metrics(
    name: str,
    logits: np.ndarray,
    targets: np.ndarray,
    folds: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in sorted(np.unique(folds)):
        valid = folds == fold
        probability = b4.sigmoid(logits[valid])
        per_ap = _per_label_ap(targets[valid], probability)
        per_f1 = np.asarray(
            [
                f1_score(targets[valid, label], logits[valid, label] >= 0, zero_division=0)
                for label in range(targets.shape[1])
            ]
        )
        support = targets[valid].sum(axis=0)
        tail = support < 60
        rows.append(
            {
                "model": name,
                "fold": int(fold),
                "rows": int(valid.sum()),
                "macro_ap": float(np.nanmean(per_ap)),
                "tail_macro_ap": float(np.nanmean(per_ap[tail])),
                "zero_margin_macro_f1": float(np.mean(per_f1)),
                "zero_margin_tail_macro_f1": float(np.mean(per_f1[tail])),
            }
        )
    return pd.DataFrame(rows)


def _probability_blend_logits(
    q14_logits: np.ndarray,
    q38_logits: np.ndarray,
    q38_weight: float,
    epsilon: float = 1e-5,
) -> np.ndarray:
    probability = (
        (1.0 - q38_weight) * b4.sigmoid(q14_logits)
        + q38_weight * b4.sigmoid(q38_logits)
    )
    probability = np.clip(probability, epsilon, 1.0 - epsilon)
    return np.log(probability / (1.0 - probability)).astype(np.float32)


def _evaluate_component(
    name: str,
    logits: np.ndarray,
    bundle: b1.DataBundle,
    folds: np.ndarray,
    config: b4.B4PConfig,
    artifact_dir: Path,
) -> dict[str, Any]:
    metrics, table, thresholds = b4.evaluate_oof_logits(
        logits, bundle.factor_binary, folds, config
    )
    table.to_csv(artifact_dir / f"per_label_{name}.csv", index=False)
    np.save(artifact_dir / f"thresholds_{name}.npy", thresholds)
    return {"experiment": name, **metrics}


def run_full_oof_decision(
    bundle: b1.DataBundle,
    folds: np.ndarray,
    q14_logits: np.ndarray,
    q38_logits: np.ndarray,
    config: b4.B4PConfig,
    artifact_dir: str | Path,
) -> dict[str, Any]:
    """Evaluate both bases, regularized stacking, and declared simple blends."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    targets = bundle.factor_binary.astype(np.int8)
    components: Mapping[str, np.ndarray] = {
        "QWEN3_14B_SHARED_SFT": np.asarray(q14_logits, dtype=np.float32),
        "QWEN38_27B_FULL64": np.asarray(q38_logits, dtype=np.float32),
    }

    # B4's calibrator is non-negative per label and cross-fitted by outer fold.
    b4_decision = b4.evaluate_b4p_oof(
        components, bundle, folds, config, artifact_dir / "NONNEGATIVE_STACK"
    )
    core_summary = pd.read_csv(
        artifact_dir / "NONNEGATIVE_STACK" / "B4P_OOF_SUMMARY.csv"
    )

    # Weight .75 was selected on Fold 0 before Folds 1/2 were observed.  It is
    # therefore a legitimate fixed continuation candidate on the latter two
    # folds, although still not an untouched external-test result.
    fixed_blend_name = "FOLD0_SELECTED_PROB_BLEND_Q38_75"
    fixed_blend_logits = _probability_blend_logits(q14_logits, q38_logits, 0.75)
    fixed_blend_row = _evaluate_component(
        fixed_blend_name, fixed_blend_logits, bundle, folds, config, artifact_dir
    )

    # The other weights remain descriptive/exploratory.
    exploratory_rows: list[dict[str, Any]] = []
    for weight in (0.25, 0.50):
        name = f"EXPLORATORY_PROB_BLEND_Q38_{int(weight * 100):02d}"
        logits = _probability_blend_logits(q14_logits, q38_logits, weight)
        exploratory_rows.append(
            _evaluate_component(name, logits, bundle, folds, config, artifact_dir)
        )
    exploratory = pd.DataFrame(exploratory_rows)
    full_summary = pd.concat(
        [core_summary, pd.DataFrame([fixed_blend_row]), exploratory], ignore_index=True
    )
    full_summary = full_summary.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    full_summary.to_csv(artifact_dir / "Q38_FULL_OOF_SUMMARY.csv", index=False)

    fold_metrics = pd.concat(
        [
            _fold_zero_margin_metrics("QWEN3_14B_SHARED_SFT", q14_logits, targets, folds),
            _fold_zero_margin_metrics("QWEN38_27B_FULL64", q38_logits, targets, folds),
            _fold_zero_margin_metrics(
                fixed_blend_name, fixed_blend_logits, targets, folds
            ),
        ],
        ignore_index=True,
    )
    fold_metrics.to_csv(artifact_dir / "Q38_FOLD_CONFIRMATION.csv", index=False)

    q14_prob, q38_prob = b4.sigmoid(q14_logits), b4.sigmoid(q38_logits)
    q14_ap, q38_ap = _per_label_ap(targets, q14_prob), _per_label_ap(targets, q38_prob)
    q14_metrics, q14_table, _ = b4.evaluate_oof_logits(q14_logits, targets, folds, config)
    q38_metrics, q38_table, _ = b4.evaluate_oof_logits(q38_logits, targets, folds, config)
    per_label = q14_table[["label", "stratum", "support", "f1"]].rename(
        columns={"f1": "q14_strict_f1"}
    )
    per_label["q14_ap"] = q14_ap
    per_label["q38_strict_f1"] = q38_table["f1"].to_numpy()
    per_label["q38_ap"] = q38_ap
    per_label["delta_strict_f1"] = per_label.q38_strict_f1 - per_label.q14_strict_f1
    per_label["delta_ap"] = per_label.q38_ap - per_label.q14_ap
    per_label = per_label.sort_values("delta_ap", ascending=False).reset_index(drop=True)
    per_label.to_csv(artifact_dir / "Q38_PER_LABEL_OOF_COMPARISON.csv", index=False)

    q14_row = core_summary.loc[core_summary.experiment == "QWEN3_14B_SHARED_SFT"].iloc[0]
    q38_row = core_summary.loc[core_summary.experiment == "QWEN38_27B_FULL64"].iloc[0]
    stack_row = core_summary.loc[core_summary.experiment == "B4P_NONNEGATIVE_STACK"].iloc[0]
    fold_pivot = fold_metrics.pivot(index="fold", columns="model", values="macro_ap")
    confirmation_folds = [int(fold) for fold in sorted(np.unique(folds)) if int(fold) != 0]
    confirmation_wins = sum(
        fold_pivot.loc[fold, "QWEN38_27B_FULL64"]
        > fold_pivot.loc[fold, "QWEN3_14B_SHARED_SFT"]
        for fold in confirmation_folds
    )

    q38_delta_f1 = float(q38_row.macro_f1 - q14_row.macro_f1)
    q38_delta_ap = float(q38_row.macro_ap - q14_row.macro_ap)
    q38_delta_tail = float(q38_row.tail_macro_f1 - q14_row.tail_macro_f1)
    q38_accepted = bool(
        q38_delta_f1 >= 0.005
        and q38_delta_ap >= 0.0
        and q38_delta_tail >= -0.010
        and confirmation_wins >= 1
    )

    fixed_row = full_summary.loc[full_summary.experiment == fixed_blend_name].iloc[0]
    fixed_delta_f1 = float(fixed_row.macro_f1 - q38_row.macro_f1)
    fixed_delta_ap = float(fixed_row.macro_ap - q38_row.macro_ap)
    fixed_delta_tail = float(fixed_row.tail_macro_f1 - q38_row.tail_macro_f1)
    fixed_ap_confirmation_wins = sum(
        fold_pivot.loc[fold, fixed_blend_name]
        > fold_pivot.loc[fold, "QWEN38_27B_FULL64"]
        for fold in confirmation_folds
    )
    fixed_f1_pivot = fold_metrics.pivot(
        index="fold", columns="model", values="zero_margin_macro_f1"
    )
    fixed_f1_confirmation_wins = sum(
        fixed_f1_pivot.loc[fold, fixed_blend_name]
        > fixed_f1_pivot.loc[fold, "QWEN38_27B_FULL64"]
        for fold in confirmation_folds
    )
    fixed_blend_accepted = bool(
        q38_accepted
        and fixed_delta_f1 >= 0.005
        and fixed_delta_ap >= 0.002
        and fixed_delta_tail >= -0.005
        and fixed_ap_confirmation_wins >= 1
        and fixed_f1_confirmation_wins >= 1
    )

    best_base = q38_row if q38_row.macro_f1 >= q14_row.macro_f1 else q14_row
    stack_delta_f1 = float(stack_row.macro_f1 - best_base.macro_f1)
    stack_delta_ap = float(stack_row.macro_ap - best_base.macro_ap)
    stack_accepted = bool(stack_delta_f1 >= 0.003 and stack_delta_ap >= -0.005)
    if fixed_blend_accepted:
        recommendation = "Q14_25_Q38_75_FIXED_PROBABILITY_BLEND"
    elif stack_accepted:
        recommendation = "Q14_Q38_NONNEGATIVE_STACK"
    elif q38_accepted:
        recommendation = "QWEN38_27B_FULL64"
    else:
        recommendation = "QWEN3_14B_SHARED_SFT"

    decision = {
        "version": "B4-Q38F-FULL64-THREE-FOLD",
        "runtime_revision": Q38_OOF_REVISION,
        "status": "complete",
        "screening_fold": 0,
        "confirmation_folds": confirmation_folds,
        "q38_standalone_accepted": q38_accepted,
        "fold0_selected_blend_accepted": fixed_blend_accepted,
        "stack_accepted": stack_accepted,
        "recommended_factor_system": recommendation,
        "q38_delta_macro_f1_vs_q14": q38_delta_f1,
        "q38_delta_macro_ap_vs_q14": q38_delta_ap,
        "q38_delta_tail_macro_f1_vs_q14": q38_delta_tail,
        "q38_confirmation_fold_ap_wins": int(confirmation_wins),
        "fixed_blend_delta_macro_f1_vs_q38": fixed_delta_f1,
        "fixed_blend_delta_macro_ap_vs_q38": fixed_delta_ap,
        "fixed_blend_delta_tail_macro_f1_vs_q38": fixed_delta_tail,
        "fixed_blend_confirmation_fold_ap_wins_vs_q38": int(fixed_ap_confirmation_wins),
        "fixed_blend_confirmation_fold_zero_f1_wins_vs_q38": int(fixed_f1_confirmation_wins),
        "stack_delta_macro_f1_vs_best_base": stack_delta_f1,
        "stack_delta_macro_ap_vs_best_base": stack_delta_ap,
        "acceptance_rule_q38": {
            "delta_macro_f1_min": 0.005,
            "delta_macro_ap_min": 0.0,
            "delta_tail_macro_f1_min": -0.010,
            "ap_wins_on_folds_1_2_min": 1,
        },
        "acceptance_rule_stack": {
            "delta_macro_f1_vs_best_base_min": 0.003,
            "delta_macro_ap_vs_best_base_min": -0.005,
        },
        "acceptance_rule_fold0_selected_blend": {
            "q38_must_first_be_accepted": True,
            "delta_macro_f1_vs_q38_min": 0.005,
            "delta_macro_ap_vs_q38_min": 0.002,
            "delta_tail_macro_f1_vs_q38_min": -0.005,
            "ap_wins_on_folds_1_2_vs_q38_min": 1,
            "zero_margin_f1_wins_on_folds_1_2_vs_q38_min": 1,
        },
        "metrics": {
            "q14": q14_metrics,
            "q38": q38_metrics,
            "fold0_selected_probability_blend_q38_75": fixed_blend_row,
            "nonnegative_stack": b4_decision["stack"],
        },
        "config": asdict(config),
        "warning": (
            "Fold 0 selected this challenger. Folds 1/2 are the continuation confirmation. "
            "The stack is cross-fitted OOF but still requires leaderboard confirmation; "
            "the 0.75 Q38 probability blend was fixed after Fold 0 and is tested on Folds 1/2; "
            "the 0.25/0.50 blends remain exploratory. Leaderboard confirmation is still required."
        ),
    }
    b4.json_dump(decision, artifact_dir / "B4_Q38_FULL_OOF_DECISION.json")
    return decision
