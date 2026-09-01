"""B8-R: document-only C-SSRS boundary verifier for suicide risk.

The existing B4 Task-1 adapter was jointly trained for risk and evidence and
showed gold evidence to most risk-training examples.  B8-R removes that
exposure mismatch: every risk decision is trained and scored from the complete
post plus one mutually-exclusive operational risk card, never from gold or
predicted evidence.

This module reuses the proven Qwen3.8 Full64/QLoRA harness in
``b4p_anchor_verifier``.  Standard attention uses PyTorch SDPA (which selects
the fastest compatible CUDA backend) while Qwen3.8's hybrid
flash-linear-attention and causal-convolution kernels remain enabled.  This
avoids the external FA2 Hub kernel's Torch-2.11 backward ABI failure.  Fold 0
is a screening gate; no calibration is fitted on Fold 0.  Only raw adapter
margins are compared with the frozen B4 baseline.
"""

from __future__ import annotations

import gc
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from torch.utils.data import Dataset

import b1_experiments as b1
import b4p_anchor_verifier as b4
import qwen38_dual_task_experiments as q38


B8R_RUNTIME_REVISION = "2026-08-28.document-only-boundary-sdpa-v2"


@dataclass(frozen=True)
class RiskOnlyConfig:
    model_name: str = "Qwen/Qwen3.8-27B"
    fold: int = 0
    n_splits: int = 3
    max_length: int = 1536
    context_chars: int = 5000
    # Extra copies of the positive exact-level pair.  Together with the four
    # base cards this approximately balances A/B targets while giving the two
    # enacted-risk classes more boundary exposure.
    positive_replays_by_class: tuple[int, int, int, int] = (1, 1, 2, 3)
    sft_epochs: float = 1.0
    sft_max_steps: int = -1
    learning_rate: float = 7.5e-5
    gradient_accumulation: int = 32
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_last_n_layers: int | None = None
    lora_target_leaves: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
    )
    gradient_checkpointing: bool = True
    score_batch_size: int = 2
    score_chunk_size: int = 256
    seed: int = 42
    # Do not use the kernels-community/flash-attn2 fallback on Torch 2.11 for
    # training: its backward path can fail in the stable-ABI aten::sum call.
    # CUDA SDPA remains flash-backed when the input shape permits it.
    attention_implementation: str = "sdpa"
    qwen35_fa2_position_guard: bool = False
    require_qwen35_fast_kernels: bool = True
    gate_weighted_f1_delta: float = 0.015
    gate_macro_f1_delta: float = 0.0
    gate_behavior_recall_delta: float = -0.03
    gate_attempt_recall_delta: float = -0.05
    bootstrap_samples: int = 1000

    def b4_config(self) -> b4.B4PConfig:
        return b4.B4PConfig(
            seed=self.seed + 1009 * self.fold,
            n_splits=self.n_splits,
            verifier_model=self.model_name,
            max_length=self.max_length,
            include_retrieval=False,
            verifier_use_chat_template=True,
            prompt_truncation_side="left",
            attention_implementation=self.attention_implementation,
            qwen35_fa2_position_guard=self.qwen35_fa2_position_guard,
            require_qwen35_fast_kernels=self.require_qwen35_fast_kernels,
            sft_epochs=self.sft_epochs,
            sft_max_steps=self.sft_max_steps,
            sft_learning_rate=self.learning_rate,
            sft_batch_size=1,
            sft_gradient_accumulation=self.gradient_accumulation,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            lora_last_n_layers=self.lora_last_n_layers,
            lora_target_leaves=self.lora_target_leaves,
            gradient_checkpointing=self.gradient_checkpointing,
            verifier_score_batch_size=self.score_batch_size,
            score_chunk_size=self.score_chunk_size,
        )


RISK_ONLY_SYSTEM = (
    "You are a deterministic C-SSRS-style research annotation verifier. "
    "Choose the single best exact risk level supported by the author's post. "
    "The four levels are mutually exclusive: if a higher enacted level is "
    "supported, lower levels must be rejected. Use only explicit first-person "
    "text; do not diagnose, invent facts, or treat another person's behavior as "
    "the author's. Output exactly A or B."
)


class StoredPromptDataset(Dataset):
    """Minimal manifest-backed dataset accepted by the B4 QLoRA trainer."""

    def __init__(self, manifest: pd.DataFrame) -> None:
        required = {"pair_id", "prompt", "target"}
        missing = required - set(manifest.columns)
        if missing:
            raise KeyError(f"risk-only manifest missing {sorted(missing)}")
        self.manifest = manifest.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[int(index)]
        return {
            "pair_id": str(row.pair_id),
            "prompt": str(row.prompt),
            "target": int(row.target),
        }


def json_dump(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
        ),
        encoding="utf-8",
    )


def make_document_risk_prompt(text: str, risk_id: int, config: RiskOnlyConfig) -> str:
    card = q38.RISK_CARDS[int(risk_id)]
    compact = q38._compact_post(str(text), config.context_chars)
    return (
        f"[SYSTEM]\n{RISK_ONLY_SYSTEM}\n[/SYSTEM]\n"
        f"[EXACT RISK CARD]\n{card.render()}\n[/EXACT RISK CARD]\n\n"
        f"[POST]\n{compact}\n[/POST]\n\n"
        "Judge the complete post, not isolated keywords. The question asks for "
        "the single best exact level, not whether the post contains any feature "
        "that can also occur at this level.\n"
        f"Question: Is {card.name} the single best exact risk level for this post?\n"
        "A = YES\nB = NO\nAnswer:"
    )


def build_training_manifest(
    bundle: b1.DataBundle,
    folds: np.ndarray,
    config: RiskOnlyConfig,
) -> pd.DataFrame:
    folds = np.asarray(folds, dtype=int)
    train_rows = np.flatnonzero(folds != int(config.fold))
    records: list[dict[str, Any]] = []
    for row in train_rows:
        gold = int(bundle.risk_ids[row])
        prompts = [make_document_risk_prompt(bundle.texts[row], risk_id, config) for risk_id in range(4)]
        for risk_id, prompt in enumerate(prompts):
            records.append(
                {
                    "pair_id": f"risk-base::{bundle.row_ids[row]}::{risk_id}",
                    "row_idx": int(row),
                    "risk_id": int(risk_id),
                    "gold_risk_id": gold,
                    "kind": "positive_base" if risk_id == gold else (
                        "negative_adjacent" if abs(risk_id - gold) == 1 else "negative_far"
                    ),
                    "target": int(risk_id == gold),
                    "prompt": prompt,
                }
            )
        for replay in range(int(config.positive_replays_by_class[gold])):
            records.append(
                {
                    "pair_id": f"risk-positive-replay::{bundle.row_ids[row]}::{gold}::{replay}",
                    "row_idx": int(row),
                    "risk_id": gold,
                    "gold_risk_id": gold,
                    "kind": "positive_replay",
                    "target": 1,
                    "prompt": prompts[gold],
                }
            )
    frame = pd.DataFrame(records)
    rng = np.random.default_rng(config.seed + 1009 * int(config.fold))
    frame = frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)
    if frame.pair_id.duplicated().any():
        raise AssertionError("duplicate B8-R training pair ids")
    return frame


def build_scoring_manifest(
    bundle: b1.DataBundle,
    rows: Sequence[int],
    config: RiskOnlyConfig,
    stage: str = "valid",
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in map(int, rows):
        for risk_id in range(4):
            records.append(
                {
                    "pair_id": f"risk-only-{stage}::{bundle.row_ids[row]}::{risk_id}",
                    "query_row_idx": row,
                    "risk_id": risk_id,
                    "prompt": make_document_risk_prompt(bundle.texts[row], risk_id, config),
                }
            )
    return pd.DataFrame(records)


def risk_matrix(
    manifest: pd.DataFrame,
    margins: np.ndarray,
    n_rows: int,
) -> np.ndarray:
    matrix = np.full((n_rows, 4), np.nan, dtype=np.float32)
    for row, risk_id, margin in zip(
        manifest.query_row_idx.astype(int),
        manifest.risk_id.astype(int),
        np.asarray(margins, dtype=np.float32),
    ):
        if np.isfinite(matrix[row, risk_id]):
            raise AssertionError(f"duplicate risk score row={row}, risk={risk_id}")
        matrix[row, risk_id] = float(margin)
    return matrix


def risk_metrics(gold: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    gold = np.asarray(gold, dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, prediction, labels=np.arange(4), zero_division=0
    )
    return {
        "weighted_f1": float(f1_score(gold, prediction, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(gold, prediction, average="macro", zero_division=0)),
        "accuracy": float(np.mean(gold == prediction)),
        "mean_absolute_severity_error": float(np.mean(np.abs(gold - prediction))),
        "confusion_matrix": confusion_matrix(gold, prediction, labels=np.arange(4)).tolist(),
        "per_class": {
            b1.RISK_LABELS[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(4)
        },
    }


def _row_standardize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    return (centered / np.maximum(scale, 1e-5)).astype(np.float32)


def _paired_bootstrap_delta(
    gold: np.ndarray,
    baseline: np.ndarray,
    challenger: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = np.zeros(int(samples), dtype=np.float32)
    for index in range(int(samples)):
        rows = rng.integers(0, len(gold), size=len(gold))
        base = f1_score(gold[rows], baseline[rows], average="weighted", zero_division=0)
        new = f1_score(gold[rows], challenger[rows], average="weighted", zero_division=0)
        values[index] = new - base
    return {
        "mean": float(values.mean()),
        "p025": float(np.quantile(values, 0.025)),
        "p50": float(np.quantile(values, 0.50)),
        "p975": float(np.quantile(values, 0.975)),
        "probability_positive": float(np.mean(values > 0)),
    }


def locate_frozen_task1_fold(root: str | Path, fold: int) -> Path:
    root = Path(root)
    candidates = [
        root / f"results/B4_TASK1_Q38_FULL64_FOLD0/Q38_FULL64/fold_{fold}/EVALUATION/q38_task1_fold_outputs.npz",
        root / f"results/B4_TASK1_Q38_FULL64_OUTER_CONFIRM/Q38_FULL64/fold_{fold}/EVALUATION/q38_task1_fold_outputs.npz",
    ]
    for path in candidates:
        if path.exists():
            return path
    discovered = sorted(root.glob(f"results/**/fold_{fold}/EVALUATION/q38_task1_fold_outputs.npz"))
    if len(discovered) == 1:
        return discovered[0]
    raise FileNotFoundError(
        f"Could not uniquely locate frozen Task1 fold {fold}; candidates={discovered}"
    )


def load_frozen_risk_margins(
    path: str | Path,
    bundle: b1.DataBundle,
    folds: np.ndarray,
    fold: int,
) -> dict[str, np.ndarray]:
    path = Path(path)
    saved = np.load(path, allow_pickle=True)
    if saved["row_ids"].astype(str).tolist() != bundle.row_ids.astype(str).tolist():
        raise AssertionError(f"frozen Task1 row mismatch: {path}")
    if "folds" in saved.files and not np.array_equal(saved["folds"].astype(int), np.asarray(folds, dtype=int)):
        raise AssertionError(f"frozen Task1 fold vector mismatch: {path}")
    valid = np.asarray(folds, dtype=int) == int(fold)
    result: dict[str, np.ndarray] = {}
    for name, key in (
        ("B4_DOCUMENT", "document_risk_margins"),
        ("B4_CONDITIONED", "conditioned_risk_margins"),
        ("B4_FIXED_BLEND", "blended_risk_margins"),
    ):
        matrix = np.asarray(saved[key], dtype=np.float32)
        if matrix.shape != (len(bundle.texts), 4) or not np.isfinite(matrix[valid]).all():
            raise ValueError(f"invalid {key} in {path}")
        result[name] = matrix
    return result


def train_risk_only_fold(
    bundle: b1.DataBundle,
    folds: np.ndarray,
    config: RiskOnlyConfig,
    output_dir: str | Path,
    overwrite: bool = False,
) -> tuple[Path, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "risk_only_training_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        expected_rows = set(np.flatnonzero(np.asarray(folds) != int(config.fold)).tolist())
        if set(manifest.row_idx.astype(int)) != expected_rows:
            raise AssertionError("resumed B8-R manifest uses different training rows")
    else:
        manifest = build_training_manifest(bundle, folds, config)
        manifest.to_csv(manifest_path, index=False)
    audit = manifest.drop(columns=["prompt"])
    audit.to_csv(output_dir / "risk_only_training_audit.csv", index=False)
    dataset = StoredPromptDataset(manifest)
    adapter = b4.train_verifier_adapter(
        dataset, config.b4_config(), output_dir / "TRAINING", overwrite=overwrite
    )
    return adapter, audit


def evaluate_risk_only_fold(
    bundle: b1.DataBundle,
    folds: np.ndarray,
    config: RiskOnlyConfig,
    adapter_path: str | Path,
    frozen_task1_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_rows = np.flatnonzero(np.asarray(folds, dtype=int) == int(config.fold))
    scoring = build_scoring_manifest(bundle, valid_rows, config)
    scoring.drop(columns=["prompt"]).to_csv(output_dir / "risk_only_scoring_audit.csv", index=False)
    b4_config = config.b4_config()
    model, tokenizer = b4.load_quantized_causal_model(
        config.model_name,
        adapter_path=adapter_path,
        training=False,
        attention_implementation=config.attention_implementation,
        qwen35_fa2_position_guard=config.qwen35_fa2_position_guard,
        require_qwen35_fast_kernels=config.require_qwen35_fast_kernels,
    )
    margins = b4.score_prompts_cached(
        model,
        tokenizer,
        scoring,
        output_dir / "SCORE_CACHE",
        b4_config,
        config.score_batch_size,
        use_chat_template=True,
    )
    matrix = risk_matrix(scoring, margins, len(bundle.texts))
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    frozen = load_frozen_risk_margins(
        frozen_task1_path, bundle, folds, config.fold
    )
    systems: dict[str, np.ndarray] = dict(frozen)
    systems["B8R_RISK_ONLY"] = matrix
    # Fixed, label-free ensemble for screening.  Per-row standardization makes
    # A/B margin scales comparable without touching Fold-0 labels.
    systems["B8R_50_B4DOC_50"] = np.full_like(matrix, np.nan)
    systems["B8R_50_B4DOC_50"][valid_rows] = (
        0.5 * _row_standardize(matrix[valid_rows])
        + 0.5 * _row_standardize(frozen["B4_DOCUMENT"][valid_rows])
    )

    gold = bundle.risk_ids[valid_rows].astype(int)
    predictions: dict[str, np.ndarray] = {
        name: np.argmax(values[valid_rows], axis=1).astype(int)
        for name, values in systems.items()
    }
    metrics = {name: risk_metrics(gold, prediction) for name, prediction in predictions.items()}
    baseline = metrics["B4_FIXED_BLEND"]
    challenger = metrics["B8R_RISK_ONLY"]
    delta_weighted = challenger["weighted_f1"] - baseline["weighted_f1"]
    delta_macro = challenger["macro_f1"] - baseline["macro_f1"]
    behavior_delta = (
        challenger["per_class"]["Behavior"]["recall"]
        - baseline["per_class"]["Behavior"]["recall"]
    )
    attempt_delta = (
        challenger["per_class"]["Attempt"]["recall"]
        - baseline["per_class"]["Attempt"]["recall"]
    )
    bootstrap = _paired_bootstrap_delta(
        gold,
        predictions["B4_FIXED_BLEND"],
        predictions["B8R_RISK_ONLY"],
        config.bootstrap_samples,
        config.seed + config.fold,
    )
    passed = bool(
        delta_weighted >= config.gate_weighted_f1_delta
        and delta_macro >= config.gate_macro_f1_delta
        and behavior_delta >= config.gate_behavior_recall_delta
        and attempt_delta >= config.gate_attempt_recall_delta
    )
    decision = {
        "runtime_revision": B8R_RUNTIME_REVISION,
        "fold": int(config.fold),
        "rows": int(len(valid_rows)),
        "primary_challenger": "B8R_RISK_ONLY",
        "paired_baseline": "B4_FIXED_BLEND",
        "metrics": metrics,
        "delta_weighted_f1": float(delta_weighted),
        "delta_macro_f1": float(delta_macro),
        "delta_behavior_recall": float(behavior_delta),
        "delta_attempt_recall": float(attempt_delta),
        "paired_bootstrap_weighted_f1_delta": bootstrap,
        "passed": passed,
        "decision": "RUN_REMAINING_FOLDS" if passed else "STOP_B8R",
        "gate": {
            "weighted_f1_delta_min": config.gate_weighted_f1_delta,
            "macro_f1_delta_min": config.gate_macro_f1_delta,
            "behavior_recall_delta_min": config.gate_behavior_recall_delta,
            "attempt_recall_delta_min": config.gate_attempt_recall_delta,
        },
        "config": asdict(config),
        "warning": (
            "Fold 0 is a screening gate. The fixed 50/50 ensemble is diagnostic only. "
            "No Fold-0-fitted threshold, blend weight, or classifier is reported."
        ),
    }
    rows: list[dict[str, Any]] = []
    for name, value in metrics.items():
        rows.append(
            {
                "system": name,
                "weighted_f1": value["weighted_f1"],
                "macro_f1": value["macro_f1"],
                "accuracy": value["accuracy"],
                "mean_absolute_severity_error": value["mean_absolute_severity_error"],
                "indicator_recall": value["per_class"]["Indicator"]["recall"],
                "ideation_recall": value["per_class"]["Ideation"]["recall"],
                "behavior_recall": value["per_class"]["Behavior"]["recall"],
                "attempt_recall": value["per_class"]["Attempt"]["recall"],
            }
        )
    pd.DataFrame(rows).sort_values("weighted_f1", ascending=False).to_csv(
        output_dir / "B8R_RISK_SUMMARY.csv", index=False
    )
    prediction_frame = pd.DataFrame(
        {
            "row_id": bundle.row_ids[valid_rows].astype(str),
            "gold_risk": [b1.RISK_LABELS[value] for value in gold],
            **{
                f"prediction__{name}": [b1.RISK_LABELS[value] for value in prediction]
                for name, prediction in predictions.items()
            },
        }
    )
    prediction_frame.to_csv(output_dir / "B8R_VALIDATION_PREDICTIONS.csv", index=False)
    np.savez_compressed(
        output_dir / "B8R_FOLD_OUTPUTS.npz",
        row_ids=bundle.row_ids.astype(str),
        folds=np.asarray(folds, dtype=int),
        valid_rows=valid_rows,
        risk_only_margins=matrix,
        **{f"margins__{name}": value for name, value in frozen.items()},
    )
    json_dump(decision, output_dir / "B8R_FOLD_DECISION.json")
    return decision


def run_fold(
    root: str | Path,
    bundle: b1.DataBundle,
    folds: np.ndarray,
    config: RiskOnlyConfig,
    output_dir: str | Path,
    overwrite_adapter: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter, _ = train_risk_only_fold(
        bundle, folds, config, output_dir, overwrite=overwrite_adapter
    )
    frozen_path = locate_frozen_task1_fold(root, config.fold)
    return evaluate_risk_only_fold(
        bundle,
        folds,
        config,
        adapter,
        frozen_path,
        output_dir / "EVALUATION",
    )


def smoke_config(config: RiskOnlyConfig, steps: int = 2) -> RiskOnlyConfig:
    return replace(
        config,
        sft_epochs=1.0,
        sft_max_steps=int(steps),
        gradient_accumulation=max(4, min(8, config.gradient_accumulation)),
    )


def _crossfit_logistic_predictions(
    features: np.ndarray,
    gold: np.ndarray,
    folds: np.ndarray,
    c_value: float = 0.1,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    gold = np.asarray(gold, dtype=int)
    folds = np.asarray(folds, dtype=int)
    prediction = np.full(len(gold), -1, dtype=int)
    for outer_fold in sorted(np.unique(folds).tolist()):
        train, valid = folds != outer_fold, folds == outer_fold
        model = LogisticRegression(
            C=float(c_value),
            max_iter=2000,
            solver="lbfgs",
            class_weight=None,
            random_state=42 + int(outer_fold),
        )
        model.fit(features[train], gold[train])
        prediction[valid] = model.predict(features[valid]).astype(int)
    if (prediction < 0).any():
        raise AssertionError("incomplete B8-R crossfit logistic predictions")
    return prediction


def aggregate_three_fold_oof(
    root: str | Path,
    bundle: b1.DataBundle,
    folds: np.ndarray,
    fold_output_dirs: Sequence[str | Path],
    config: RiskOnlyConfig,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Strictly aggregate three B8-R outer folds and crossfit meta-decoders."""

    if len(fold_output_dirs) != 3:
        raise ValueError("B8-R OOF aggregation requires exactly three fold directories")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = np.asarray(folds, dtype=int)
    n_rows = len(bundle.texts)
    new = np.full((n_rows, 4), np.nan, dtype=np.float32)
    frozen_full = {
        name: np.full((n_rows, 4), np.nan, dtype=np.float32)
        for name in ("B4_DOCUMENT", "B4_CONDITIONED", "B4_FIXED_BLEND")
    }
    for fold, directory in enumerate(map(Path, fold_output_dirs)):
        path = directory / "EVALUATION/B8R_FOLD_OUTPUTS.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        saved = np.load(path, allow_pickle=True)
        if saved["row_ids"].astype(str).tolist() != bundle.row_ids.astype(str).tolist():
            raise AssertionError(f"B8-R OOF row mismatch: {path}")
        valid = folds == fold
        matrix = np.asarray(saved["risk_only_margins"], dtype=np.float32)
        if matrix.shape != (n_rows, 4) or not np.isfinite(matrix[valid]).all():
            raise ValueError(f"invalid B8-R margins: {path}")
        new[valid] = matrix[valid]
        frozen = load_frozen_risk_margins(
            locate_frozen_task1_fold(root, fold), bundle, folds, fold
        )
        for name, values in frozen.items():
            frozen_full[name][valid] = values[valid]
    matrices = {"B8R_RISK_ONLY": new, **frozen_full}
    if any(not np.isfinite(value).all() for value in matrices.values()):
        raise AssertionError("incomplete B8-R/frozen three-fold OOF matrices")

    gold = bundle.risk_ids.astype(int)
    predictions: dict[str, np.ndarray] = {
        name: np.argmax(values, axis=1).astype(int) for name, values in matrices.items()
    }
    fixed_ensemble = 0.5 * _row_standardize(new) + 0.5 * _row_standardize(
        frozen_full["B4_DOCUMENT"]
    )
    predictions["B8R_50_B4DOC_50"] = np.argmax(fixed_ensemble, axis=1).astype(int)
    predictions["B8R_CROSSFIT_CALIBRATED"] = _crossfit_logistic_predictions(
        _row_standardize(new), gold, folds, c_value=0.1
    )
    stack_features = np.column_stack(
        [
            _row_standardize(new),
            _row_standardize(frozen_full["B4_DOCUMENT"]),
            _row_standardize(frozen_full["B4_CONDITIONED"]),
        ]
    )
    predictions["B8R_B4_CROSSFIT_STACK"] = _crossfit_logistic_predictions(
        stack_features, gold, folds, c_value=0.1
    )

    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for name, prediction in predictions.items():
        value = risk_metrics(gold, prediction)
        summary_rows.append(
            {
                "system": name,
                "weighted_f1": value["weighted_f1"],
                "macro_f1": value["macro_f1"],
                "accuracy": value["accuracy"],
                "mean_absolute_severity_error": value["mean_absolute_severity_error"],
            }
        )
        for fold in sorted(np.unique(folds).tolist()):
            valid = folds == fold
            fold_value = risk_metrics(gold[valid], prediction[valid])
            fold_rows.append(
                {
                    "outer_fold": int(fold),
                    "system": name,
                    "weighted_f1": fold_value["weighted_f1"],
                    "macro_f1": fold_value["macro_f1"],
                }
            )
    summary = pd.DataFrame(summary_rows)
    fold_frame = pd.DataFrame(fold_rows)
    baseline = summary.set_index("system").loc["B4_FIXED_BLEND"]
    baseline_folds = fold_frame[fold_frame.system == "B4_FIXED_BLEND"].set_index(
        "outer_fold"
    )
    candidates = (
        "B8R_RISK_ONLY",
        "B8R_50_B4DOC_50",
        "B8R_CROSSFIT_CALIBRATED",
        "B8R_B4_CROSSFIT_STACK",
    )
    gate_rows: list[dict[str, Any]] = []
    for name in candidates:
        row = summary.set_index("system").loc[name]
        candidate_folds = fold_frame[fold_frame.system == name].set_index("outer_fold")
        delta = candidate_folds.weighted_f1 - baseline_folds.weighted_f1
        gate_rows.append(
            {
                "system": name,
                "delta_weighted_f1": float(row.weighted_f1 - baseline.weighted_f1),
                "delta_macro_f1": float(row.macro_f1 - baseline.macro_f1),
                "folds_improved": int((delta > 0).sum()),
                "worst_fold_delta": float(delta.min()),
                "passed": bool(
                    row.weighted_f1 - baseline.weighted_f1 >= config.gate_weighted_f1_delta
                    and row.macro_f1 - baseline.macro_f1 >= config.gate_macro_f1_delta
                    and int((delta > 0).sum()) >= 2
                    and float(delta.min()) >= -0.01
                ),
            }
        )
    gate = pd.DataFrame(gate_rows).sort_values(
        ["passed", "delta_weighted_f1"], ascending=[False, False]
    )
    accepted = gate[gate.passed]
    selected = str(accepted.iloc[0].system) if len(accepted) else None
    decision = {
        "runtime_revision": B8R_RUNTIME_REVISION,
        "status": "complete",
        "confirmed": selected is not None,
        "selected_system": selected,
        "summary": summary.to_dict(orient="records"),
        "gate_results": gate.to_dict(orient="records"),
        "evaluation": (
            "three grouped outer folds; logistic calibration/stack predictions are "
            "crossfit and never trained on the row being evaluated"
        ),
    }
    summary.sort_values("weighted_f1", ascending=False).to_csv(
        output_dir / "B8R_OOF_SUMMARY.csv", index=False
    )
    fold_frame.to_csv(output_dir / "B8R_OOF_FOLDS.csv", index=False)
    gate.to_csv(output_dir / "B8R_OOF_GATE.csv", index=False)
    np.savez_compressed(
        output_dir / "B8R_ALL_OOF.npz",
        row_ids=bundle.row_ids.astype(str),
        folds=folds,
        targets=gold,
        risk_only_margins=new,
        **{f"prediction__{name}": value for name, value in predictions.items()},
        **{f"margins__{name}": value for name, value in frozen_full.items()},
    )
    json_dump(decision, output_dir / "B8R_OOF_DECISION.json")
    return decision
