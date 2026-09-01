"""B1-R residual factor-classification experiments for Google Colab.

This module deliberately keeps the already working B1 encoders intact.  It
adds two leak-safe layers around their OOF predictions:

1. a strongly regularised, per-label residual stacker; and
2. a shared label-wise cross-encoder trained from balanced positives and OOF
   hard negatives.

The residual form anchors every prediction to the team's real Task2 baseline.
An auxiliary component therefore has to demonstrate useful OOF residual signal
before it can materially change the baseline.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import shutil
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

import b1_experiments as b1


FACTOR_LABELS = b1.FACTOR_LABELS
N_LABELS = len(FACTOR_LABELS)


# These cards are frozen model inputs, not generated annotations.  They clarify
# the positive concept and the most common semantic boundary for each label.
FACTOR_CARDS: dict[str, str] = {
    "cognitive deficits": (
        "Impaired thinking, concentration, memory, judgement, problem solving, or cognitive functioning. "
        "Do not infer this merely from emotional distress."
    ),
    "coping strategy": (
        "An action or mental strategy used to manage distress, such as distraction, therapy, exercise, "
        "writing, avoidance, or substance-based coping."
    ),
    "dysfunctional family": (
        "Persistent family conflict, neglect, abuse, rejection, instability, or seriously unhealthy family dynamics."
    ),
    "emotion dysregulation": (
        "Difficulty controlling intense, rapidly changing, overwhelming, or explosive emotions. "
        "Ordinary sadness alone is insufficient."
    ),
    "exposure to others' suicide": (
        "The person was exposed to another individual's suicide, suicide attempt, or suicidal behaviour."
    ),
    "hopelessness": (
        "A belief that the future will not improve, there is no way out, or effort is pointless. "
        "Distinguish it from a temporary negative mood."
    ),
    "interpersonal difficulty": (
        "Conflict, breakup, rejection, bullying, isolation, or difficulty maintaining peer, romantic, or work relationships."
    ),
    "interpersonal violence": (
        "Physical, sexual, or severe psychological violence between people, including assault or domestic violence."
    ),
    "low self-esteem": (
        "Negative self-worth, self-hatred, worthlessness, or a persistently devalued view of oneself."
    ),
    "low socio-economic status": (
        "Poverty, serious financial hardship, unemployment-related deprivation, insecure housing, or lack of basic resources."
    ),
    "meaning in life": (
        "A stated sense of purpose, valued life direction, or reasons that make life meaningful."
    ),
    "mental health issues": (
        "A stated or clearly described mental-health condition or substantial psychiatric symptoms, such as depression, "
        "anxiety, psychosis, bipolar symptoms, or an eating disorder."
    ),
    "physical health/characteristic": (
        "Physical illness, pain, disability, sleep or bodily-health problem, or a physical characteristic central to distress."
    ),
    "poor school performance": (
        "Academic failure, falling grades, inability to meet school requirements, or distress explicitly caused by poor performance."
    ),
    "poor social support": (
        "Absent, unreliable, invalidating, or insufficient practical or emotional support. "
        "Being physically alone is not automatically poor support."
    ),
    "prior self-harm or suicidal thought/attempt": (
        "A previous or ongoing history of self-harm, suicidal thoughts, suicide planning, or a suicide attempt."
    ),
    "psychological capital": (
        "Protective inner resources such as hope, optimism, resilience, confidence, or belief in one's capacity to recover."
    ),
    "sense of responsibility": (
        "Responsibility or obligation toward family, dependants, work, pets, or others that influences behaviour or protects life."
    ),
    "sexual orientation related issues": (
        "Distress, discrimination, rejection, concealment, or conflict specifically related to sexual orientation or gender identity."
    ),
    "social support": (
        "Available and helpful emotional, informational, or practical support from friends, family, professionals, or community."
    ),
    "stressful life event": (
        "A concrete major stressor or disruptive event, such as bereavement, breakup, job loss, legal trouble, relocation, "
        "pandemic disruption, or acute academic pressure."
    ),
    "substance use": (
        "Use or misuse of alcohol, drugs, or other psychoactive substances relevant to the person's condition or coping."
    ),
    "suicide means (with access)": (
        "A suicide method or means is available or accessible to the person, not merely mentioned abstractly."
    ),
    "traumatic experience": (
        "A past or current event experienced as traumatic, including abuse, assault, disaster, severe loss, or other lasting trauma."
    ),
}

if set(FACTOR_CARDS) != set(FACTOR_LABELS):
    raise RuntimeError("FACTOR_CARDS must define every factor label exactly once")


def _json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def probabilities_to_logits(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p)).astype(np.float32)


def _align_rows(
    matrix: np.ndarray,
    saved_row_ids: np.ndarray | None,
    expected_row_ids: np.ndarray,
    source: Path,
) -> np.ndarray:
    if saved_row_ids is None:
        if len(matrix) != len(expected_row_ids):
            raise ValueError(
                f"{source}: {len(matrix)} rows but training bundle has {len(expected_row_ids)} rows"
            )
        warnings.warn(
            f"{source} has no row_ids; positional alignment is being used. "
            "Export row_ids with the real baseline OOF before making a final claim."
        )
        return matrix

    saved = np.asarray(saved_row_ids).astype(str)
    expected = np.asarray(expected_row_ids).astype(str)
    if len(np.unique(saved)) != len(saved):
        raise ValueError(f"{source}: row_ids are not unique")
    lookup = {value: idx for idx, value in enumerate(saved)}
    missing = [value for value in expected if value not in lookup]
    if missing:
        raise ValueError(f"{source}: missing {len(missing)} expected row_ids, including {missing[:5]}")
    return matrix[np.asarray([lookup[value] for value in expected], dtype=np.int64)]


def load_oof_logits(
    path: str | Path,
    bundle: b1.DataBundle,
    *,
    expected_folds: np.ndarray | None = None,
    strict_fold_match: bool = False,
) -> np.ndarray:
    """Load and align an Nx24 OOF artifact from NPZ or CSV.

    Accepted NPZ keys are ``logits`` or ``probabilities``.  CSV files may hold
    either 24 columns named after the factors, ``logit_<factor>``,
    ``prob_<factor>``, or 24 numeric prediction columns plus ``row_id``.
    """

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)

    row_ids: np.ndarray | None = None
    saved_folds: np.ndarray | None = None
    is_probability = False

    if source.suffix.lower() == ".npz":
        saved = np.load(source, allow_pickle=True)
        if "logits" in saved:
            matrix = np.asarray(saved["logits"], dtype=np.float32)
        elif "probabilities" in saved:
            matrix = np.asarray(saved["probabilities"], dtype=np.float32)
            is_probability = True
        else:
            raise KeyError(f"{source}: NPZ needs a logits or probabilities array")
        row_ids = np.asarray(saved["row_ids"]) if "row_ids" in saved else None
        saved_folds = np.asarray(saved["folds"]) if "folds" in saved else None
    elif source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
        if "row_id" in frame.columns:
            row_ids = frame["row_id"].astype(str).to_numpy()
        canonical = {str(column).strip().lower(): column for column in frame.columns}
        logit_columns = []
        probability_columns = []
        for label in FACTOR_LABELS:
            choices = [f"logit_{label}", f"logit::{label}", label]
            found = next((canonical[key] for key in choices if key in canonical), None)
            if found is not None:
                logit_columns.append(found)
            choices = [f"prob_{label}", f"probability_{label}", f"p::{label}"]
            found = next((canonical[key] for key in choices if key in canonical), None)
            if found is not None:
                probability_columns.append(found)
        if len(logit_columns) == N_LABELS:
            matrix = frame[logit_columns].to_numpy(dtype=np.float32)
        elif len(probability_columns) == N_LABELS:
            matrix = frame[probability_columns].to_numpy(dtype=np.float32)
            is_probability = True
        else:
            excluded = {"row_id", "fold", "idx", "index"}
            numeric = [
                column
                for column in frame.select_dtypes(include=[np.number]).columns
                if str(column).strip().lower() not in excluded
            ]
            if len(numeric) != N_LABELS:
                raise ValueError(
                    f"{source}: could not identify 24 ordered logit/probability columns; "
                    f"found {len(numeric)} numeric candidates"
                )
            matrix = frame[numeric].to_numpy(dtype=np.float32)
            is_probability = bool(np.nanmin(matrix) >= 0.0 and np.nanmax(matrix) <= 1.0)
        if "fold" in frame.columns:
            saved_folds = frame["fold"].to_numpy(dtype=np.int64)
    else:
        raise ValueError(f"Unsupported OOF format: {source.suffix}")

    if matrix.ndim != 2 or matrix.shape[1] != N_LABELS:
        raise ValueError(f"{source}: expected [N,{N_LABELS}], got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{source}: predictions contain NaN or infinity")
    matrix = _align_rows(matrix, row_ids, bundle.row_ids, source)

    if saved_folds is not None and expected_folds is not None:
        aligned_folds = _align_rows(
            np.asarray(saved_folds).reshape(-1, 1), row_ids, bundle.row_ids, source
        ).reshape(-1)
        if not np.array_equal(aligned_folds.astype(int), np.asarray(expected_folds).astype(int)):
            message = (
                f"{source}: saved folds do not match B1-R folds. The predictions can still be "
                "used if they are genuine OOF predictions, but identical folds are preferred."
            )
            if strict_fold_match:
                raise ValueError(message)
            warnings.warn(message)

    if is_probability:
        matrix = probabilities_to_logits(matrix)
    return np.clip(matrix.astype(np.float32), -15.0, 15.0)


def discover_real_baseline_oof(root: str | Path) -> Path | None:
    root = Path(root)
    candidates = [
        root / "task2_oof.npz",
        root / "task2_oof.csv",
        root / "factor_oof.npz",
        root / "results" / "task2_oof.npz",
        root / "results" / "task2_oof.csv",
        root / "results" / "TASK2_BASELINE" / "oof_complete.npz",
        root / "results" / "Task2_Baseline" / "oof_complete.npz",
        root / "results" / "task2_baseline" / "oof_complete.npz",
    ]
    return next((path for path in candidates if path.exists()), None)


@dataclass
class ResidualStackerConfig:
    name: str = "B1R_CORE_STACK"
    l2_residual: float = 0.25
    l2_intercept: float = 1.0
    positive_weight_power: float = 0.35
    max_positive_weight: float = 4.0
    residual_lower_bound: float = -0.25
    residual_upper_bound: float = 1.00
    intercept_bound: float = 3.0
    max_iter: int = 250
    threshold_kappa: float = 25.0


@dataclass
class ResidualParameters:
    coefficients: np.ndarray
    intercepts: np.ndarray


def _fit_one_residual_label(
    component_logits: np.ndarray,
    target: np.ndarray,
    config: ResidualStackerConfig,
) -> tuple[np.ndarray, float]:
    """Fit z = anchor + sum(beta_k * (component_k-anchor)) + bias."""

    x = np.asarray(component_logits, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    n_components = x.shape[1]
    if n_components == 1 or y.sum() == 0 or y.sum() == len(y):
        return np.zeros(max(0, n_components - 1), dtype=np.float32), 0.0

    anchor = x[:, 0]
    delta = x[:, 1:] - anchor[:, None]
    positives = max(1.0, float(y.sum()))
    negatives = max(1.0, float(len(y) - y.sum()))
    pos_weight = min(
        config.max_positive_weight,
        max(1.0, (negatives / positives) ** config.positive_weight_power),
    )
    sample_weight = np.where(y > 0.5, pos_weight, 1.0)
    sample_weight /= sample_weight.mean()

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        beta = theta[:-1]
        bias = theta[-1]
        logits = anchor + delta @ beta + bias
        per_sample = np.logaddexp(0.0, logits) - y * logits
        loss = float(np.mean(sample_weight * per_sample))
        loss += 0.5 * config.l2_residual * float(beta @ beta)
        loss += 0.5 * config.l2_intercept * float(bias * bias)
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        residual = sample_weight * (probabilities - y) / len(y)
        grad_beta = delta.T @ residual + config.l2_residual * beta
        grad_bias = float(residual.sum() + config.l2_intercept * bias)
        return loss, np.concatenate([grad_beta, [grad_bias]])

    initial = np.zeros(n_components, dtype=np.float64)
    bounds = [
        (config.residual_lower_bound, config.residual_upper_bound)
        for _ in range(n_components - 1)
    ] + [(-config.intercept_bound, config.intercept_bound)]
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": config.max_iter, "ftol": 1e-9},
    )
    if not result.success:
        warnings.warn(f"Residual optimisation did not fully converge: {result.message}")
    return result.x[:-1].astype(np.float32), float(result.x[-1])


def fit_residual_stacker(
    components: Sequence[np.ndarray],
    targets: np.ndarray,
    config: ResidualStackerConfig,
) -> ResidualParameters:
    if not components:
        raise ValueError("At least one OOF component is required")
    stacked = np.stack([np.asarray(value, dtype=np.float32) for value in components], axis=-1)
    if stacked.shape[:2] != targets.shape:
        raise ValueError(f"Components {stacked.shape} do not align with targets {targets.shape}")
    coefficients = np.zeros((N_LABELS, max(0, len(components) - 1)), dtype=np.float32)
    intercepts = np.zeros(N_LABELS, dtype=np.float32)
    for label in range(N_LABELS):
        beta, bias = _fit_one_residual_label(stacked[:, label, :], targets[:, label], config)
        coefficients[label] = beta
        intercepts[label] = bias
    return ResidualParameters(coefficients=coefficients, intercepts=intercepts)


def apply_residual_stacker(
    components: Sequence[np.ndarray],
    parameters: ResidualParameters,
) -> np.ndarray:
    anchor = np.asarray(components[0], dtype=np.float32)
    result = anchor + parameters.intercepts[None, :]
    for component_index, component in enumerate(components[1:]):
        delta = np.asarray(component, dtype=np.float32) - anchor
        result = result + delta * parameters.coefficients[:, component_index][None, :]
    return np.clip(result, -15.0, 15.0).astype(np.float32)


def crossfit_residual_stacker(
    components: Sequence[np.ndarray],
    targets: np.ndarray,
    folds: np.ndarray,
    config: ResidualStackerConfig,
) -> tuple[np.ndarray, ResidualParameters, dict[str, Any]]:
    n = len(targets)
    crossfit_logits = np.full((n, N_LABELS), np.nan, dtype=np.float32)
    fold_parameters: dict[str, Any] = {}
    for fold in sorted(np.unique(folds)):
        train = np.asarray(folds) != fold
        valid = ~train
        parameters = fit_residual_stacker(
            [component[train] for component in components], targets[train], config
        )
        crossfit_logits[valid] = apply_residual_stacker(
            [component[valid] for component in components], parameters
        )
        fold_parameters[str(int(fold))] = {
            "coefficients": parameters.coefficients.tolist(),
            "intercepts": parameters.intercepts.tolist(),
        }
    if np.isnan(crossfit_logits).any():
        raise RuntimeError("Cross-fitted residual logits contain missing values")
    final_parameters = fit_residual_stacker(components, targets, config)
    return crossfit_logits, final_parameters, fold_parameters


def evaluate_oof_logits(
    logits: np.ndarray,
    targets: np.ndarray,
    folds: np.ndarray,
    *,
    threshold_kappa: float = 25.0,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray, np.ndarray]:
    probabilities = sigmoid(logits)
    predictions, thresholds = b1.nested_threshold_predictions(
        probabilities, targets, folds, kappa=threshold_kappa
    )
    metrics, per_label = b1.factor_metrics(targets, predictions, probabilities)
    return metrics, per_label, predictions, thresholds


def run_residual_stack_experiment(
    bundle: b1.DataBundle,
    folds: np.ndarray,
    component_names: Sequence[str],
    component_logits: Sequence[np.ndarray],
    config: ResidualStackerConfig,
    artifact_root: str | Path,
    *,
    baseline_is_real: bool,
) -> dict[str, Any]:
    if len(component_names) != len(component_logits):
        raise ValueError("component_names and component_logits must have equal length")
    if len(set(component_names)) != len(component_names):
        raise ValueError("component_names must be unique")
    output_dir = Path(artifact_root) / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    crossfit_logits, final_parameters, fold_parameters = crossfit_residual_stacker(
        component_logits, bundle.factor_binary, folds, config
    )
    metrics, per_label, predictions, thresholds = evaluate_oof_logits(
        crossfit_logits,
        bundle.factor_binary,
        folds,
        threshold_kappa=config.threshold_kappa,
    )
    anchor_metrics, anchor_per_label, _, _ = evaluate_oof_logits(
        component_logits[0],
        bundle.factor_binary,
        folds,
        threshold_kappa=config.threshold_kappa,
    )

    metrics.update(
        {
            "experiment": config.name,
            "components": list(component_names),
            "anchor": component_names[0],
            "baseline_is_real": bool(baseline_is_real),
            "delta_macro_f1": metrics["macro_f1"] - anchor_metrics["macro_f1"],
            "delta_tail_macro_f1": metrics["tail_macro_f1"] - anchor_metrics["tail_macro_f1"],
            "delta_mid_macro_f1": metrics["mid_macro_f1"] - anchor_metrics["mid_macro_f1"],
            "delta_head_macro_f1": metrics["head_macro_f1"] - anchor_metrics["head_macro_f1"],
            "anchor_metrics": anchor_metrics,
            "elapsed_minutes": (time.perf_counter() - start) / 60.0,
        }
    )

    weight_table = per_label.copy()
    weight_table = weight_table.rename(columns={"f1": "stack_f1"})
    weight_table["anchor_f1"] = anchor_per_label["f1"].to_numpy()
    weight_table["delta_f1"] = weight_table["stack_f1"] - weight_table["anchor_f1"]
    weight_table["intercept"] = final_parameters.intercepts
    for idx, name in enumerate(component_names[1:]):
        weight_table[f"residual_weight::{name}"] = final_parameters.coefficients[:, idx]

    np.savez_compressed(
        output_dir / "oof_complete.npz",
        logits=crossfit_logits,
        probabilities=sigmoid(crossfit_logits),
        predictions=predictions,
        thresholds=thresholds,
        folds=folds,
        row_ids=bundle.row_ids,
        final_coefficients=final_parameters.coefficients,
        final_intercepts=final_parameters.intercepts,
        component_names=np.asarray(component_names, dtype=object),
    )
    weight_table.to_csv(output_dir / "per_label_weights.csv", index=False)
    _json_dump(asdict(config), output_dir / "config.json")
    _json_dump(fold_parameters, output_dir / "fold_parameters.json")
    _json_dump(metrics, output_dir / "summary.json")
    print(f"[{config.name}] {metrics}")
    return metrics


@dataclass
class RerankerConfig:
    name: str = "B1R_LABELWISE_RERANKER"
    model_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 512
    epochs: int = 2
    train_batch_size: int = 16
    eval_batch_size: int = 64
    grad_accumulation: int = 2
    learning_rate: float = 2.0e-5
    head_learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    min_positive_pairs_per_label: int = 48
    max_positive_pairs_per_label: int = 256
    negative_ratio: float = 1.5
    hard_negative_fraction: float = 0.70
    hard_negative_weight: float = 1.35
    random_negative_weight: float = 0.85
    max_grad_norm: float = 1.0
    seed: int = 42
    num_workers: int = 2
    save_checkpoints: bool = False
    resume: bool = True


@dataclass(frozen=True)
class PairRecord:
    post_index: int
    label_index: int
    target: int
    weight: float


def build_labelwise_training_pairs(
    targets: np.ndarray,
    seed_probabilities: np.ndarray,
    train_indices: np.ndarray,
    config: RerankerConfig,
    *,
    seed: int,
) -> tuple[list[PairRecord], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    records: list[PairRecord] = []
    diagnostics: list[dict[str, Any]] = []

    for label in range(N_LABELS):
        positives = train_indices[targets[train_indices, label] == 1]
        negatives = train_indices[targets[train_indices, label] == 0]
        if len(positives) == 0 or len(negatives) == 0:
            continue

        positive_target = int(
            min(
                config.max_positive_pairs_per_label,
                max(config.min_positive_pairs_per_label, len(positives)),
            )
        )
        selected_positive = rng.choice(
            positives, size=positive_target, replace=positive_target > len(positives)
        )
        negative_target = max(1, int(math.ceil(positive_target * config.negative_ratio)))
        hard_target = min(
            len(negatives), int(round(negative_target * config.hard_negative_fraction))
        )
        negative_order = negatives[np.argsort(seed_probabilities[negatives, label])[::-1]]
        selected_hard = negative_order[:hard_target]
        remaining = np.setdiff1d(negatives, selected_hard, assume_unique=False)
        random_target = negative_target - len(selected_hard)
        if random_target > 0:
            pool = remaining if len(remaining) else negatives
            selected_random = rng.choice(pool, size=random_target, replace=random_target > len(pool))
        else:
            selected_random = np.empty(0, dtype=np.int64)

        records.extend(
            PairRecord(int(index), label, 1, 1.0) for index in selected_positive
        )
        records.extend(
            PairRecord(int(index), label, 0, config.hard_negative_weight)
            for index in selected_hard
        )
        records.extend(
            PairRecord(int(index), label, 0, config.random_negative_weight)
            for index in selected_random
        )
        diagnostics.append(
            {
                "label": FACTOR_LABELS[label],
                "raw_positives": len(positives),
                "positive_pairs": len(selected_positive),
                "hard_negative_pairs": len(selected_hard),
                "random_negative_pairs": len(selected_random),
                "hard_negative_min_probability": (
                    float(seed_probabilities[selected_hard, label].min())
                    if len(selected_hard)
                    else None
                ),
            }
        )

    order = rng.permutation(len(records))
    records = [records[int(index)] for index in order]
    return records, pd.DataFrame(diagnostics)


def build_validation_pairs(indices: Iterable[int]) -> list[PairRecord]:
    return [
        PairRecord(int(post_index), label_index, 0, 1.0)
        for post_index in indices
        for label_index in range(N_LABELS)
    ]


def format_pair_text(post: str, label_index: int) -> str:
    label = FACTOR_LABELS[label_index]
    card = FACTOR_CARDS[label]
    return (
        f"Candidate psychosocial factor: {label}\n"
        f"Definition and boundary: {card}\n"
        "Decision: Does the post contain evidence that this factor is present?\n\n"
        f"Post:\n{post}"
    )


class PairDataset(Dataset):
    def __init__(self, texts: Sequence[str], records: Sequence[PairRecord]):
        self.texts = texts
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "text": format_pair_text(self.texts[record.post_index], record.label_index),
            "post_index": record.post_index,
            "label_index": record.label_index,
            "target": float(record.target),
            "weight": float(record.weight),
        }


class PairCollator:
    def __init__(self, tokenizer: Any, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, items: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            [item["text"] for item in items],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["post_index"] = torch.tensor(
            [item["post_index"] for item in items], dtype=torch.long
        )
        encoded["label_index"] = torch.tensor(
            [item["label_index"] for item in items], dtype=torch.long
        )
        encoded["target"] = torch.tensor([item["target"] for item in items], dtype=torch.float32)
        encoded["weight"] = torch.tensor([item["weight"] for item in items], dtype=torch.float32)
        return encoded


class LabelwiseReranker(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = int(self.backbone.config.hidden_size)
        self.dropout = nn.Dropout(0.10)
        self.head = nn.Linear(hidden, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = {
            key: batch[key]
            for key in ("input_ids", "attention_mask", "token_type_ids")
            if key in batch
        }
        output = self.backbone(**inputs).last_hidden_state
        pooled = b1.masked_mean(output, batch["attention_mask"])
        return self.head(self.dropout(pooled)).squeeze(-1).float()


def _reranker_optimizer(model: LabelwiseReranker, config: RerankerConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [
            {
                "params": model.backbone.parameters(),
                "lr": config.learning_rate,
            },
            {
                "params": model.head.parameters(),
                "lr": config.head_learning_rate,
            },
        ],
        weight_decay=config.weight_decay,
    )


@torch.no_grad()
def evaluate_reranker(
    model: LabelwiseReranker,
    loader: DataLoader,
    targets: np.ndarray,
    validation_indices: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    logits = np.full((len(targets), N_LABELS), np.nan, dtype=np.float32)
    for raw_batch in loader:
        batch = b1.move_batch(raw_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            scores = model(batch)
        post_indices = batch["post_index"].cpu().numpy()
        label_indices = batch["label_index"].cpu().numpy()
        logits[post_indices, label_indices] = scores.float().cpu().numpy()
    validation_logits = logits[validation_indices]
    if np.isnan(validation_logits).any():
        raise RuntimeError("Reranker validation grid is incomplete")
    probabilities = sigmoid(validation_logits)
    validation_targets = targets[validation_indices]
    return {
        "logits": validation_logits,
        "macro_ap": b1.safe_macro_ap(validation_targets, probabilities),
        "macro_f1_at_05": float(
            f1_score(
                validation_targets,
                probabilities >= 0.5,
                average="macro",
                zero_division=0,
            )
        ),
    }


def train_reranker_fold(
    bundle: b1.DataBundle,
    folds: np.ndarray,
    fold: int,
    seed_probabilities: np.ndarray,
    config: RerankerConfig,
    output_dir: Path,
) -> dict[str, Any]:
    b1.seed_everything(config.seed + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_indices = np.where(folds != fold)[0]
    validation_indices = np.where(folds == fold)[0]
    records, pair_diagnostics = build_labelwise_training_pairs(
        bundle.factor_binary,
        seed_probabilities,
        train_indices,
        config,
        seed=config.seed + fold,
    )
    validation_records = build_validation_pairs(validation_indices)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    collator = PairCollator(tokenizer, config.max_length)
    train_loader = DataLoader(
        PairDataset(bundle.texts, records),
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        PairDataset(bundle.texts, validation_records),
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )
    model = LabelwiseReranker(config.model_name).to(device)
    optimizer = _reranker_optimizer(model, config)
    updates_per_epoch = math.ceil(len(train_loader) / config.grad_accumulation)
    total_updates = max(1, updates_per_epoch * config.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * config.warmup_ratio),
        num_training_steps=total_updates,
    )

    best_metric = -np.inf
    history: list[dict[str, float]] = []
    temp_dir = Path(tempfile.mkdtemp(prefix=f"b1r_rerank_fold{fold}_"))
    best_path = temp_dir / "best.pt"
    start = time.perf_counter()

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, raw_batch in enumerate(train_loader):
            batch = b1.move_batch(raw_batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                scores = model(batch)
                per_pair = F.binary_cross_entropy_with_logits(
                    scores, batch["target"], reduction="none"
                )
                loss = (per_pair * batch["weight"]).sum() / batch["weight"].sum().clamp_min(1.0)
                scaled_loss = loss / config.grad_accumulation
            scaled_loss.backward()
            running_loss += float(loss.detach().cpu())
            should_step = (step + 1) % config.grad_accumulation == 0 or step + 1 == len(train_loader)
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        validation = evaluate_reranker(
            model, validation_loader, bundle.factor_binary, validation_indices, device
        )
        selection_metric = 0.8 * validation["macro_ap"] + 0.2 * validation["macro_f1_at_05"]
        record = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(1, len(train_loader)),
            "val_macro_ap": float(validation["macro_ap"]),
            "val_macro_f1_at_05": float(validation["macro_f1_at_05"]),
            "selection_metric": float(selection_metric),
        }
        history.append(record)
        print(f"[{config.name}] fold={fold} {record}")
        if selection_metric > best_metric:
            best_metric = selection_metric
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    validation = evaluate_reranker(
        model, validation_loader, bundle.factor_binary, validation_indices, device
    )
    elapsed = time.perf_counter() - start

    if config.save_checkpoints:
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / f"fold_{fold}.pt")
    pair_diagnostics.to_csv(output_dir / f"fold_{fold}_pairs.csv", index=False)

    result = {
        "idx": validation_indices,
        "logits": validation["logits"],
        "history": history,
        "elapsed_seconds": elapsed,
        "train_pairs": len(records),
    }
    del model, optimizer, scheduler, train_loader, validation_loader
    shutil.rmtree(temp_dir, ignore_errors=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_reranker_experiment(
    bundle: b1.DataBundle,
    folds: np.ndarray,
    seed_logits: np.ndarray,
    config: RerankerConfig,
    artifact_root: str | Path,
) -> dict[str, Any]:
    output_dir = Path(artifact_root) / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(asdict(config), output_dir / "config.json")
    seed_probabilities = sigmoid(seed_logits)
    seed_fingerprint = hashlib.sha1(
        np.round(np.asarray(seed_logits, dtype=np.float32), 4).tobytes()
    ).hexdigest()
    seed_metadata_path = output_dir / "hard_negative_seed.json"
    if seed_metadata_path.exists():
        previous = json.loads(seed_metadata_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != seed_fingerprint:
            raise RuntimeError(
                f"{config.name} 已有的 folds 来自另一组 hard-negative seed logits。"
                "请更换 config.name 后重跑，避免错误复用旧 reranker OOF。"
            )
    _json_dump(
        {"fingerprint": seed_fingerprint, "shape": list(seed_logits.shape)},
        seed_metadata_path,
    )
    logits = np.full((len(bundle.texts), N_LABELS), np.nan, dtype=np.float32)
    elapsed_by_fold: list[float] = []
    train_pairs_by_fold: dict[str, int] = {}
    suite_start = time.perf_counter()

    for position, fold in enumerate(sorted(np.unique(folds))):
        fold = int(fold)
        fold_path = output_dir / f"fold_{fold}_oof.npz"
        history_path = output_dir / f"fold_{fold}_history.json"
        if config.resume and fold_path.exists():
            saved = np.load(fold_path)
            idx = saved["idx"].astype(np.int64)
            logits[idx] = saved["logits"]
            elapsed = float(saved["elapsed_seconds"])
            elapsed_by_fold.append(elapsed)
            train_pairs_by_fold[str(fold)] = int(saved.get("train_pairs", 0))
            print(f"[{config.name}] resumed fold {fold} ({elapsed / 60:.1f} min)")
            continue

        result = train_reranker_fold(
            bundle, folds, fold, seed_probabilities, config, output_dir
        )
        idx = np.asarray(result["idx"], dtype=np.int64)
        logits[idx] = result["logits"]
        elapsed = float(result["elapsed_seconds"])
        elapsed_by_fold.append(elapsed)
        train_pairs_by_fold[str(fold)] = int(result["train_pairs"])
        np.savez_compressed(
            fold_path,
            idx=idx,
            logits=result["logits"],
            elapsed_seconds=np.asarray(elapsed),
            train_pairs=np.asarray(result["train_pairs"]),
        )
        _json_dump(result["history"], history_path)
        median_minutes = float(np.median(elapsed_by_fold) / 60.0)
        remaining = len(np.unique(folds)) - position - 1
        print(
            f"[{config.name}] fold {fold}: {elapsed / 60:.1f} min; "
            f"estimated {remaining * median_minutes:.1f} min remaining"
        )

    if np.isnan(logits).any():
        raise RuntimeError("Reranker OOF logits contain missing values")
    metrics, per_label, predictions, thresholds = evaluate_oof_logits(
        logits, bundle.factor_binary, folds
    )
    metrics.update(
        {
            "experiment": config.name,
            "elapsed_minutes": (time.perf_counter() - suite_start) / 60.0,
            "recorded_fold_minutes": float(np.sum(elapsed_by_fold) / 60.0),
            "median_fold_minutes": float(np.median(elapsed_by_fold) / 60.0),
            "train_pairs_by_fold": train_pairs_by_fold,
            "model_name": config.model_name,
            "max_length": config.max_length,
            "epochs": config.epochs,
        }
    )
    np.savez_compressed(
        output_dir / "oof_complete.npz",
        logits=logits,
        probabilities=sigmoid(logits),
        predictions=predictions,
        thresholds=thresholds,
        folds=folds,
        row_ids=bundle.row_ids,
    )
    per_label.to_csv(output_dir / "per_label_metrics.csv", index=False)
    _json_dump(metrics, output_dir / "summary.json")
    print(f"[{config.name}] summary: {metrics}")
    return metrics


def make_decision(
    core_summary: dict[str, Any],
    final_summary: dict[str, Any] | None,
    *,
    baseline_path: str | None,
    artifact_root: str | Path,
) -> dict[str, Any]:
    candidates = [core_summary] + ([final_summary] if final_summary is not None else [])

    def passes(summary: dict[str, Any]) -> bool:
        return bool(
            summary["delta_macro_f1"] >= 0.003
            and summary["delta_tail_macro_f1"] >= 0.0
            and summary["delta_head_macro_f1"] >= -0.005
        )

    eligible = [summary for summary in candidates if passes(summary)]
    candidate = max(eligible, key=lambda item: item["macro_f1"]) if eligible else max(
        candidates, key=lambda item: item["macro_f1"]
    )
    accepted = bool(eligible)
    decision = {
        "version": "B1-R",
        "baseline_path": baseline_path,
        "baseline_is_real": bool(candidate.get("baseline_is_real", False)),
        "eligible_for_baseline_upgrade_claim": bool(
            candidate.get("baseline_is_real", False) and accepted
        ),
        "accepted": accepted,
        "recommended_factor_system": candidate["experiment"] if accepted else candidate["anchor"],
        "acceptance_rule": {
            "delta_macro_f1_min": 0.003,
            "delta_tail_macro_f1_min": 0.0,
            "delta_head_macro_f1_min": -0.005,
        },
        "core": core_summary,
        "final": final_summary,
    }
    output_path = Path(artifact_root) / "B1R_DECISION.json"
    _json_dump(decision, output_path)
    return decision


def environment_report() -> dict[str, Any]:
    return {
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "factor_labels": N_LABELS,
    }
