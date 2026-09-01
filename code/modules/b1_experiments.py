"""CARE-MA B1 factor/graph ablation harness for Google Colab.

This module is intentionally standalone.  The companion notebook imports it from
the same Google Drive directory.  It implements:

* leak-aware user grouped folds;
* BCE/ASL document and label-wise clause-MIL factor models;
* ABALONE-inspired prototype + queue multi-label contrastive loss;
* weak count loss from raw duplicate factor annotations;
* cross-fitted per-label thresholds;
* an optional risk OOF model for graph experiments;
* nested-cross-fitted low-rank and one-layer prior-GAT meta graphs.

The contrastive implementation is an engineering adaptation, not a verbatim
reproduction of the original ABALONE training recipe: it uses a detached FIFO
queue instead of a second momentum encoder to fit comfortably into one A100.
"""

from __future__ import annotations

import ast
import dataclasses
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
import time
import unicodedata
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup


FACTOR_LABELS = [
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
]

RISK_LABELS = ["Indicator", "Ideation", "Behavior", "Attempt"]
RISK_TO_ID = {name.lower(): idx for idx, name in enumerate(RISK_LABELS)}
LABEL_TO_ID = {name: idx for idx, name in enumerate(FACTOR_LABELS)}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, default=str)


def normalize_text(text: Any) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def canonical_factor(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _coerce_binary_series(series: pd.Series, column: str) -> np.ndarray:
    """Convert common clean-CSV binary encodings to an int8 array."""
    if pd.api.types.is_bool_dtype(series):
        values = series.fillna(False).astype(np.int8).to_numpy()
    elif pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").fillna(0).to_numpy()
        if not np.isin(numeric, [0, 1]).all():
            warnings.warn(f"Column {column!r} contains values outside 0/1; values > 0 are treated as positive.")
        values = (numeric > 0).astype(np.int8)
    else:
        mapping = {
            "": 0,
            "0": 0,
            "0.0": 0,
            "false": 0,
            "no": 0,
            "none": 0,
            "nan": 0,
            "1": 1,
            "1.0": 1,
            "true": 1,
            "yes": 1,
        }
        normalized = series.fillna("").astype(str).str.strip().str.lower()
        unknown = sorted(set(normalized) - set(mapping))
        if unknown:
            raise ValueError(f"Column {column!r} has unsupported binary values: {unknown[:10]}")
        values = normalized.map(mapping).to_numpy(dtype=np.int8)
    return values


def parse_factor_cell(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set, np.ndarray)):
        raw = list(value)
    else:
        text = str(value).strip()
        if not text or text.lower() in {"none", "nan", "[]"}:
            return []
        try:
            parsed = ast.literal_eval(text)
            raw = list(parsed) if isinstance(parsed, (list, tuple, set)) else [parsed]
        except (ValueError, SyntaxError):
            separator = ";" if ";" in text else ","
            raw = text.split(separator)
    return [canonical_factor(item) for item in raw if str(item).strip()]


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    normalized = {re.sub(r"[^a-z0-9]+", "", str(c).lower()): c for c in df.columns}
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]+", "", candidate.lower())
        if key in normalized:
            return normalized[key]
    raise KeyError(f"Could not find any of {list(candidates)} in columns {df.columns.tolist()}")


def resolve_train_path(root: str | Path, explicit: str | Path | None = None) -> Path:
    root = Path(root)
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            root / "train.xlsx",
            root / "ieee" / "train.xlsx",
            root / "train_clean.csv",
            root / "train.csv",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No training file found. Checked: " + ", ".join(map(str, candidates)))


@dataclass
class DataBundle:
    frame: pd.DataFrame
    texts: list[str]
    row_ids: np.ndarray
    user_ids: np.ndarray
    factor_binary: np.ndarray
    factor_count: np.ndarray
    risk_ids: np.ndarray
    clause_counts: np.ndarray
    count_signal_available: bool
    source_path: str


def load_training_data(root: str | Path, explicit: str | Path | None = None) -> DataBundle:
    path = resolve_train_path(root, explicit)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    text_col = _find_column(df, ["post", "text", "content"])
    user_col = _find_column(df, ["anon_user_id", "user_id", "userid"])
    risk_col = _find_column(df, ["suicide risk", "risk", "risk_label", "risk_level"])
    try:
        row_col = _find_column(df, ["row_id", "id"])
    except KeyError:
        row_col = "__generated_row_id"
        df[row_col] = [f"row_{idx:05d}" for idx in range(len(df))]

    y = np.zeros((len(df), len(FACTOR_LABELS)), dtype=np.int8)
    counts = np.zeros_like(y, dtype=np.int16)
    factor_source = "raw_list"
    try:
        factor_col = _find_column(df, ["factors", "factor", "labels"])
    except KeyError:
        factor_col = None

    if factor_col is not None:
        parsed = [parse_factor_cell(value) for value in df[factor_col].tolist()]
        unknown = sorted({label for labels in parsed for label in labels if label not in LABEL_TO_ID})
        if unknown:
            raise ValueError(f"Unknown factor labels after normalization: {unknown}")
        for i, labels in enumerate(parsed):
            counter = Counter(labels)
            for label, count in counter.items():
                j = LABEL_TO_ID[label]
                y[i, j] = 1
                counts[i, j] = count
    else:
        factor_source = "wide_binary"
        wide_columns: dict[str, str] = {}
        for column in df.columns:
            text = str(column).strip()
            if text.lower().startswith("f_"):
                wide_columns[canonical_factor(text[2:])] = column
        missing = [label for label in FACTOR_LABELS if label not in wide_columns]
        extra = sorted(set(wide_columns) - set(FACTOR_LABELS))
        if missing:
            raise KeyError(
                "No raw factors column was found, and the clean wide table is missing factor columns: "
                + repr(missing)
            )
        if extra:
            warnings.warn(f"Ignoring unrecognized f_* columns: {extra}")
        for label, j in LABEL_TO_ID.items():
            y[:, j] = _coerce_binary_series(df[wide_columns[label]], wide_columns[label])
        # A wide binary table has already discarded duplicate annotation counts.
        counts = y.astype(np.int16, copy=True)

    risks = []
    for value in df[risk_col].tolist():
        key = str(value).strip().lower()
        if key in RISK_TO_ID:
            risks.append(RISK_TO_ID[key])
            continue
        try:
            numeric = int(float(value))
        except (TypeError, ValueError):
            numeric = -1
        if numeric not in range(len(RISK_LABELS)):
            raise ValueError(f"Unknown risk label: {value!r}")
        risks.append(numeric)

    texts = df[text_col].fillna("").astype(str).tolist()
    clause_counts = np.asarray([len(split_clauses(text)) for text in texts], dtype=np.int16)
    count_signal_available = factor_source == "raw_list" and bool(np.any(counts > 1))
    if not count_signal_available:
        warnings.warn(
            "No duplicate factor annotations were found. Count experiments will be disabled. "
            "Use the raw train.xlsx rather than a deduplicated train_clean.csv."
        )

    return DataBundle(
        frame=df,
        texts=texts,
        row_ids=df[row_col].astype(str).to_numpy(),
        user_ids=df[user_col].astype(str).to_numpy(),
        factor_binary=y,
        factor_count=counts,
        risk_ids=np.asarray(risks, dtype=np.int64),
        clause_counts=clause_counts,
        count_signal_available=count_signal_available,
        source_path=str(path),
    )


class DSU:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def make_leak_safe_folds(bundle: DataBundle, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Group users and exact normalized duplicates, then greedily balance labels."""
    dsu = DSU()
    for user, text in zip(bundle.user_ids, bundle.texts):
        digest = hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()
        dsu.union(f"u:{user}", f"t:{digest}")

    components: dict[str, list[int]] = defaultdict(list)
    for idx, user in enumerate(bundle.user_ids):
        components[dsu.find(f"u:{user}")].append(idx)

    risk_onehot = np.eye(len(RISK_LABELS), dtype=np.float32)[bundle.risk_ids]
    all_targets = np.concatenate([bundle.factor_binary, risk_onehot], axis=1).astype(np.float64)
    target_per_fold = all_targets.sum(axis=0) / n_splits
    size_target = len(bundle.texts) / n_splits

    rng = np.random.default_rng(seed)
    groups = []
    rarity = 1.0 / np.sqrt(all_targets.sum(axis=0) + 1.0)
    for root, indices in components.items():
        label_sum = all_targets[indices].sum(axis=0)
        score = float((label_sum * rarity).sum() + 0.05 * len(indices))
        groups.append((root, indices, label_sum, score, rng.random()))
    groups.sort(key=lambda item: (item[3], len(item[1]), item[4]), reverse=True)

    fold_label = np.zeros((n_splits, all_targets.shape[1]), dtype=np.float64)
    fold_size = np.zeros(n_splits, dtype=np.float64)
    assignment: dict[str, int] = {}

    for root, indices, label_sum, _, _ in groups:
        best_fold, best_cost = None, None
        for fold in range(n_splits):
            candidate_label = fold_label.copy()
            candidate_size = fold_size.copy()
            candidate_label[fold] += label_sum
            candidate_size[fold] += len(indices)
            label_cost = np.square(candidate_label - target_per_fold).sum(axis=0)
            label_cost = float((label_cost / (target_per_fold + 1.0)).mean())
            size_cost = float(np.square(candidate_size - size_target).mean() / (size_target + 1.0))
            empty_bonus = -0.05 if fold_size[fold] == 0 else 0.0
            cost = label_cost + 0.25 * size_cost + empty_bonus
            if best_cost is None or cost < best_cost:
                best_fold, best_cost = fold, cost
        assignment[root] = int(best_fold)
        fold_label[best_fold] += label_sum
        fold_size[best_fold] += len(indices)

    folds = np.empty(len(bundle.texts), dtype=np.int64)
    for root, indices in components.items():
        folds[indices] = assignment[root]

    for fold in range(n_splits):
        assert np.any(folds == fold), f"Fold {fold} is empty"
    for user in np.unique(bundle.user_ids):
        assert len(np.unique(folds[bundle.user_ids == user])) == 1
    return folds


_CLAUSE_BOUNDARY = re.compile(r"(?:[.!?]+[\"')\]]*\s+)|(?:\n+)|(?:;\s+)")


def split_clauses(text: str, max_clauses: int = 32) -> list[tuple[int, int]]:
    text = str(text or "")
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _CLAUSE_BOUNDARY.finditer(text):
        end = match.end()
        if text[start:end].strip():
            spans.append((start, end))
        start = end
    if text[start:].strip():
        spans.append((start, len(text)))
    if not spans:
        spans = [(0, max(1, len(text)))]

    expanded: list[tuple[int, int]] = []
    for left, right in spans:
        if right - left <= 600:
            expanded.append((left, right))
            continue
        cursor = left
        while cursor < right:
            stop = min(cursor + 500, right)
            if stop < right:
                boundary = text.rfind(" ", cursor + 250, stop)
                if boundary > cursor:
                    stop = boundary
            expanded.append((cursor, stop))
            cursor = stop

    if len(expanded) > max_clauses:
        head = expanded[: max_clauses - 1]
        head.append((expanded[max_clauses - 1][0], expanded[-1][1]))
        expanded = head
    return expanded


class TextDataset(Dataset):
    def __init__(self, bundle: DataBundle, indices: np.ndarray):
        self.bundle = bundle
        self.indices = np.asarray(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, Any]:
        idx = int(self.indices[position])
        return {
            "idx": idx,
            "text": self.bundle.texts[idx],
            "factor_y": self.bundle.factor_binary[idx].astype(np.float32),
            "factor_count": self.bundle.factor_count[idx].astype(np.float32),
            "risk_y": int(self.bundle.risk_ids[idx]),
            "clauses": split_clauses(self.bundle.texts[idx]),
            "clause_count": int(self.bundle.clause_counts[idx]),
        }


class BatchCollator:
    def __init__(self, tokenizer: Any, max_length: int, use_clauses: bool):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_clauses = use_clauses

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            [item["text"] for item in items],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")
        batch: dict[str, torch.Tensor] = {
            **encoded,
            "idx": torch.tensor([item["idx"] for item in items], dtype=torch.long),
            "factor_y": torch.tensor(np.stack([item["factor_y"] for item in items])),
            "factor_count": torch.tensor(np.stack([item["factor_count"] for item in items])),
            "risk_y": torch.tensor([item["risk_y"] for item in items], dtype=torch.long),
            "clause_count": torch.tensor([item["clause_count"] for item in items], dtype=torch.float32),
        }
        if self.use_clauses:
            max_clauses = max(len(item["clauses"]) for item in items)
            token_count = offsets.shape[1]
            clause_token_mask = torch.zeros(
                len(items), max_clauses, token_count, dtype=torch.float32
            )
            for b, item in enumerate(items):
                for s, (left, right) in enumerate(item["clauses"]):
                    token_left = offsets[b, :, 0]
                    token_right = offsets[b, :, 1]
                    valid = (token_right > token_left) & (token_right > left) & (token_left < right)
                    valid &= encoded["attention_mask"][b].bool()
                    clause_token_mask[b, s, valid] = 1.0
            batch["clause_token_mask"] = clause_token_mask
        return batch


def load_backbone(model_name: str) -> nn.Module:
    kwargs: dict[str, Any] = {}
    if torch.cuda.is_available():
        # Keep master weights in fp32 for stable fine-tuning; the training loop
        # uses bf16 autocast for activations and matrix multiplications.
        kwargs["attn_implementation"] = "sdpa"
    try:
        return AutoModel.from_pretrained(model_name, **kwargs)
    except (TypeError, ValueError):
        kwargs.pop("attn_implementation", None)
        return AutoModel.from_pretrained(model_name, **kwargs)


def masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    weight = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0, clip: float = 0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(logits.dtype)
        probs_pos = torch.sigmoid(logits)
        probs_neg = 1.0 - probs_pos
        if self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)
        log_loss = targets * torch.log(probs_pos.clamp_min(1e-8))
        log_loss += (1.0 - targets) * torch.log(probs_neg.clamp_min(1e-8))
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = probs_pos * targets + probs_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            log_loss *= torch.pow(1.0 - pt, gamma)
        return -log_loss.mean()


def weak_count_loss(
    support_logits: torch.Tensor,
    clause_valid: torch.Tensor,
    raw_counts: torch.Tensor,
    binary_targets: torch.Tensor,
    count_cap: int = 5,
    negative_weight: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor]:
    support = torch.sigmoid(support_logits) * clause_valid[:, None, :].to(support_logits.dtype)
    pred_count = support.sum(dim=-1)
    target_count = raw_counts.to(pred_count.dtype).clamp(max=count_cap)
    elem = F.smooth_l1_loss(
        torch.log1p(pred_count), torch.log1p(target_count), reduction="none"
    )
    positive = binary_targets.bool()
    negative = ~positive
    pos_loss = elem[positive].mean() if positive.any() else elem.new_zeros(())
    neg_loss = elem[negative].mean() if negative.any() else elem.new_zeros(())
    return pos_loss + negative_weight * neg_loss, pred_count


class MultiLabelPrototypeQueueLoss(nn.Module):
    """ABALONE-inspired multi-label supervised contrastive loss.

    A detached FIFO queue is used instead of a full momentum encoder.  Each
    positive label also has a trainable prototype, guaranteeing a positive pair.
    """

    def __init__(
        self,
        num_labels: int,
        embedding_dim: int = 256,
        queue_size: int = 512,
        temperature: float = 0.10,
        beta: float = 0.10,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.embedding_dim = embedding_dim
        self.queue_size = queue_size
        self.temperature = temperature
        self.beta = beta
        self.prototypes = nn.Parameter(torch.randn(num_labels, embedding_dim) * 0.02)
        self.register_buffer("queue", torch.zeros(queue_size, embedding_dim))
        self.register_buffer("queue_labels", torch.zeros(queue_size, num_labels, dtype=torch.bool))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_filled", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        embeddings = F.normalize(embeddings.detach(), dim=-1)
        labels = labels.detach().bool()
        batch_size = embeddings.shape[0]
        if batch_size >= self.queue_size:
            self.queue.copy_(embeddings[-self.queue_size :])
            self.queue_labels.copy_(labels[-self.queue_size :])
            self.queue_ptr.zero_()
            self.queue_filled.fill_(self.queue_size)
            return
        ptr = int(self.queue_ptr.item())
        end = ptr + batch_size
        if end <= self.queue_size:
            self.queue[ptr:end] = embeddings
            self.queue_labels[ptr:end] = labels
        else:
            first = self.queue_size - ptr
            self.queue[ptr:] = embeddings[:first]
            self.queue_labels[ptr:] = labels[:first]
            self.queue[: end - self.queue_size] = embeddings[first:]
            self.queue_labels[: end - self.queue_size] = labels[first:]
        self.queue_ptr.fill_(end % self.queue_size)
        self.queue_filled.fill_(min(self.queue_size, int(self.queue_filled.item()) + batch_size))

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        z = F.normalize(embeddings.float(), dim=-1)
        labels_bool = labels.bool()
        prototypes = F.normalize(self.prototypes.float(), dim=-1)
        filled = int(self.queue_filled.item())
        queue = self.queue[:filled].float()
        queue_labels = self.queue_labels[:filled]

        sample_candidates = torch.cat([z, queue], dim=0)
        sample_labels = torch.cat([labels_bool, queue_labels], dim=0)
        all_candidates = torch.cat([sample_candidates, prototypes], dim=0)
        losses: list[torch.Tensor] = []

        for i in range(z.shape[0]):
            positive_labels = torch.where(labels_bool[i])[0]
            if positive_labels.numel() == 0:
                continue
            similarity = (z[i : i + 1] @ all_candidates.T).squeeze(0) / self.temperature
            instance_valid = torch.ones(sample_candidates.shape[0], dtype=torch.bool, device=z.device)
            instance_valid[i] = False
            denominator_valid = torch.cat(
                [instance_valid, torch.ones(self.num_labels, dtype=torch.bool, device=z.device)]
            )
            denominator_log_weight = torch.cat(
                [
                    torch.full(
                        (sample_candidates.shape[0],),
                        math.log(self.beta),
                        dtype=similarity.dtype,
                        device=z.device,
                    ),
                    torch.zeros(self.num_labels, dtype=similarity.dtype, device=z.device),
                ]
            )
            log_den = torch.logsumexp(
                similarity[denominator_valid] + denominator_log_weight[denominator_valid], dim=0
            )

            per_label: list[torch.Tensor] = []
            for label_id in positive_labels.tolist():
                positive_instances = sample_labels[:, label_id] & instance_valid
                positive_indices = torch.where(positive_instances)[0]
                positive_logits: list[torch.Tensor] = []
                positive_weights: list[torch.Tensor] = []
                for candidate_idx in positive_indices.tolist():
                    union = (labels_bool[i] | sample_labels[candidate_idx]).sum().clamp_min(1)
                    positive_logits.append(similarity[candidate_idx])
                    positive_weights.append(1.0 / union.float())
                prototype_idx = sample_candidates.shape[0] + label_id
                positive_logits.append(similarity[prototype_idx])
                positive_weights.append(similarity.new_tensor(1.0))
                weights = torch.stack(positive_weights)
                log_probs = torch.stack(positive_logits) - log_den
                per_label.append(-(weights * log_probs).sum() / weights.sum().clamp_min(1e-8))
            losses.append(torch.stack(per_label).mean())

        result = torch.stack(losses).mean() if losses else z.new_zeros(())
        self.enqueue(z, labels_bool)
        return result


@dataclass
class ExperimentConfig:
    name: str
    model_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 512
    epochs: int = 2
    train_batch_size: int = 8
    eval_batch_size: int = 16
    grad_accumulation: int = 2
    backbone_lr: float = 1.5e-5
    head_lr: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    seed: int = 42
    n_splits: int = 5
    loss_name: str = "asl"
    use_mil: bool = False
    use_contrastive: bool = False
    use_count: bool = False
    lambda_contrastive: float = 0.10
    lambda_count: float = 0.02
    count_cap: int = 5
    count_negative_weight: float = 0.10
    projection_dim: int = 256
    queue_size: int = 512
    contrastive_temperature: float = 0.10
    contrastive_beta: float = 0.10
    mil_temperature: float = 0.50
    max_grad_norm: float = 1.0
    num_workers: int = 2
    save_checkpoints: bool = False
    resume: bool = True


class FactorModel(nn.Module):
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.backbone = load_backbone(config.model_name)
        hidden = int(self.backbone.config.hidden_size)
        self.dropout = nn.Dropout(0.10)
        self.doc_head = nn.Linear(hidden, len(FACTOR_LABELS))
        self.label_queries = nn.Parameter(torch.randn(len(FACTOR_LABELS), hidden) * 0.02)
        self.label_bias = nn.Parameter(torch.zeros(len(FACTOR_LABELS)))
        self.pool_mix = nn.Parameter(torch.zeros(len(FACTOR_LABELS)))
        self.projection = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, config.projection_dim)
        )
        self.contrastive = (
            MultiLabelPrototypeQueueLoss(
                len(FACTOR_LABELS),
                embedding_dim=config.projection_dim,
                queue_size=config.queue_size,
                temperature=config.contrastive_temperature,
                beta=config.contrastive_beta,
            )
            if config.use_contrastive
            else None
        )

    @torch.no_grad()
    def initialize_label_queries(self, tokenizer: Any, device: torch.device) -> None:
        encoded = tokenizer(
            FACTOR_LABELS,
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        self.backbone.eval()
        outputs = self.backbone(**encoded)
        vectors = masked_mean(outputs.last_hidden_state, encoded["attention_mask"])
        self.label_queries.copy_(vectors.to(self.label_queries.dtype))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        model_inputs = {
            key: batch[key]
            for key in ("input_ids", "attention_mask", "token_type_ids")
            if key in batch
        }
        outputs = self.backbone(**model_inputs)
        hidden = outputs.last_hidden_state
        document = self.dropout(masked_mean(hidden, batch["attention_mask"]))
        result: dict[str, torch.Tensor] = {
            "embedding": self.projection(document).float(),
        }
        if not self.config.use_mil:
            result["logits"] = self.doc_head(document).float()
            return result

        clause_token_mask = batch["clause_token_mask"].to(hidden.dtype)
        denom = clause_token_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        clause_vectors = torch.einsum("bst,bth->bsh", clause_token_mask, hidden) / denom
        queries = F.normalize(self.label_queries.float(), dim=-1)
        clause_vectors_norm = F.normalize(clause_vectors.float(), dim=-1)
        support_logits = torch.einsum("bsh,lh->bls", clause_vectors_norm, queries)
        support_logits = support_logits * math.sqrt(queries.shape[-1]) / 8.0
        support_logits = support_logits + self.label_bias[None, :, None]
        clause_valid = clause_token_mask.sum(dim=-1) > 0
        masked_scores = support_logits.masked_fill(~clause_valid[:, None, :], -1e4)
        max_score = masked_scores.max(dim=-1).values
        tau = self.config.mil_temperature
        n_valid = clause_valid.sum(dim=-1, keepdim=True).clamp_min(1).float()
        smooth_max = tau * torch.logsumexp(masked_scores / tau, dim=-1)
        smooth_max = smooth_max - tau * torch.log(n_valid)
        mix = torch.sigmoid(self.pool_mix)[None, :]
        result["logits"] = mix * max_score + (1.0 - mix) * smooth_max
        result["support_logits"] = support_logits
        result["clause_valid"] = clause_valid
        return result


class RiskModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.backbone = load_backbone(model_name)
        hidden = int(self.backbone.config.hidden_size)
        self.dropout = nn.Dropout(0.10)
        self.head = nn.Linear(hidden, len(RISK_LABELS))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        model_inputs = {
            key: batch[key]
            for key in ("input_ids", "attention_mask", "token_type_ids")
            if key in batch
        }
        outputs = self.backbone(**model_inputs)
        document = masked_mean(outputs.last_hidden_state, batch["attention_mask"])
        return self.head(self.dropout(document)).float()


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def make_optimizer(model: nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    backbone, heads = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (backbone if name.startswith("backbone.") else heads).append(param)
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": config.backbone_lr},
            {"params": heads, "lr": config.head_lr},
        ],
        weight_decay=config.weight_decay,
    )


def safe_macro_ap(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    values = []
    for label in range(y_true.shape[1]):
        if y_true[:, label].sum() == 0:
            continue
        values.append(average_precision_score(y_true[:, label], probabilities[:, label]))
    return float(np.mean(values)) if values else 0.0


def evaluate_factor_loader(
    model: FactorModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray | float]:
    model.eval()
    all_idx, all_logits, all_targets, all_counts, all_pred_counts, all_clause_counts = [], [], [], [], [], []
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(batch)
            all_idx.append(batch["idx"].cpu().numpy())
            all_logits.append(output["logits"].float().cpu().numpy())
            all_targets.append(batch["factor_y"].cpu().numpy())
            all_counts.append(batch["factor_count"].cpu().numpy())
            all_clause_counts.append(batch["clause_count"].cpu().numpy())
            if "support_logits" in output:
                pred_count = (
                    torch.sigmoid(output["support_logits"])
                    * output["clause_valid"][:, None, :].to(output["support_logits"].dtype)
                ).sum(dim=-1)
                all_pred_counts.append(pred_count.float().cpu().numpy())
            else:
                all_pred_counts.append(np.full_like(all_targets[-1], np.nan))
    logits = np.concatenate(all_logits)
    targets = np.concatenate(all_targets).astype(np.int8)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    return {
        "idx": np.concatenate(all_idx),
        "logits": logits,
        "probabilities": probabilities,
        "targets": targets,
        "raw_counts": np.concatenate(all_counts),
        "pred_counts": np.concatenate(all_pred_counts),
        "clause_counts": np.concatenate(all_clause_counts),
        "macro_ap": safe_macro_ap(targets, probabilities),
        "macro_f1_at_05": float(f1_score(targets, probabilities >= 0.5, average="macro", zero_division=0)),
    }


def train_factor_fold(
    bundle: DataBundle,
    folds: np.ndarray,
    fold: int,
    config: ExperimentConfig,
    output_dir: Path,
) -> dict[str, Any]:
    seed_everything(config.seed + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    train_idx = np.where(folds != fold)[0]
    val_idx = np.where(folds == fold)[0]
    collator = BatchCollator(tokenizer, config.max_length, use_clauses=config.use_mil)
    train_loader = DataLoader(
        TextDataset(bundle, train_idx),
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )
    val_loader = DataLoader(
        TextDataset(bundle, val_idx),
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )

    model = FactorModel(config).to(device)
    if config.use_mil:
        model.initialize_label_queries(tokenizer, device)
    asl = AsymmetricLoss()
    optimizer = make_optimizer(model, config)
    updates_per_epoch = math.ceil(len(train_loader) / config.grad_accumulation)
    total_updates = max(1, updates_per_epoch * config.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * config.warmup_ratio),
        num_training_steps=total_updates,
    )

    best_metric = -np.inf
    history: list[dict[str, float]] = []
    temp_dir = Path(tempfile.mkdtemp(prefix=f"b1_{config.name}_fold{fold}_"))
    best_path = temp_dir / "best.pt"
    start_time = time.perf_counter()

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = defaultdict(float)
        for step, raw_batch in enumerate(train_loader):
            batch = move_batch(raw_batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(batch)
                if config.loss_name == "bce":
                    classification_loss = F.binary_cross_entropy_with_logits(
                        output["logits"], batch["factor_y"]
                    )
                else:
                    classification_loss = asl(output["logits"], batch["factor_y"])
                loss = classification_loss
                contrastive_loss = loss.new_zeros(())
                count_loss = loss.new_zeros(())
                if config.use_contrastive and model.contrastive is not None:
                    contrastive_loss = model.contrastive(output["embedding"], batch["factor_y"])
                    warm = min(1.0, (epoch * len(train_loader) + step + 1) / max(1, 0.10 * len(train_loader) * config.epochs))
                    loss = loss + config.lambda_contrastive * warm * contrastive_loss
                if config.use_count:
                    count_loss, _ = weak_count_loss(
                        output["support_logits"],
                        output["clause_valid"],
                        batch["factor_count"],
                        batch["factor_y"],
                        count_cap=config.count_cap,
                        negative_weight=config.count_negative_weight,
                    )
                    count_warm = 0.0 if epoch == 0 else min(1.0, epoch / 2.0)
                    loss = loss + config.lambda_count * count_warm * count_loss
                scaled_loss = loss / config.grad_accumulation
            scaled_loss.backward()
            running["loss"] += float(loss.detach().cpu())
            running["classification"] += float(classification_loss.detach().cpu())
            running["contrastive"] += float(contrastive_loss.detach().cpu())
            running["count"] += float(count_loss.detach().cpu())
            should_step = (step + 1) % config.grad_accumulation == 0 or step + 1 == len(train_loader)
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        validation = evaluate_factor_loader(model, val_loader, device)
        selection_metric = 0.7 * float(validation["macro_ap"]) + 0.3 * float(validation["macro_f1_at_05"])
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": running["loss"] / max(1, len(train_loader)),
            "train_classification": running["classification"] / max(1, len(train_loader)),
            "train_contrastive": running["contrastive"] / max(1, len(train_loader)),
            "train_count": running["count"] / max(1, len(train_loader)),
            "val_macro_ap": float(validation["macro_ap"]),
            "val_macro_f1_at_05": float(validation["macro_f1_at_05"]),
            "selection_metric": selection_metric,
        }
        history.append(epoch_record)
        print(f"[{config.name}] fold={fold} {epoch_record}")
        if selection_metric > best_metric:
            best_metric = selection_metric
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    validation = evaluate_factor_loader(model, val_loader, device)
    elapsed = time.perf_counter() - start_time

    if config.save_checkpoints:
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / f"fold_{fold}.pt")

    result = {
        "idx": validation["idx"],
        "logits": validation["logits"],
        "pred_counts": validation["pred_counts"],
        "elapsed_seconds": elapsed,
        "history": history,
    }
    del model, optimizer, scheduler, train_loader, val_loader
    shutil.rmtree(temp_dir, ignore_errors=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _thresholds_from_training(
    probabilities: np.ndarray,
    targets: np.ndarray,
    kappa: float = 20.0,
) -> np.ndarray:
    grid = np.linspace(0.05, 0.80, 76)
    global_scores = [
        f1_score(targets, probabilities >= threshold, average="macro", zero_division=0)
        for threshold in grid
    ]
    global_threshold = float(grid[int(np.argmax(global_scores))])
    thresholds = np.full(targets.shape[1], global_threshold, dtype=np.float32)
    for label in range(targets.shape[1]):
        support = int(targets[:, label].sum())
        if support == 0:
            continue
        scores = [
            f1_score(targets[:, label], probabilities[:, label] >= threshold, zero_division=0)
            for threshold in grid
        ]
        local = float(grid[int(np.argmax(scores))])
        weight = support / (support + kappa)
        thresholds[label] = weight * local + (1.0 - weight) * global_threshold
    return thresholds


def nested_threshold_predictions(
    probabilities: np.ndarray,
    targets: np.ndarray,
    folds: np.ndarray,
    kappa: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.zeros_like(targets, dtype=np.int8)
    fold_thresholds = []
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        valid = folds == fold
        thresholds = _thresholds_from_training(probabilities[train], targets[train], kappa=kappa)
        predictions[valid] = (probabilities[valid] >= thresholds).astype(np.int8)
        fold_thresholds.append(thresholds)
    final_thresholds = _thresholds_from_training(probabilities, targets, kappa=kappa)
    return predictions, final_thresholds


def factor_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0
    )
    groups = np.where(support < 60, "tail", np.where(support < 200, "mid", "head"))
    def group_mean(name: str) -> float:
        values = f1[groups == name]
        return float(values.mean()) if len(values) else float("nan")

    metrics = {
        "macro_f1": float(f1.mean()),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "macro_ap": safe_macro_ap(targets, probabilities),
        "tail_macro_f1": group_mean("tail"),
        "mid_macro_f1": group_mean("mid"),
        "head_macro_f1": group_mean("head"),
        "mean_predicted_labels": float(predictions.sum(axis=1).mean()),
    }
    table = pd.DataFrame(
        {
            "label": FACTOR_LABELS,
            "group": groups,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    return metrics, table


def count_diagnostics(
    pred_counts: np.ndarray,
    raw_counts: np.ndarray,
    targets: np.ndarray,
    clause_counts: np.ndarray,
) -> dict[str, float | None]:
    if np.isnan(pred_counts).all():
        return {"positive_count_spearman": None, "negative_length_spearman": None}
    pos = targets.astype(bool) & np.isfinite(pred_counts)
    neg = (~targets.astype(bool)) & np.isfinite(pred_counts)
    repeated_clause_counts = np.repeat(clause_counts[:, None], targets.shape[1], axis=1)

    def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
        if len(left) <= 2 or np.unique(left).size <= 1 or np.unique(right).size <= 1:
            return float("nan")
        return float(spearmanr(left, right).statistic)

    positive_rho = safe_spearman(pred_counts[pos], raw_counts[pos])
    negative_rho = safe_spearman(pred_counts[neg], repeated_clause_counts[neg])
    return {
        "positive_count_spearman": None if np.isnan(positive_rho) else float(positive_rho),
        "negative_length_spearman": None if np.isnan(negative_rho) else float(negative_rho),
    }


def run_factor_experiment(
    bundle: DataBundle,
    folds: np.ndarray,
    config: ExperimentConfig,
    artifact_root: str | Path,
) -> dict[str, Any]:
    artifact_root = Path(artifact_root)
    output_dir = artifact_root / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    json_dump(asdict(config), output_dir / "config.json")
    if config.use_count and not bundle.count_signal_available:
        raise RuntimeError("Count experiment requested, but the loaded data has no duplicate counts.")

    n = len(bundle.texts)
    logits = np.full((n, len(FACTOR_LABELS)), np.nan, dtype=np.float32)
    pred_counts = np.full_like(logits, np.nan)
    elapsed_by_fold: list[float] = []
    histories: dict[str, Any] = {}
    suite_start = time.perf_counter()
    actual_folds = [int(value) for value in sorted(np.unique(folds))]
    if len(actual_folds) != config.n_splits:
        warnings.warn(
            f"{config.name}: config.n_splits={config.n_splits}, but the supplied fold array "
            f"contains {len(actual_folds)} folds {actual_folds}. The fold array is authoritative."
        )

    for fold_position, fold in enumerate(actual_folds):
        fold_path = output_dir / f"fold_{fold}_oof.npz"
        history_path = output_dir / f"fold_{fold}_history.json"
        if config.resume and fold_path.exists():
            saved = np.load(fold_path)
            idx = saved["idx"]
            logits[idx] = saved["logits"]
            pred_counts[idx] = saved["pred_counts"]
            elapsed = float(saved["elapsed_seconds"])
            elapsed_by_fold.append(elapsed)
            print(f"[{config.name}] resumed fold {fold} ({elapsed / 60:.1f} min)")
            continue

        result = train_factor_fold(bundle, folds, fold, config, output_dir)
        idx = np.asarray(result["idx"], dtype=np.int64)
        logits[idx] = result["logits"]
        pred_counts[idx] = result["pred_counts"]
        elapsed = float(result["elapsed_seconds"])
        elapsed_by_fold.append(elapsed)
        histories[str(fold)] = result["history"]
        np.savez_compressed(
            fold_path,
            idx=idx,
            logits=result["logits"],
            pred_counts=result["pred_counts"],
            elapsed_seconds=np.asarray(elapsed),
        )
        json_dump(result["history"], history_path)
        median_minutes = np.median(elapsed_by_fold) / 60
        remaining = len(actual_folds) - fold_position - 1
        print(
            f"[{config.name}] fold {fold} completed in {elapsed / 60:.1f} min; "
            f"estimated {remaining * median_minutes:.1f} min remaining for this experiment."
        )

    if np.isnan(logits).any():
        raise RuntimeError(f"{config.name}: OOF logits contain missing values")
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    nested_predictions, final_thresholds = nested_threshold_predictions(
        probabilities, bundle.factor_binary, folds
    )
    metrics, per_label = factor_metrics(bundle.factor_binary, nested_predictions, probabilities)
    metrics.update(
        count_diagnostics(
            pred_counts,
            bundle.factor_count,
            bundle.factor_binary,
            bundle.clause_counts,
        )
    )
    metrics.update(
        {
            "experiment": config.name,
            "elapsed_minutes": (time.perf_counter() - suite_start) / 60,
            "recorded_fold_minutes": float(np.sum(elapsed_by_fold) / 60),
            "median_fold_minutes": float(np.median(elapsed_by_fold) / 60),
            "model_name": config.model_name,
            "max_length": config.max_length,
            "epochs": config.epochs,
        }
    )
    np.savez_compressed(
        output_dir / "oof_complete.npz",
        logits=logits,
        probabilities=probabilities,
        predictions=nested_predictions,
        thresholds=final_thresholds,
        pred_counts=pred_counts,
        folds=folds,
        row_ids=bundle.row_ids,
    )
    per_label.to_csv(output_dir / "per_label_metrics.csv", index=False)
    json_dump(metrics, output_dir / "summary.json")
    print(f"[{config.name}] summary: {metrics}")
    return metrics


@dataclass
class RiskOOFConfig:
    model_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 512
    epochs: int = 2
    train_batch_size: int = 8
    eval_batch_size: int = 16
    grad_accumulation: int = 2
    backbone_lr: float = 1.5e-5
    head_lr: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    seed: int = 42
    n_splits: int = 5
    num_workers: int = 2
    resume: bool = True


def evaluate_risk_loader(
    model: RiskModel, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    indices, logits = [], []
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(batch)
            indices.append(batch["idx"].cpu().numpy())
            logits.append(output.float().cpu().numpy())
    return np.concatenate(indices), np.concatenate(logits)


def train_risk_fold(
    bundle: DataBundle,
    folds: np.ndarray,
    fold: int,
    config: RiskOOFConfig,
) -> dict[str, Any]:
    seed_everything(config.seed + 1000 + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    collator = BatchCollator(tokenizer, config.max_length, use_clauses=False)
    train_idx = np.where(folds != fold)[0]
    val_idx = np.where(folds == fold)[0]
    train_loader = DataLoader(
        TextDataset(bundle, train_idx),
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )
    val_loader = DataLoader(
        TextDataset(bundle, val_idx),
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )
    model = RiskModel(config.model_name).to(device)
    exp_config = ExperimentConfig(name="risk_proxy")
    exp_config.backbone_lr = config.backbone_lr
    exp_config.head_lr = config.head_lr
    exp_config.weight_decay = config.weight_decay
    optimizer = make_optimizer(model, exp_config)
    updates_per_epoch = math.ceil(len(train_loader) / config.grad_accumulation)
    total_updates = max(1, updates_per_epoch * config.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * config.warmup_ratio),
        num_training_steps=total_updates,
    )
    train_counts = np.bincount(bundle.risk_ids[train_idx], minlength=len(RISK_LABELS)).astype(np.float32)
    class_weights = np.sqrt(train_counts.sum() / np.maximum(train_counts, 1.0))
    class_weights /= class_weights.mean()
    class_weights_tensor = torch.tensor(class_weights, device=device)
    best_score = -np.inf
    temp_dir = Path(tempfile.mkdtemp(prefix=f"b1_risk_fold{fold}_"))
    best_path = temp_dir / "best.pt"
    start = time.perf_counter()

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for step, raw_batch in enumerate(train_loader):
            batch = move_batch(raw_batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(batch)
                loss = F.cross_entropy(output, batch["risk_y"], weight=class_weights_tensor)
                scaled_loss = loss / config.grad_accumulation
            scaled_loss.backward()
            running += float(loss.detach().cpu())
            should_step = (step + 1) % config.grad_accumulation == 0 or step + 1 == len(train_loader)
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        val_positions, val_logits = evaluate_risk_loader(model, val_loader, device)
        val_pred = val_logits.argmax(axis=1)
        score = f1_score(bundle.risk_ids[val_positions], val_pred, average="weighted", zero_division=0)
        print(
            f"[RISK_PROXY] fold={fold} epoch={epoch + 1} "
            f"loss={running / max(1, len(train_loader)):.4f} wf1={score:.4f}"
        )
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    val_positions, val_logits = evaluate_risk_loader(model, val_loader, device)
    result = {
        "idx": val_positions,
        "logits": val_logits,
        "elapsed_seconds": time.perf_counter() - start,
    }
    del model, optimizer, scheduler, train_loader, val_loader
    shutil.rmtree(temp_dir, ignore_errors=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_risk_oof(
    bundle: DataBundle,
    folds: np.ndarray,
    config: RiskOOFConfig,
    artifact_root: str | Path,
) -> dict[str, Any]:
    output_dir = Path(artifact_root) / "RISK_PROXY"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_dump(asdict(config), output_dir / "config.json")
    logits = np.full((len(bundle.texts), len(RISK_LABELS)), np.nan, dtype=np.float32)
    fold_times = []
    for fold in range(config.n_splits):
        path = output_dir / f"fold_{fold}_oof.npz"
        if config.resume and path.exists():
            saved = np.load(path)
            logits[saved["idx"]] = saved["logits"]
            fold_times.append(float(saved["elapsed_seconds"]))
            print(f"[RISK_PROXY] resumed fold {fold}")
            continue
        result = train_risk_fold(bundle, folds, fold, config)
        logits[result["idx"]] = result["logits"]
        fold_times.append(float(result["elapsed_seconds"]))
        np.savez_compressed(
            path,
            idx=result["idx"],
            logits=result["logits"],
            elapsed_seconds=np.asarray(result["elapsed_seconds"]),
        )
    if np.isnan(logits).any():
        raise RuntimeError("Risk OOF logits contain missing values")
    predictions = logits.argmax(axis=1)
    weighted_f1 = float(f1_score(bundle.risk_ids, predictions, average="weighted", zero_division=0))
    result = {
        "risk_weighted_f1": weighted_f1,
        "median_fold_minutes": float(np.median(fold_times) / 60),
        "elapsed_minutes": float(np.sum(fold_times) / 60),
    }
    np.savez_compressed(
        output_dir / "oof_complete.npz",
        logits=logits,
        predictions=predictions,
        folds=folds,
        row_ids=bundle.row_ids,
    )
    json_dump(result, output_dir / "summary.json")
    print(f"[RISK_PROXY] summary: {result}")
    return result


def load_risk_oof(
    artifact_root: str | Path,
    bundle: DataBundle,
    external_path: str | Path | None = None,
) -> np.ndarray:
    candidates = []
    if external_path:
        candidates.append(Path(external_path))
    root = Path(artifact_root)
    candidates.extend(
        [
            root / "RISK_PROXY" / "oof_complete.npz",
            root / "risk_oof.npz",
            root / "task1a_oof.npz",
            root / "risk_oof.csv",
            root / "task1a_oof.csv",
        ]
    )
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".npz":
            saved = np.load(path, allow_pickle=True)
            logits = saved["logits"]
            if logits.shape != (len(bundle.texts), len(RISK_LABELS)):
                raise ValueError(f"Unexpected risk logits shape in {path}: {logits.shape}")
            return logits.astype(np.float32)
        frame = pd.read_csv(path)
        row_col = _find_column(frame, ["row_id", "id"])
        logit_columns = []
        for label in RISK_LABELS:
            possibilities = [
                f"logit_{label.lower()}",
                f"risk_logit_{label.lower()}",
                label,
            ]
            logit_columns.append(_find_column(frame, possibilities))
        lookup = frame.set_index(frame[row_col].astype(str))[logit_columns]
        return lookup.loc[bundle.row_ids].to_numpy(dtype=np.float32)
    raise FileNotFoundError("No risk OOF file found. Run run_risk_oof or provide external_path.")


def build_graph_prior(
    risk_targets: np.ndarray,
    factor_targets: np.ndarray,
    top_factor_edges: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    risk_onehot = np.eye(len(RISK_LABELS), dtype=np.float32)[risk_targets]
    labels = np.concatenate([risk_onehot, factor_targets.astype(np.float32)], axis=1)
    num_nodes = labels.shape[1]
    prior = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            a = labels[:, i] > 0
            b = labels[:, j] > 0
            n11 = float((a & b).sum()) + 0.5
            n10 = float((a & ~b).sum()) + 0.5
            n01 = float((~a & b).sum()) + 0.5
            n00 = float((~a & ~b).sum()) + 0.5
            log_odds = math.log((n11 * n00) / (n10 * n01))
            support = min(float(a.sum()), float(b.sum()))
            shrink = support / (support + 20.0)
            prior[i, j] = math.tanh(0.5 * log_odds) * shrink

    mask = np.eye(num_nodes, dtype=np.float32)
    # Adjacent ordinal risk nodes.
    for i in range(len(RISK_LABELS) - 1):
        mask[i, i + 1] = mask[i + 1, i] = 1.0
    # Risk-factor edges are the most interpretable block and are all allowed.
    mask[:4, 4:] = 1.0
    mask[4:, :4] = 1.0
    # Keep only a tiny number of factor-factor edges per target node.
    for i in range(4, num_nodes):
        candidates = np.arange(4, num_nodes)
        candidates = candidates[candidates != i]
        chosen = candidates[np.argsort(np.abs(prior[i, candidates]))[-top_factor_edges:]]
        mask[i, chosen] = 1.0
        mask[chosen, i] = 1.0
    return prior, mask


class LowRankLabelGraph(nn.Module):
    def __init__(self, edge_mask: np.ndarray, rank: int = 4):
        super().__init__()
        num_nodes = edge_mask.shape[0]
        self.u = nn.Parameter(torch.randn(num_nodes, rank) * 0.02)
        self.v = nn.Parameter(torch.randn(num_nodes, rank) * 0.02)
        self.gate = nn.Parameter(torch.tensor(-4.0))
        self.register_buffer("edge_mask", torch.tensor(edge_mask, dtype=torch.float32))

    def forward(self, raw_logits: torch.Tensor, normalized_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weight = (self.u @ self.v.T) * self.edge_mask
        delta = torch.tanh(normalized_logits) @ weight.T
        adjusted = raw_logits + torch.sigmoid(self.gate) * delta
        return adjusted, delta


class PriorGAT(nn.Module):
    def __init__(self, edge_prior: np.ndarray, edge_mask: np.ndarray, hidden: int = 32, heads: int = 2):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads")
        num_nodes = edge_mask.shape[0]
        self.num_nodes = num_nodes
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.label_embedding = nn.Parameter(torch.randn(num_nodes, 16) * 0.02)
        self.input_projection = nn.Linear(18, hidden)
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.output = nn.Linear(hidden, 1)
        self.edge_message = nn.Parameter(torch.randn(heads, self.head_dim) * 0.02)
        self.prior_scale = nn.Parameter(torch.tensor(0.25))
        self.gate = nn.Parameter(torch.tensor(-4.0))
        self.dropout = nn.Dropout(0.20)
        self.register_buffer("edge_prior", torch.tensor(edge_prior, dtype=torch.float32))
        self.register_buffer("edge_mask", torch.tensor(edge_mask, dtype=torch.bool))

    def forward(self, raw_logits: torch.Tensor, normalized_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = raw_logits.shape[0]
        label_emb = self.label_embedding[None, :, :].expand(batch, -1, -1)
        node_input = torch.cat(
            [normalized_logits.unsqueeze(-1), torch.sigmoid(raw_logits).unsqueeze(-1), label_emb],
            dim=-1,
        )
        hidden = torch.tanh(self.input_projection(node_input))
        q = self.q(hidden).view(batch, self.num_nodes, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(hidden).view(batch, self.num_nodes, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(hidden).view(batch, self.num_nodes, self.heads, self.head_dim).transpose(1, 2)
        attention = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.head_dim)
        attention = attention + self.prior_scale * self.edge_prior.abs()[None, None, :, :]
        attention = attention.masked_fill(~self.edge_mask[None, None, :, :], -1e4)
        attention = self.dropout(torch.softmax(attention, dim=-1))
        message = torch.einsum("bhij,bhjd->bhid", attention, v)
        signed_edge = self.edge_prior[None, None, :, :, None] * self.edge_message[None, :, None, None, :]
        signed_message = (attention[..., None] * signed_edge).sum(dim=3)
        message = message + signed_message
        message = message.transpose(1, 2).reshape(batch, self.num_nodes, self.hidden)
        delta = self.output(torch.tanh(message)).squeeze(-1)
        adjusted = raw_logits + torch.sigmoid(self.gate) * delta
        return adjusted, delta


@dataclass
class GraphConfig:
    name: str
    kind: str = "lowrank"
    epochs: int = 120
    batch_size: int = 128
    learning_rate: float = 3e-3
    weight_decay: float = 1e-3
    delta_l2: float = 1e-3
    seed: int = 42
    rank: int = 4
    hidden: int = 32
    heads: int = 2


def train_graph_outer_fold(
    base_logits: np.ndarray,
    risk_targets: np.ndarray,
    factor_targets: np.ndarray,
    folds: np.ndarray,
    outer_fold: int,
    config: GraphConfig,
) -> tuple[np.ndarray, np.ndarray]:
    seed_everything(config.seed + 2000 + outer_fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_idx = np.where(folds != outer_fold)[0]
    valid_idx = np.where(folds == outer_fold)[0]
    prior, mask = build_graph_prior(risk_targets[train_idx], factor_targets[train_idx])
    mean = base_logits[train_idx].mean(axis=0, keepdims=True)
    std = base_logits[train_idx].std(axis=0, keepdims=True).clip(min=0.10)
    if config.kind == "lowrank":
        model: nn.Module = LowRankLabelGraph(mask, rank=config.rank)
    elif config.kind == "gat":
        model = PriorGAT(prior, mask, hidden=config.hidden, heads=config.heads)
    else:
        raise ValueError(f"Unknown graph kind: {config.kind}")
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    asl = AsymmetricLoss()
    x = torch.tensor(base_logits[train_idx], dtype=torch.float32)
    x_norm = torch.tensor((base_logits[train_idx] - mean) / std, dtype=torch.float32)
    risk_y = torch.tensor(risk_targets[train_idx], dtype=torch.long)
    factor_y = torch.tensor(factor_targets[train_idx], dtype=torch.float32)

    for epoch in range(config.epochs):
        order = torch.randperm(len(train_idx))
        model.train()
        for left in range(0, len(train_idx), config.batch_size):
            positions = order[left : left + config.batch_size]
            raw_batch = x[positions].to(device)
            norm_batch = x_norm[positions].to(device)
            risk_batch = risk_y[positions].to(device)
            factor_batch = factor_y[positions].to(device)
            adjusted, delta = model(raw_batch, norm_batch)
            risk_loss = F.cross_entropy(adjusted[:, :4], risk_batch)
            factor_loss = asl(adjusted[:, 4:], factor_batch)
            loss = risk_loss + factor_loss + config.delta_l2 * delta.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        raw_valid = torch.tensor(base_logits[valid_idx], dtype=torch.float32, device=device)
        norm_valid = torch.tensor((base_logits[valid_idx] - mean) / std, dtype=torch.float32, device=device)
        adjusted, _ = model(raw_valid, norm_valid)
    adjusted_np = adjusted.cpu().numpy()
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return valid_idx, adjusted_np


def graph_metrics(
    adjusted_logits: np.ndarray,
    risk_targets: np.ndarray,
    factor_targets: np.ndarray,
    folds: np.ndarray,
) -> dict[str, float]:
    risk_pred = adjusted_logits[:, :4].argmax(axis=1)
    factor_probs = 1.0 / (1.0 + np.exp(-np.clip(adjusted_logits[:, 4:], -30, 30)))
    factor_pred, _ = nested_threshold_predictions(factor_probs, factor_targets, folds)
    risk_wf1 = float(f1_score(risk_targets, risk_pred, average="weighted", zero_division=0))
    factor_macro = float(f1_score(factor_targets, factor_pred, average="macro", zero_division=0))
    return {
        "risk_weighted_f1": risk_wf1,
        "factor_macro_f1": factor_macro,
        "partial_composite": 0.4 * risk_wf1 + 0.3 * factor_macro,
    }


def run_graph_experiment(
    factor_experiment_name: str,
    risk_logits: np.ndarray,
    bundle: DataBundle,
    folds: np.ndarray,
    config: GraphConfig,
    artifact_root: str | Path,
) -> dict[str, Any]:
    artifact_root = Path(artifact_root)
    factor_oof_path = artifact_root / factor_experiment_name / "oof_complete.npz"
    if not factor_oof_path.exists():
        raise FileNotFoundError(factor_oof_path)
    factor_saved = np.load(factor_oof_path)
    factor_logits = factor_saved["logits"].astype(np.float32)
    base_logits = np.concatenate([risk_logits.astype(np.float32), factor_logits], axis=1)
    output_dir = artifact_root / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    json_dump(asdict(config), output_dir / "config.json")
    adjusted = np.full_like(base_logits, np.nan)
    start = time.perf_counter()
    for outer_fold in sorted(np.unique(folds)):
        valid_idx, valid_logits = train_graph_outer_fold(
            base_logits,
            bundle.risk_ids,
            bundle.factor_binary,
            folds,
            int(outer_fold),
            config,
        )
        adjusted[valid_idx] = valid_logits
    if np.isnan(adjusted).any():
        raise RuntimeError(f"{config.name}: adjusted logits contain missing values")
    base = graph_metrics(base_logits, bundle.risk_ids, bundle.factor_binary, folds)
    graph = graph_metrics(adjusted, bundle.risk_ids, bundle.factor_binary, folds)
    summary: dict[str, Any] = {
        "experiment": config.name,
        "factor_source": factor_experiment_name,
        "elapsed_minutes": (time.perf_counter() - start) / 60,
        **{f"base_{key}": value for key, value in base.items()},
        **{f"graph_{key}": value for key, value in graph.items()},
        "delta_partial_composite": graph["partial_composite"] - base["partial_composite"],
        "delta_factor_macro_f1": graph["factor_macro_f1"] - base["factor_macro_f1"],
        "delta_risk_weighted_f1": graph["risk_weighted_f1"] - base["risk_weighted_f1"],
    }
    np.savez_compressed(
        output_dir / "oof_complete.npz",
        base_logits=base_logits,
        adjusted_logits=adjusted,
        folds=folds,
        row_ids=bundle.row_ids,
    )
    json_dump(summary, output_dir / "summary.json")
    print(f"[{config.name}] summary: {summary}")
    return summary


def proxy_experiment_configs(
    model_name: str = "answerdotai/ModernBERT-base",
    seed: int = 42,
    epochs: int = 2,
    max_length: int = 512,
    n_splits: int = 5,
) -> list[ExperimentConfig]:
    common = dict(
        model_name=model_name,
        seed=seed,
        epochs=epochs,
        max_length=max_length,
        n_splits=n_splits,
    )
    return [
        ExperimentConfig(name="F0_BCE_DOC", loss_name="bce", **common),
        ExperimentConfig(name="F1_ASL_DOC", loss_name="asl", **common),
        ExperimentConfig(name="F2_ASL_MIL", loss_name="asl", use_mil=True, **common),
        ExperimentConfig(
            name="F3_ASL_MIL_MSC",
            loss_name="asl",
            use_mil=True,
            use_contrastive=True,
            **common,
        ),
        ExperimentConfig(
            name="F4_ASL_MIL_COUNT",
            loss_name="asl",
            use_mil=True,
            use_count=True,
            **common,
        ),
        ExperimentConfig(
            name="F5_ASL_MIL_MSC_COUNT",
            loss_name="asl",
            use_mil=True,
            use_contrastive=True,
            use_count=True,
            **common,
        ),
    ]


def select_confirmation_configs(
    summaries: list[dict[str, Any]],
    full_model_name: str = "answerdotai/ModernBERT-large",
    seed: int = 42,
    epochs: int = 3,
    max_length: int = 1024,
    baseline_name: str = "F1_ASL_DOC",
    top_k: int = 2,
    n_splits: int = 5,
) -> list[ExperimentConfig]:
    by_name = {item["experiment"]: item for item in summaries}
    candidates = [
        item
        for item in summaries
        if item["experiment"] in {
            "F2_ASL_MIL",
            "F3_ASL_MIL_MSC",
            "F4_ASL_MIL_COUNT",
            "F5_ASL_MIL_MSC_COUNT",
        }
        and item["experiment"] != baseline_name
    ]
    candidates.sort(key=lambda item: (item["macro_f1"], item["tail_macro_f1"]), reverse=True)
    selected_names = [baseline_name] + [item["experiment"] for item in candidates[:top_k]]
    proxy_map = {config.name: config for config in proxy_experiment_configs()}
    results = []
    for name in selected_names:
        source = proxy_map[name]
        results.append(
            dataclasses.replace(
                source,
                name=f"FULL_{name}",
                model_name=full_model_name,
                seed=seed,
                epochs=epochs,
                max_length=max_length,
                n_splits=n_splits,
                train_batch_size=3,
                eval_batch_size=6,
                grad_accumulation=6,
                backbone_lr=1.0e-5,
                # Confirmation is an ablation stage. Saving 15 large-model
                # checkpoints can exhaust the Drive shown in the project setup.
                # Retrain only the selected winner with checkpoints afterwards.
                save_checkpoints=False,
            )
        )
    if baseline_name not in by_name:
        warnings.warn("F1_ASL_DOC summary was not present; it is still included as the full baseline.")
    return results


def summarize_experiments(artifact_root: str | Path) -> pd.DataFrame:
    records = []
    for path in Path(artifact_root).glob("*/summary.json"):
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        if "macro_f1" in record:
            records.append(record)
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    return frame.sort_values(["macro_f1", "tail_macro_f1"], ascending=False).reset_index(drop=True)


def recommend_b1(summary_frame: pd.DataFrame) -> dict[str, Any]:
    if summary_frame.empty:
        return {"status": "no experiments found"}
    names = set(summary_frame["experiment"])
    if "FULL_F2_ASL_MIL" in names:
        baseline_name = "FULL_F2_ASL_MIL"
    elif "FULL_F1_ASL_DOC" in names:
        baseline_name = "FULL_F1_ASL_DOC"
    elif "F2_ASL_MIL" in names:
        baseline_name = "F2_ASL_MIL"
    else:
        baseline_name = "F1_ASL_DOC"
    baseline_row = summary_frame[summary_frame["experiment"] == baseline_name]
    if baseline_row.empty:
        baseline_row = summary_frame.iloc[[0]]
        baseline_name = str(baseline_row.iloc[0]["experiment"])
    baseline = baseline_row.iloc[0]
    candidates = summary_frame[summary_frame["experiment"].str.startswith("FULL_")]
    if candidates.empty:
        candidates = summary_frame
    winner = candidates.sort_values(["macro_f1", "tail_macro_f1"], ascending=False).iloc[0]
    delta_macro = float(winner["macro_f1"] - baseline["macro_f1"])
    delta_tail = float(winner["tail_macro_f1"] - baseline["tail_macro_f1"])
    accepted = bool(delta_macro >= 0.003 and delta_tail >= 0.0)
    return {
        "baseline": baseline_name,
        "winner": str(winner["experiment"]),
        "delta_macro_f1": delta_macro,
        "delta_tail_macro_f1": delta_tail,
        "accept_winner_over_baseline": accepted,
        "recommended_factor_model": str(winner["experiment"] if accepted else baseline_name),
        "note": "Graph modules are decided separately by nested partial-composite delta.",
    }


def timing_projection(
    completed_summaries: pd.DataFrame,
    remaining_proxy_experiments: int = 0,
    remaining_full_experiments: int = 0,
    folds: int = 5,
) -> dict[str, float | str]:
    if completed_summaries.empty or "median_fold_minutes" not in completed_summaries:
        return {
            "status": "Run at least one fold to obtain a hardware-specific estimate.",
            "estimated_remaining_hours": float("nan"),
        }
    proxy_rows = completed_summaries[~completed_summaries["experiment"].str.startswith("FULL_")]
    full_rows = completed_summaries[completed_summaries["experiment"].str.startswith("FULL_")]
    proxy_minutes = float(proxy_rows["median_fold_minutes"].median()) if not proxy_rows.empty else 10.0
    full_minutes = float(full_rows["median_fold_minutes"].median()) if not full_rows.empty else proxy_minutes * 3.0
    remaining = folds * (
        remaining_proxy_experiments * proxy_minutes + remaining_full_experiments * full_minutes
    )
    return {
        "proxy_median_fold_minutes": proxy_minutes,
        "full_median_fold_minutes": full_minutes,
        "estimated_remaining_hours": remaining / 60.0,
        "status": "Estimate uses observed median fold duration and excludes queue/setup interruptions.",
    }


def environment_report() -> dict[str, Any]:
    report = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "device_memory_gb": (
            torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if torch.cuda.is_available()
            else None
        ),
        "factor_labels": len(FACTOR_LABELS),
        "risk_labels": RISK_LABELS,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report
