"""Stage-gated experiments for the two CARE-MA B1 innovations.

The module contains two independent, leak-safe three-fold harnesses:

* Label-conditioned Top-K Evidence Bottleneck (LCEB) for the 24 factors.
* Evidence-conditioned ordinal risk modelling with token/span supervision.

It deliberately reuses the data/fold/metric contracts in ``b1_experiments`` so
that architecture deltas are comparable with the earlier B1 and B1-R runs.
"""

from __future__ import annotations

import difflib
import gc
import json
import math
import re
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
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

import b1_experiments as b1
from b1r_experiments import FACTOR_CARDS, N_LABELS, sigmoid


def _json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)


# ---------------------------------------------------------------------------
# Innovation A: Label-conditioned Top-K Evidence Bottleneck
# ---------------------------------------------------------------------------


@dataclass
class FactorBottleneckConfig:
    name: str
    model_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 512
    epochs: int = 1
    train_batch_size: int = 8
    eval_batch_size: int = 16
    grad_accumulation: int = 2
    backbone_lr: float = 1.5e-5
    head_lr: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    seed: int = 42
    n_splits: int = 3
    projection_dim: int = 256
    top_k: int = 3
    support_temperature: float = 0.25
    doc_aux_weight: float = 0.20
    lambda_sufficiency: float = 0.0
    lambda_comprehensiveness: float = 0.0
    lambda_compactness: float = 0.0
    lambda_semantic_drift: float = 0.002
    comprehensiveness_margin: float = 0.35
    semantic_residual_scale: float = 0.10
    max_grad_norm: float = 1.0
    num_workers: int = 2
    save_checkpoints: bool = False
    resume: bool = True


class LabelConditionedEvidenceBottleneck(nn.Module):
    """A semantic label anchor attends to clauses through a hard Top-K gate."""

    def __init__(self, config: FactorBottleneckConfig):
        super().__init__()
        self.config = config
        self.backbone = b1.load_backbone(config.model_name)
        hidden = int(self.backbone.config.hidden_size)
        dim = config.projection_dim
        self.dropout = nn.Dropout(0.10)
        self.clause_projection = nn.Sequential(
            nn.Linear(hidden, dim), nn.GELU(), nn.LayerNorm(dim)
        )
        self.label_projection = nn.Sequential(
            nn.Linear(hidden, dim), nn.GELU(), nn.LayerNorm(dim)
        )
        self.label_semantic_residual = nn.Parameter(torch.zeros(N_LABELS, dim))
        self.label_bias = nn.Parameter(torch.zeros(N_LABELS))
        self.pair_head = nn.Sequential(
            nn.Linear(4 * dim, dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(dim, 1),
        )
        self.doc_head = nn.Linear(hidden, N_LABELS)
        self.register_buffer("label_semantics", torch.zeros(N_LABELS, hidden))

    @torch.no_grad()
    def initialize_label_semantics(self, tokenizer: Any, device: torch.device) -> None:
        texts = [
            f"Psychosocial factor: {label}. Definition: {FACTOR_CARDS[label]}"
            for label in b1.FACTOR_LABELS
        ]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=96,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        was_training = self.backbone.training
        self.backbone.eval()
        output = self.backbone(**encoded).last_hidden_state
        semantics = b1.masked_mean(output, encoded["attention_mask"])
        self.label_semantics.copy_(semantics.to(self.label_semantics.dtype))
        self.backbone.train(was_training)

    def _pair_logits(
        self,
        clause_vectors: torch.Tensor,
        query: torch.Tensor,
        support_scores: torch.Tensor,
        clause_valid: torch.Tensor,
        *,
        remove_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = support_scores
        if remove_mask is not None:
            scores = scores.masked_fill(remove_mask, -1e4)
        scores = scores.masked_fill(~clause_valid[:, None, :], -1e4)
        k = min(self.config.top_k, scores.shape[-1])
        top_values, top_indices = torch.topk(scores, k=k, dim=-1)
        top_valid = top_values > -5e3
        weights = torch.softmax(top_values.masked_fill(~top_valid, -1e4), dim=-1)
        weights = weights * top_valid.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        batch, labels, _, dim = (
            clause_vectors.shape[0],
            query.shape[0],
            clause_vectors.shape[1],
            clause_vectors.shape[2],
        )
        expanded = clause_vectors[:, None, :, :].expand(batch, labels, -1, -1)
        selected = torch.gather(
            expanded,
            2,
            top_indices[..., None].expand(batch, labels, k, dim),
        )
        evidence = torch.sum(selected * weights[..., None], dim=2)
        q = query[None, :, :].expand(batch, -1, -1)
        features = torch.cat([evidence, q, evidence * q, torch.abs(evidence - q)], dim=-1)
        logits = self.pair_head(features).squeeze(-1) + self.label_bias[None, :]
        selected_mask = torch.zeros_like(scores, dtype=torch.bool)
        selected_mask.scatter_(2, top_indices, top_valid)
        return logits.float(), selected_mask, weights.float(), top_indices

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        inputs = {
            key: batch[key]
            for key in ("input_ids", "attention_mask", "token_type_ids")
            if key in batch
        }
        hidden = self.backbone(**inputs).last_hidden_state
        document = self.dropout(b1.masked_mean(hidden, batch["attention_mask"]))
        clause_mask = batch["clause_token_mask"].to(hidden.dtype)
        denominator = clause_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        clause_hidden = torch.einsum("bst,bth->bsh", clause_mask, hidden) / denominator
        clause_vectors = F.normalize(self.clause_projection(clause_hidden).float(), dim=-1)
        query = self.label_projection(self.label_semantics.float())
        query = query + self.config.semantic_residual_scale * self.label_semantic_residual
        query = F.normalize(query, dim=-1)
        support_scores = torch.einsum("bsd,ld->bls", clause_vectors, query)
        support_scores = support_scores / max(1e-4, self.config.support_temperature)
        clause_valid = clause_mask.sum(dim=-1) > 0

        logits, selected_mask, weights, top_indices = self._pair_logits(
            clause_vectors, query, support_scores, clause_valid
        )
        removed_logits, _, _, _ = self._pair_logits(
            clause_vectors,
            query,
            support_scores,
            clause_valid,
            remove_mask=selected_mask,
        )
        return {
            "logits": logits,
            "doc_logits": self.doc_head(document).float(),
            "removed_logits": removed_logits,
            "attention_weights": weights,
            "selected_indices": top_indices,
            "has_remaining_clause": clause_valid.sum(dim=-1) > self.config.top_k,
            "semantic_drift": self.label_semantic_residual.square().mean(),
        }


def factor_bottleneck_loss(
    output: dict[str, torch.Tensor],
    targets: torch.Tensor,
    config: FactorBottleneckConfig,
    asl: b1.AsymmetricLoss,
    constraint_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    classification = asl(output["logits"], targets)
    doc_aux = asl(output["doc_logits"], targets)
    sufficiency = F.mse_loss(
        torch.sigmoid(output["logits"]),
        torch.sigmoid(output["doc_logits"]).detach(),
    )
    positive = targets > 0.5
    valid_comp = positive & output["has_remaining_clause"][:, None]
    comp_values = F.relu(
        config.comprehensiveness_margin
        - output["logits"]
        + output["removed_logits"]
    )
    comprehensiveness = (
        comp_values[valid_comp].mean()
        if valid_comp.any()
        else output["logits"].new_zeros(())
    )
    weights = output["attention_weights"].clamp_min(1e-8)
    entropy = -(weights * weights.log()).sum(dim=-1)
    compactness = entropy.mean() / max(1e-6, math.log(max(2, config.top_k)))
    loss = classification + config.doc_aux_weight * doc_aux
    loss = loss + constraint_scale * config.lambda_sufficiency * sufficiency
    loss = loss + constraint_scale * config.lambda_comprehensiveness * comprehensiveness
    loss = loss + constraint_scale * config.lambda_compactness * compactness
    loss = loss + config.lambda_semantic_drift * output["semantic_drift"]
    pieces = {
        "classification": float(classification.detach().cpu()),
        "doc_aux": float(doc_aux.detach().cpu()),
        "sufficiency": float(sufficiency.detach().cpu()),
        "comprehensiveness": float(comprehensiveness.detach().cpu()),
        "compactness": float(compactness.detach().cpu()),
    }
    return loss, pieces


@torch.no_grad()
def evaluate_factor_bottleneck(
    model: LabelConditionedEvidenceBottleneck,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    indices: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    doc_logits: list[np.ndarray] = []
    removed_logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    selected: list[np.ndarray] = []
    for raw_batch in loader:
        batch = b1.move_batch(raw_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(batch)
        indices.append(batch["idx"].cpu().numpy())
        logits.append(output["logits"].cpu().numpy())
        doc_logits.append(output["doc_logits"].cpu().numpy())
        removed_logits.append(output["removed_logits"].cpu().numpy())
        targets.append(batch["factor_y"].cpu().numpy())
        batch_selected = output["selected_indices"].cpu().numpy()
        if batch_selected.shape[-1] < model.config.top_k:
            pad_width = model.config.top_k - batch_selected.shape[-1]
            batch_selected = np.pad(
                batch_selected,
                ((0, 0), (0, 0), (0, pad_width)),
                mode="constant",
                constant_values=-1,
            )
        selected.append(batch_selected)
    all_logits = np.concatenate(logits)
    all_targets = np.concatenate(targets).astype(np.int8)
    probabilities = sigmoid(all_logits)
    return {
        "idx": np.concatenate(indices),
        "logits": all_logits,
        "doc_logits": np.concatenate(doc_logits),
        "removed_logits": np.concatenate(removed_logits),
        "targets": all_targets,
        "selected_indices": np.concatenate(selected),
        "macro_ap": b1.safe_macro_ap(all_targets, probabilities),
        "macro_f1_at_05": float(
            f1_score(all_targets, probabilities >= 0.5, average="macro", zero_division=0)
        ),
    }


def _factor_optimizer(
    model: LabelConditionedEvidenceBottleneck,
    config: FactorBottleneckConfig,
) -> torch.optim.Optimizer:
    head_parameters = []
    backbone_parameters = []
    for name, parameter in model.named_parameters():
        (backbone_parameters if name.startswith("backbone.") else head_parameters).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": config.backbone_lr},
            {"params": head_parameters, "lr": config.head_lr},
        ],
        weight_decay=config.weight_decay,
    )


def train_factor_bottleneck_fold(
    bundle: b1.DataBundle,
    folds: np.ndarray,
    fold: int,
    config: FactorBottleneckConfig,
    output_dir: Path,
) -> dict[str, Any]:
    b1.seed_everything(config.seed + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    collator = b1.BatchCollator(tokenizer, config.max_length, use_clauses=True)
    train_indices = np.where(folds != fold)[0]
    validation_indices = np.where(folds == fold)[0]
    train_loader = DataLoader(
        b1.TextDataset(bundle, train_indices),
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        b1.TextDataset(bundle, validation_indices),
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )
    model = LabelConditionedEvidenceBottleneck(config).to(device)
    model.initialize_label_semantics(tokenizer, device)
    optimizer = _factor_optimizer(model, config)
    updates_per_epoch = math.ceil(len(train_loader) / config.grad_accumulation)
    total_updates = max(1, updates_per_epoch * config.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * config.warmup_ratio),
        num_training_steps=total_updates,
    )
    asl = b1.AsymmetricLoss()
    best_metric = -np.inf
    history: list[dict[str, Any]] = []
    temp_dir = Path(tempfile.mkdtemp(prefix=f"lceb_{fold}_"))
    best_path = temp_dir / "best.pt"
    started = time.perf_counter()

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running: dict[str, float] = {
            "loss": 0.0,
            "classification": 0.0,
            "doc_aux": 0.0,
            "sufficiency": 0.0,
            "comprehensiveness": 0.0,
            "compactness": 0.0,
        }
        for step, raw_batch in enumerate(train_loader):
            batch = b1.move_batch(raw_batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(batch)
                global_step = epoch * len(train_loader) + step + 1
                constraint_scale = min(1.0, global_step / max(1.0, 0.25 * len(train_loader) * config.epochs))
                loss, pieces = factor_bottleneck_loss(
                    output,
                    batch["factor_y"],
                    config,
                    asl,
                    constraint_scale=constraint_scale,
                )
                scaled = loss / config.grad_accumulation
            scaled.backward()
            running["loss"] += float(loss.detach().cpu())
            for key, value in pieces.items():
                running[key] += value
            should_step = (step + 1) % config.grad_accumulation == 0 or step + 1 == len(train_loader)
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        validation = evaluate_factor_bottleneck(model, validation_loader, device)
        selection = 0.7 * validation["macro_ap"] + 0.3 * validation["macro_f1_at_05"]
        record = {
            "epoch": epoch + 1,
            **{key: value / max(1, len(train_loader)) for key, value in running.items()},
            "val_macro_ap": float(validation["macro_ap"]),
            "val_macro_f1_at_05": float(validation["macro_f1_at_05"]),
            "selection_metric": float(selection),
        }
        history.append(record)
        print(f"[{config.name}] fold={fold} {record}")
        if selection > best_metric:
            best_metric = selection
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    validation = evaluate_factor_bottleneck(model, validation_loader, device)
    elapsed = time.perf_counter() - started
    if config.save_checkpoints:
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / f"fold_{fold}.pt")
    result = {
        "idx": validation["idx"],
        "logits": validation["logits"],
        "doc_logits": validation["doc_logits"],
        "removed_logits": validation["removed_logits"],
        "selected_indices": validation["selected_indices"],
        "history": history,
        "elapsed_seconds": elapsed,
    }
    del model, optimizer, scheduler, train_loader, validation_loader
    shutil.rmtree(temp_dir, ignore_errors=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_factor_bottleneck_experiment(
    bundle: b1.DataBundle,
    folds: np.ndarray,
    config: FactorBottleneckConfig,
    artifact_root: str | Path,
) -> dict[str, Any]:
    output_dir = Path(artifact_root) / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(asdict(config), output_dir / "config.json")
    logits = np.full((len(bundle.texts), N_LABELS), np.nan, dtype=np.float32)
    doc_logits = np.full_like(logits, np.nan)
    removed_logits = np.full_like(logits, np.nan)
    selected = np.full(
        (len(bundle.texts), N_LABELS, config.top_k), -1, dtype=np.int16
    )
    elapsed_by_fold: list[float] = []
    started = time.perf_counter()
    for position, fold_value in enumerate(sorted(np.unique(folds))):
        fold = int(fold_value)
        fold_path = output_dir / f"fold_{fold}_oof.npz"
        if config.resume and fold_path.exists():
            saved = np.load(fold_path)
            idx = saved["idx"].astype(np.int64)
            logits[idx] = saved["logits"]
            if "doc_logits" not in saved or "removed_logits" not in saved:
                raise RuntimeError(
                    f"{fold_path} 来自旧版脚本，缺少 bottleneck 诊断数组。"
                    "请更换实验 name 后重新运行。"
                )
            doc_logits[idx] = saved["doc_logits"]
            removed_logits[idx] = saved["removed_logits"]
            selected[idx] = saved["selected_indices"]
            elapsed_by_fold.append(float(saved["elapsed_seconds"]))
            print(f"[{config.name}] resumed fold {fold}")
            continue
        result = train_factor_bottleneck_fold(bundle, folds, fold, config, output_dir)
        idx = np.asarray(result["idx"], dtype=np.int64)
        logits[idx] = result["logits"]
        doc_logits[idx] = result["doc_logits"]
        removed_logits[idx] = result["removed_logits"]
        selected[idx] = result["selected_indices"].astype(np.int16)
        elapsed = float(result["elapsed_seconds"])
        elapsed_by_fold.append(elapsed)
        np.savez_compressed(
            fold_path,
            idx=idx,
            logits=result["logits"],
            doc_logits=result["doc_logits"],
            removed_logits=result["removed_logits"],
            selected_indices=result["selected_indices"],
            elapsed_seconds=np.asarray(elapsed),
        )
        _json_dump(result["history"], output_dir / f"fold_{fold}_history.json")
        remaining = len(np.unique(folds)) - position - 1
        print(
            f"[{config.name}] fold {fold}: {elapsed / 60:.1f} min; "
            f"estimated {remaining * np.median(elapsed_by_fold) / 60:.1f} min remaining"
        )
    if np.isnan(logits).any() or np.isnan(doc_logits).any() or np.isnan(removed_logits).any():
        raise RuntimeError(f"{config.name}: incomplete OOF logits")
    probabilities = sigmoid(logits)
    predictions, thresholds = b1.nested_threshold_predictions(
        probabilities, bundle.factor_binary, folds, kappa=25.0
    )
    metrics, per_label = b1.factor_metrics(bundle.factor_binary, predictions, probabilities)
    sufficiency_gap = np.abs(probabilities - sigmoid(doc_logits))
    positive_mask = bundle.factor_binary > 0
    comprehensiveness_drop = probabilities - sigmoid(removed_logits)
    metrics.update(
        {
            "experiment": config.name,
            "elapsed_minutes": (time.perf_counter() - started) / 60.0,
            "recorded_fold_minutes": float(np.sum(elapsed_by_fold) / 60.0),
            "median_fold_minutes": float(np.median(elapsed_by_fold) / 60.0),
            "model_name": config.model_name,
            "max_length": config.max_length,
            "epochs": config.epochs,
            "sufficiency_probability_gap": float(sufficiency_gap.mean()),
            "positive_comprehensiveness_drop": float(
                comprehensiveness_drop[positive_mask].mean()
            ),
            "positive_comprehensiveness_success_rate": float(
                (comprehensiveness_drop[positive_mask] > 0).mean()
            ),
            "top_k": config.top_k,
            "lambda_sufficiency": config.lambda_sufficiency,
            "lambda_comprehensiveness": config.lambda_comprehensiveness,
            "lambda_compactness": config.lambda_compactness,
        }
    )
    np.savez_compressed(
        output_dir / "oof_complete.npz",
        logits=logits,
        doc_logits=doc_logits,
        removed_logits=removed_logits,
        probabilities=probabilities,
        predictions=predictions,
        thresholds=thresholds,
        folds=folds,
        row_ids=bundle.row_ids,
        selected_indices=selected,
    )
    per_label.to_csv(output_dir / "per_label_metrics.csv", index=False)
    _json_dump(metrics, output_dir / "summary.json")
    print(f"[{config.name}] summary: {metrics}")
    return metrics


# ---------------------------------------------------------------------------
# Innovation B: Evidence-conditioned ordinal risk prediction
# ---------------------------------------------------------------------------


_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True)
class EvidenceAnnotation:
    spans: tuple[tuple[int, int], ...]
    has_evidence: bool
    matched_phrases: int
    total_phrases: int
    exact_phrases: int
    fuzzy_phrases: int


def parse_evidence_phrases(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    phrases = [item.strip() for item in str(value).split(";") if item.strip()]
    return [item for item in phrases if item.lower() not in {"none", "nan", "n/a"}]


def _word_tokens_with_spans(text: str) -> list[tuple[str, int, int]]:
    return [
        (match.group(0).lower(), match.start(), match.end())
        for match in _WORD_RE.finditer(text)
    ]


def _fuzzy_phrase_span(text: str, phrase: str) -> tuple[int, int] | None:
    text_tokens = _word_tokens_with_spans(text)
    phrase_tokens = [token for token, _, _ in _word_tokens_with_spans(phrase)]
    if not text_tokens or not phrase_tokens:
        return None
    target_length = len(phrase_tokens)
    radius = max(2, int(math.ceil(target_length * 0.30)))
    best: tuple[float, int, int] | None = None
    text_words = [token for token, _, _ in text_tokens]
    for length in range(max(1, target_length - radius), target_length + radius + 1):
        for start in range(0, len(text_words) - length + 1):
            window = text_words[start : start + length]
            score = difflib.SequenceMatcher(None, phrase_tokens, window).ratio()
            if best is None or score > best[0]:
                best = (score, start, start + length)
    threshold = 0.72 if target_length <= 2 else 0.58
    if best is None or best[0] < threshold:
        return None
    _, start, end = best
    return text_tokens[start][1], text_tokens[end - 1][2]


def find_evidence_spans(text: str, evidence_value: Any) -> EvidenceAnnotation:
    text = str(text or "")
    phrases = parse_evidence_phrases(evidence_value)
    spans: list[tuple[int, int]] = []
    lower_text = text.lower()
    matched = 0
    exact = 0
    fuzzy = 0
    for phrase in phrases:
        start = lower_text.find(phrase.lower())
        if start >= 0:
            span = (start, start + len(phrase))
            exact += 1
        else:
            span = _fuzzy_phrase_span(text, phrase)
            if span is not None:
                fuzzy += 1
        if span is not None:
            spans.append(span)
            matched += 1
    spans = sorted(set(spans))
    return EvidenceAnnotation(
        spans=tuple(spans),
        has_evidence=bool(phrases),
        matched_phrases=matched,
        total_phrases=len(phrases),
        exact_phrases=exact,
        fuzzy_phrases=fuzzy,
    )


def prepare_evidence_annotations(
    bundle: b1.DataBundle,
) -> tuple[list[EvidenceAnnotation], dict[str, Any]]:
    evidence_column = b1._find_column(
        bundle.frame,
        ["evidence for suicide risk level", "evidence", "risk_evidence"],
    )
    values = bundle.frame[evidence_column].tolist()
    annotations = [
        find_evidence_spans(text, value) for text, value in zip(bundle.texts, values)
    ]
    total = sum(item.total_phrases for item in annotations)
    matched = sum(item.matched_phrases for item in annotations)
    exact = sum(item.exact_phrases for item in annotations)
    fuzzy = sum(item.fuzzy_phrases for item in annotations)
    diagnostics = {
        "evidence_column": evidence_column,
        "rows": len(annotations),
        "rows_with_evidence": sum(item.has_evidence for item in annotations),
        "total_non_none_phrases": total,
        "matched_phrases": matched,
        "exact_phrases": exact,
        "fuzzy_phrases": fuzzy,
        "match_rate": matched / max(1, total),
        "unmatched_phrases": total - matched,
    }
    return annotations, diagnostics


class RiskEvidenceDataset(Dataset):
    def __init__(
        self,
        bundle: b1.DataBundle,
        annotations: Sequence[EvidenceAnnotation],
        indices: Sequence[int],
    ):
        self.bundle = bundle
        self.annotations = annotations
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = int(self.indices[item])
        return {
            "idx": index,
            "text": self.bundle.texts[index],
            "risk": int(self.bundle.risk_ids[index]),
            "annotation": self.annotations[index],
        }


class RiskEvidenceCollator:
    def __init__(self, tokenizer: Any, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, items: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            [item["text"] for item in items],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        token_valid = (offsets[..., 1] > offsets[..., 0]) & encoded["attention_mask"].bool()
        gold = torch.zeros_like(encoded["attention_mask"], dtype=torch.float32)
        evidence_supervised = torch.ones(len(items), dtype=torch.float32)
        has_positive = torch.zeros(len(items), dtype=torch.float32)
        for row, item in enumerate(items):
            annotation: EvidenceAnnotation = item["annotation"]
            for token in range(offsets.shape[1]):
                start, end = offsets[row, token].tolist()
                if end <= start:
                    continue
                if any(start < span_end and end > span_start for span_start, span_end in annotation.spans):
                    gold[row, token] = 1.0
            visible_positive = bool(gold[row].sum() > 0)
            if annotation.has_evidence and not visible_positive:
                # Either fuzzy matching failed or every gold phrase was truncated.
                evidence_supervised[row] = 0.0
            if visible_positive:
                has_positive[row] = 1.0
        encoded.update(
            {
                "idx": torch.tensor([item["idx"] for item in items], dtype=torch.long),
                "risk_y": torch.tensor([item["risk"] for item in items], dtype=torch.long),
                "offset_mapping": offsets.long(),
                "token_valid": token_valid,
                "evidence_y": gold,
                "evidence_supervised": evidence_supervised,
                "has_positive_evidence": has_positive,
            }
        )
        return encoded


@dataclass
class RiskEvidenceConfig:
    name: str
    model_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 512
    epochs: int = 1
    train_batch_size: int = 8
    eval_batch_size: int = 16
    grad_accumulation: int = 2
    backbone_lr: float = 1.5e-5
    head_lr: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    seed: int = 42
    n_splits: int = 3
    use_ordinal: bool = False
    use_evidence: bool = False
    condition_risk_on_evidence: bool = False
    use_counterfactual: bool = False
    lambda_ordinal: float = 0.25
    lambda_evidence: float = 0.45
    lambda_dice: float = 0.20
    lambda_sufficiency: float = 0.15
    lambda_comprehensiveness: float = 0.05
    evidence_positive_weight: float = 5.0
    comprehensiveness_margin: float = 0.25
    ordinal_blend: float = 0.25
    max_grad_norm: float = 1.0
    num_workers: int = 2
    save_checkpoints: bool = False
    resume: bool = True


class RiskEvidenceJointModel(nn.Module):
    def __init__(self, config: RiskEvidenceConfig):
        super().__init__()
        self.config = config
        self.backbone = b1.load_backbone(config.model_name)
        hidden = int(self.backbone.config.hidden_size)
        self.dropout = nn.Dropout(0.10)
        self.evidence_head = nn.Linear(hidden, 1) if config.use_evidence else None
        risk_input = 2 * hidden if config.condition_risk_on_evidence else hidden
        self.risk_head = nn.Linear(risk_input, len(b1.RISK_LABELS))
        self.ordinal_head = nn.Linear(risk_input, len(b1.RISK_LABELS) - 1) if config.use_ordinal else None
        self.evidence_risk_head = (
            nn.Linear(hidden, len(b1.RISK_LABELS)) if config.use_counterfactual else None
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        inputs = {
            key: batch[key]
            for key in ("input_ids", "attention_mask", "token_type_ids")
            if key in batch
        }
        hidden = self.backbone(**inputs).last_hidden_state
        document = self.dropout(b1.masked_mean(hidden, batch["attention_mask"]))
        result: dict[str, torch.Tensor] = {}
        evidence_pool = document
        remaining_pool = document
        if self.evidence_head is not None:
            evidence_logits = self.evidence_head(hidden).squeeze(-1).float()
            valid = batch["token_valid"].to(hidden.dtype)
            evidence_weight = torch.sigmoid(evidence_logits).to(hidden.dtype) * valid
            evidence_pool = torch.sum(hidden * evidence_weight[..., None], dim=1)
            evidence_pool = evidence_pool / evidence_weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
            remaining_weight = (1.0 - torch.sigmoid(evidence_logits)).to(hidden.dtype) * valid
            remaining_pool = torch.sum(hidden * remaining_weight[..., None], dim=1)
            remaining_pool = remaining_pool / remaining_weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
            result["evidence_logits"] = evidence_logits
        risk_features = (
            torch.cat([document, evidence_pool], dim=-1)
            if self.config.condition_risk_on_evidence
            else document
        )
        risk_features = self.dropout(risk_features)
        result["risk_logits"] = self.risk_head(risk_features).float()
        if self.ordinal_head is not None:
            result["ordinal_logits"] = self.ordinal_head(risk_features).float()
        if self.evidence_risk_head is not None:
            result["evidence_risk_logits"] = self.evidence_risk_head(self.dropout(evidence_pool)).float()
            result["remaining_risk_logits"] = self.evidence_risk_head(self.dropout(remaining_pool)).float()
        return result


def _dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits) * valid
    targets = targets * valid
    intersection = (probabilities * targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def risk_evidence_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: RiskEvidenceConfig,
    constraint_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    risk = F.cross_entropy(output["risk_logits"], batch["risk_y"])
    total = risk
    zero = risk.new_zeros(())
    ordinal = zero
    evidence = zero
    dice = zero
    sufficiency = zero
    comprehensiveness = zero
    if config.use_ordinal:
        ordinal_targets = (
            batch["risk_y"][:, None] > torch.arange(3, device=batch["risk_y"].device)[None, :]
        ).to(output["ordinal_logits"].dtype)
        ordinal = F.binary_cross_entropy_with_logits(output["ordinal_logits"], ordinal_targets)
        total = total + config.lambda_ordinal * ordinal
    if config.use_evidence:
        valid = batch["token_valid"].to(output["evidence_logits"].dtype)
        valid = valid * batch["evidence_supervised"][:, None]
        token_loss = F.binary_cross_entropy_with_logits(
            output["evidence_logits"], batch["evidence_y"], reduction="none"
        )
        positive_scale = torch.where(
            batch["evidence_y"] > 0.5,
            torch.full_like(token_loss, config.evidence_positive_weight),
            torch.ones_like(token_loss),
        )
        evidence = (token_loss * positive_scale * valid).sum() / valid.sum().clamp_min(1.0)
        supervised_rows = batch["evidence_supervised"] > 0.5
        if supervised_rows.any():
            dice = _dice_loss(
                output["evidence_logits"][supervised_rows],
                batch["evidence_y"][supervised_rows],
                batch["token_valid"][supervised_rows].to(output["evidence_logits"].dtype),
            )
        total = total + config.lambda_evidence * evidence + config.lambda_dice * dice
    if config.use_counterfactual:
        positive_rows = batch["has_positive_evidence"] > 0.5
        if positive_rows.any():
            sufficiency = F.cross_entropy(
                output["evidence_risk_logits"][positive_rows], batch["risk_y"][positive_rows]
            )
            labels = batch["risk_y"][positive_rows]
            evidence_score = output["evidence_risk_logits"][positive_rows].gather(
                1, labels[:, None]
            ).squeeze(1)
            remaining_score = output["remaining_risk_logits"][positive_rows].gather(
                1, labels[:, None]
            ).squeeze(1)
            comprehensiveness = F.relu(
                config.comprehensiveness_margin - evidence_score + remaining_score
            ).mean()
        total = total + constraint_scale * config.lambda_sufficiency * sufficiency
        total = total + constraint_scale * config.lambda_comprehensiveness * comprehensiveness
    return total, {
        "risk": float(risk.detach().cpu()),
        "ordinal": float(ordinal.detach().cpu()),
        "evidence": float(evidence.detach().cpu()),
        "dice": float(dice.detach().cpu()),
        "sufficiency": float(sufficiency.detach().cpu()),
        "comprehensiveness": float(comprehensiveness.detach().cpu()),
    }


def ordinal_distribution(ordinal_logits: np.ndarray) -> np.ndarray:
    q = sigmoid(ordinal_logits)
    q = np.minimum.accumulate(q, axis=1)
    distribution = np.column_stack(
        [1.0 - q[:, 0], q[:, 0] - q[:, 1], q[:, 1] - q[:, 2], q[:, 2]]
    )
    distribution = np.clip(distribution, 0.0, 1.0)
    return distribution / distribution.sum(axis=1, keepdims=True).clip(min=1e-8)


def risk_probabilities(
    risk_logits: np.ndarray,
    ordinal_logits: np.ndarray | None,
    ordinal_blend: float,
) -> np.ndarray:
    shifted = risk_logits - risk_logits.max(axis=1, keepdims=True)
    class_probabilities = np.exp(shifted)
    class_probabilities /= class_probabilities.sum(axis=1, keepdims=True)
    if ordinal_logits is None:
        return class_probabilities
    return (
        (1.0 - ordinal_blend) * class_probabilities
        + ordinal_blend * ordinal_distribution(ordinal_logits)
    )


@torch.no_grad()
def evaluate_risk_evidence_loader(
    model: RiskEvidenceJointModel,
    loader: DataLoader,
    config: RiskEvidenceConfig,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    indices: list[np.ndarray] = []
    risk_logits: list[np.ndarray] = []
    ordinal_logits: list[np.ndarray] = []
    token_probabilities: dict[int, np.ndarray] = {}
    token_targets: dict[int, np.ndarray] = {}
    token_offsets: dict[int, np.ndarray] = {}
    evidence_supervised: dict[int, bool] = {}
    for raw_batch in loader:
        batch = b1.move_batch(raw_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(batch)
        batch_indices = batch["idx"].cpu().numpy()
        indices.append(batch_indices)
        risk_logits.append(output["risk_logits"].cpu().numpy())
        if config.use_ordinal:
            ordinal_logits.append(output["ordinal_logits"].cpu().numpy())
        if config.use_evidence:
            probabilities = torch.sigmoid(output["evidence_logits"]).cpu().numpy()
            valid = batch["token_valid"].cpu().numpy().astype(bool)
            targets = batch["evidence_y"].cpu().numpy()
            offsets = batch["offset_mapping"].cpu().numpy()
            supervised = batch["evidence_supervised"].cpu().numpy() > 0.5
            for row, index in enumerate(batch_indices):
                keep = valid[row]
                token_probabilities[int(index)] = probabilities[row][keep].astype(np.float32)
                token_targets[int(index)] = targets[row][keep].astype(np.int8)
                token_offsets[int(index)] = offsets[row][keep].astype(np.int32)
                evidence_supervised[int(index)] = bool(supervised[row])
    all_indices = np.concatenate(indices)
    all_risk_logits = np.concatenate(risk_logits)
    all_ordinal = np.concatenate(ordinal_logits) if ordinal_logits else None
    probabilities = risk_probabilities(all_risk_logits, all_ordinal, config.ordinal_blend)
    # Loader order is deterministic for validation, but metrics explicitly use idx.
    targets = np.asarray([loader.dataset.bundle.risk_ids[index] for index in all_indices])
    predictions = probabilities.argmax(axis=1)
    result = {
        "idx": all_indices,
        "risk_logits": all_risk_logits,
        "ordinal_logits": all_ordinal,
        "risk_weighted_f1": float(
            f1_score(targets, predictions, average="weighted", zero_division=0)
        ),
        "risk_macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "token_probabilities": token_probabilities,
        "token_targets": token_targets,
        "token_offsets": token_offsets,
        "evidence_supervised": evidence_supervised,
    }
    if config.use_evidence:
        y_true = []
        y_score = []
        for index in all_indices:
            if evidence_supervised[int(index)]:
                y_true.append(token_targets[int(index)])
                y_score.append(token_probabilities[int(index)])
        if y_true and np.concatenate(y_true).sum() > 0:
            result["evidence_token_ap"] = float(
                average_precision_score(np.concatenate(y_true), np.concatenate(y_score))
            )
        else:
            result["evidence_token_ap"] = 0.0
    else:
        result["evidence_token_ap"] = 0.0
    return result


def _risk_optimizer(
    model: RiskEvidenceJointModel,
    config: RiskEvidenceConfig,
) -> torch.optim.Optimizer:
    backbone, heads = [], []
    for name, parameter in model.named_parameters():
        (backbone if name.startswith("backbone.") else heads).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": config.backbone_lr},
            {"params": heads, "lr": config.head_lr},
        ],
        weight_decay=config.weight_decay,
    )


def train_risk_evidence_fold(
    bundle: b1.DataBundle,
    annotations: Sequence[EvidenceAnnotation],
    folds: np.ndarray,
    fold: int,
    config: RiskEvidenceConfig,
    output_dir: Path,
) -> dict[str, Any]:
    b1.seed_everything(config.seed + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    collator = RiskEvidenceCollator(tokenizer, config.max_length)
    train_indices = np.where(folds != fold)[0]
    validation_indices = np.where(folds == fold)[0]
    train_loader = DataLoader(
        RiskEvidenceDataset(bundle, annotations, train_indices),
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        RiskEvidenceDataset(bundle, annotations, validation_indices),
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )
    model = RiskEvidenceJointModel(config).to(device)
    optimizer = _risk_optimizer(model, config)
    updates_per_epoch = math.ceil(len(train_loader) / config.grad_accumulation)
    total_updates = max(1, updates_per_epoch * config.epochs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * config.warmup_ratio),
        num_training_steps=total_updates,
    )
    best_metric = -np.inf
    history: list[dict[str, Any]] = []
    temp_dir = Path(tempfile.mkdtemp(prefix=f"risk_evidence_{fold}_"))
    best_path = temp_dir / "best.pt"
    started = time.perf_counter()

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        keys = [
            "loss", "risk", "ordinal", "evidence", "dice", "sufficiency", "comprehensiveness"
        ]
        running = {key: 0.0 for key in keys}
        for step, raw_batch in enumerate(train_loader):
            batch = b1.move_batch(raw_batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(batch)
                global_step = epoch * len(train_loader) + step + 1
                constraint_scale = min(1.0, global_step / max(1.0, 0.25 * len(train_loader) * config.epochs))
                loss, pieces = risk_evidence_loss(
                    output, batch, config, constraint_scale=constraint_scale
                )
                scaled = loss / config.grad_accumulation
            scaled.backward()
            running["loss"] += float(loss.detach().cpu())
            for key, value in pieces.items():
                running[key] += value
            should_step = (step + 1) % config.grad_accumulation == 0 or step + 1 == len(train_loader)
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        validation = evaluate_risk_evidence_loader(model, validation_loader, config, device)
        selection = validation["risk_weighted_f1"]
        if config.use_evidence:
            selection = 0.8 * selection + 0.2 * validation["evidence_token_ap"]
        record = {
            "epoch": epoch + 1,
            **{key: value / max(1, len(train_loader)) for key, value in running.items()},
            "val_risk_weighted_f1": validation["risk_weighted_f1"],
            "val_risk_macro_f1": validation["risk_macro_f1"],
            "val_evidence_token_ap": validation["evidence_token_ap"],
            "selection_metric": float(selection),
        }
        history.append(record)
        print(f"[{config.name}] fold={fold} {record}")
        if selection > best_metric:
            best_metric = selection
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    validation = evaluate_risk_evidence_loader(model, validation_loader, config, device)
    elapsed = time.perf_counter() - started
    if config.save_checkpoints:
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / f"fold_{fold}.pt")
    result = {**validation, "history": history, "elapsed_seconds": elapsed}
    del model, optimizer, scheduler, train_loader, validation_loader
    shutil.rmtree(temp_dir, ignore_errors=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _best_evidence_threshold(
    token_probabilities: Sequence[np.ndarray],
    token_targets: Sequence[np.ndarray],
    supervised: np.ndarray,
    indices: np.ndarray,
) -> float:
    grid = np.linspace(0.15, 0.75, 25)
    selected = [int(index) for index in indices if supervised[int(index)]]
    if not selected:
        return 0.5
    y_true = np.concatenate([token_targets[index] for index in selected])
    y_score = np.concatenate([token_probabilities[index] for index in selected])
    scores = [f1_score(y_true, y_score >= threshold, zero_division=0) for threshold in grid]
    return float(grid[int(np.argmax(scores))])


def _token_predictions_to_spans(
    probabilities: np.ndarray,
    offsets: np.ndarray,
    threshold: float,
) -> list[tuple[int, int]]:
    selected = np.where(probabilities >= threshold)[0]
    spans: list[list[int]] = []
    for token in selected:
        start, end = map(int, offsets[token])
        if end <= start:
            continue
        if spans and start <= spans[-1][1] + 2:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end])
    return [(start, end) for start, end in spans]


def _interval_f1(left: tuple[int, int], right: tuple[int, int]) -> float:
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    if intersection == 0:
        return 0.0
    return 2.0 * intersection / max(1, (left[1] - left[0]) + (right[1] - right[0]))


def evidence_phrase_f1(
    predicted: Sequence[Sequence[tuple[int, int]]],
    gold: Sequence[Sequence[tuple[int, int]]],
) -> float:
    true_positive = false_positive = false_negative = 0
    for predicted_spans, gold_spans in zip(predicted, gold):
        candidates = sorted(
            [
                (_interval_f1(pred, target), p_idx, g_idx)
                for p_idx, pred in enumerate(predicted_spans)
                for g_idx, target in enumerate(gold_spans)
                if _interval_f1(pred, target) >= 0.5
            ],
            reverse=True,
        )
        used_pred: set[int] = set()
        used_gold: set[int] = set()
        for _, p_idx, g_idx in candidates:
            if p_idx not in used_pred and g_idx not in used_gold:
                used_pred.add(p_idx)
                used_gold.add(g_idx)
        true_positive += len(used_pred)
        false_positive += len(predicted_spans) - len(used_pred)
        false_negative += len(gold_spans) - len(used_gold)
    return 2.0 * true_positive / max(1, 2 * true_positive + false_positive + false_negative)


def finalize_risk_evidence_oof(
    bundle: b1.DataBundle,
    annotations: Sequence[EvidenceAnnotation],
    folds: np.ndarray,
    risk_logits: np.ndarray,
    ordinal_logits: np.ndarray | None,
    token_probabilities: Sequence[np.ndarray],
    token_targets: Sequence[np.ndarray],
    token_offsets: Sequence[np.ndarray],
    supervised: np.ndarray,
    config: RiskEvidenceConfig,
) -> tuple[dict[str, Any], np.ndarray, list[list[tuple[int, int]]], np.ndarray]:
    probabilities = risk_probabilities(risk_logits, ordinal_logits, config.ordinal_blend)
    risk_predictions = probabilities.argmax(axis=1)
    metrics: dict[str, Any] = {
        "risk_weighted_f1": float(
            f1_score(bundle.risk_ids, risk_predictions, average="weighted", zero_division=0)
        ),
        "risk_macro_f1": float(
            f1_score(bundle.risk_ids, risk_predictions, average="macro", zero_division=0)
        ),
    }
    predicted_spans: list[list[tuple[int, int]]] = [[] for _ in bundle.texts]
    fold_thresholds: list[float] = []
    if config.use_evidence:
        for fold_value in sorted(np.unique(folds)):
            train_indices = np.where(folds != fold_value)[0]
            valid_indices = np.where(folds == fold_value)[0]
            threshold = _best_evidence_threshold(
                token_probabilities, token_targets, supervised, train_indices
            )
            fold_thresholds.append(threshold)
            for index in valid_indices:
                predicted_spans[index] = _token_predictions_to_spans(
                    token_probabilities[index], token_offsets[index], threshold
                )
        supervised_indices = np.where(supervised)[0]
        y_true = np.concatenate([token_targets[index] for index in supervised_indices])
        y_pred = np.concatenate(
            [
                (token_probabilities[index] >= fold_thresholds[int(folds[index])]).astype(np.int8)
                for index in supervised_indices
            ]
        )
        metrics["evidence_token_f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        metrics["evidence_phrase_f1_proxy"] = float(
            evidence_phrase_f1(predicted_spans, [list(item.spans) for item in annotations])
        )
        metrics["partial_composite_proxy"] = (
            0.4 * metrics["risk_weighted_f1"]
            + 0.3 * metrics["evidence_phrase_f1_proxy"]
        )
    else:
        metrics["evidence_token_f1"] = None
        metrics["evidence_phrase_f1_proxy"] = None
        metrics["partial_composite_proxy"] = None
    return metrics, risk_predictions, predicted_spans, np.asarray(fold_thresholds)


def run_risk_evidence_experiment(
    bundle: b1.DataBundle,
    annotations: Sequence[EvidenceAnnotation],
    folds: np.ndarray,
    config: RiskEvidenceConfig,
    artifact_root: str | Path,
) -> dict[str, Any]:
    output_dir = Path(artifact_root) / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(asdict(config), output_dir / "config.json")
    n = len(bundle.texts)
    risk_logits = np.full((n, len(b1.RISK_LABELS)), np.nan, dtype=np.float32)
    ordinal_logits = (
        np.full((n, len(b1.RISK_LABELS) - 1), np.nan, dtype=np.float32)
        if config.use_ordinal
        else None
    )
    token_probabilities: list[np.ndarray] = [np.empty(0, dtype=np.float32) for _ in range(n)]
    token_targets: list[np.ndarray] = [np.empty(0, dtype=np.int8) for _ in range(n)]
    token_offsets: list[np.ndarray] = [np.empty((0, 2), dtype=np.int32) for _ in range(n)]
    supervised = np.zeros(n, dtype=bool)
    elapsed_by_fold: list[float] = []
    started = time.perf_counter()

    for position, fold_value in enumerate(sorted(np.unique(folds))):
        fold = int(fold_value)
        fold_path = output_dir / f"fold_{fold}_oof.npz"
        if config.resume and fold_path.exists():
            saved = np.load(fold_path, allow_pickle=True)
            idx = saved["idx"].astype(np.int64)
            risk_logits[idx] = saved["risk_logits"]
            if config.use_ordinal:
                ordinal_logits[idx] = saved["ordinal_logits"]
            if config.use_evidence:
                for local, index in enumerate(idx):
                    token_probabilities[index] = saved["token_probabilities"][local]
                    token_targets[index] = saved["token_targets"][local]
                    token_offsets[index] = saved["token_offsets"][local]
                    supervised[index] = bool(saved["evidence_supervised"][local])
            elapsed_by_fold.append(float(saved["elapsed_seconds"]))
            print(f"[{config.name}] resumed fold {fold}")
            continue
        result = train_risk_evidence_fold(
            bundle, annotations, folds, fold, config, output_dir
        )
        idx = np.asarray(result["idx"], dtype=np.int64)
        risk_logits[idx] = result["risk_logits"]
        if config.use_ordinal:
            ordinal_logits[idx] = result["ordinal_logits"]
        fold_token_probabilities = []
        fold_token_targets = []
        fold_token_offsets = []
        fold_supervised = []
        if config.use_evidence:
            for index in idx:
                index = int(index)
                token_probabilities[index] = result["token_probabilities"][index]
                token_targets[index] = result["token_targets"][index]
                token_offsets[index] = result["token_offsets"][index]
                supervised[index] = result["evidence_supervised"][index]
                fold_token_probabilities.append(token_probabilities[index])
                fold_token_targets.append(token_targets[index])
                fold_token_offsets.append(token_offsets[index])
                fold_supervised.append(supervised[index])
        elapsed = float(result["elapsed_seconds"])
        elapsed_by_fold.append(elapsed)
        payload = {
            "idx": idx,
            "risk_logits": result["risk_logits"],
            "elapsed_seconds": np.asarray(elapsed),
        }
        if config.use_ordinal:
            payload["ordinal_logits"] = result["ordinal_logits"]
        if config.use_evidence:
            payload.update(
                {
                    "token_probabilities": np.asarray(fold_token_probabilities, dtype=object),
                    "token_targets": np.asarray(fold_token_targets, dtype=object),
                    "token_offsets": np.asarray(fold_token_offsets, dtype=object),
                    "evidence_supervised": np.asarray(fold_supervised, dtype=bool),
                }
            )
        np.savez_compressed(fold_path, **payload)
        _json_dump(result["history"], output_dir / f"fold_{fold}_history.json")
        remaining = len(np.unique(folds)) - position - 1
        print(
            f"[{config.name}] fold {fold}: {elapsed / 60:.1f} min; "
            f"estimated {remaining * np.median(elapsed_by_fold) / 60:.1f} min remaining"
        )

    if np.isnan(risk_logits).any() or (
        ordinal_logits is not None and np.isnan(ordinal_logits).any()
    ):
        raise RuntimeError(f"{config.name}: incomplete OOF logits")
    metrics, risk_predictions, predicted_spans, evidence_thresholds = finalize_risk_evidence_oof(
        bundle,
        annotations,
        folds,
        risk_logits,
        ordinal_logits,
        token_probabilities,
        token_targets,
        token_offsets,
        supervised,
        config,
    )
    metrics.update(
        {
            "experiment": config.name,
            "elapsed_minutes": (time.perf_counter() - started) / 60.0,
            "recorded_fold_minutes": float(np.sum(elapsed_by_fold) / 60.0),
            "median_fold_minutes": float(np.median(elapsed_by_fold) / 60.0),
            "model_name": config.model_name,
            "max_length": config.max_length,
            "epochs": config.epochs,
            "use_ordinal": config.use_ordinal,
            "use_evidence": config.use_evidence,
            "condition_risk_on_evidence": config.condition_risk_on_evidence,
            "use_counterfactual": config.use_counterfactual,
        }
    )
    np.savez_compressed(
        output_dir / "oof_complete.npz",
        risk_logits=risk_logits,
        ordinal_logits=(ordinal_logits if ordinal_logits is not None else np.empty((n, 0))),
        risk_predictions=risk_predictions,
        folds=folds,
        row_ids=bundle.row_ids,
        evidence_thresholds=evidence_thresholds,
    )
    span_rows = []
    for index, spans in enumerate(predicted_spans):
        span_rows.append(
            {
                "row_id": str(bundle.row_ids[index]),
                "predicted_spans": spans,
                "predicted_phrases": [bundle.texts[index][start:end] for start, end in spans],
                "gold_spans": list(annotations[index].spans),
                "gold_phrases": [
                    bundle.texts[index][start:end] for start, end in annotations[index].spans
                ],
                "evidence_supervised": bool(supervised[index]),
            }
        )
    _json_dump(span_rows, output_dir / "oof_evidence_spans.json")
    _json_dump(metrics, output_dir / "summary.json")
    print(f"[{config.name}] summary: {metrics}")
    return metrics


def innovation_decision(
    factor_baseline: dict[str, Any],
    factor_plain: dict[str, Any],
    factor_full: dict[str, Any],
    risk_results: Sequence[dict[str, Any]],
    artifact_root: str | Path,
) -> dict[str, Any]:
    factor_delta = factor_full["macro_f1"] - factor_baseline["macro_f1"]
    tail_delta = factor_full["tail_macro_f1"] - factor_baseline["tail_macro_f1"]
    factor_constraints_delta = factor_full["macro_f1"] - factor_plain["macro_f1"]
    risk_by_name = {item["experiment"]: item for item in risk_results}
    risk_only_candidates = [item for item in risk_results if not item.get("use_evidence", False)]
    risk_only = max(risk_only_candidates, key=lambda item: item["risk_weighted_f1"])
    joint_candidates = [
        item for item in risk_results if item.get("condition_risk_on_evidence", False)
    ]
    if not joint_candidates:
        raise ValueError("risk_results does not contain an evidence-conditioned joint model")
    joint = max(
        joint_candidates,
        key=lambda item: item.get("partial_composite_proxy") or -np.inf,
    )
    multitask_candidates = [
        item
        for item in risk_results
        if item.get("use_evidence", False)
        and not item.get("condition_risk_on_evidence", False)
    ]
    multitask = (
        max(
            multitask_candidates,
            key=lambda item: item.get("partial_composite_proxy") or -np.inf,
        )
        if multitask_candidates
        else None
    )
    risk_delta = joint["risk_weighted_f1"] - risk_only["risk_weighted_f1"]
    conditioning_delta = (
        joint["partial_composite_proxy"] - multitask["partial_composite_proxy"]
        if multitask is not None
        else None
    )
    head_delta = factor_full["head_macro_f1"] - factor_baseline["head_macro_f1"]
    factor_accept = bool(
        factor_delta >= 0.005
        and tail_delta >= 0.015
        and head_delta >= -0.005
        and factor_constraints_delta >= 0.0
    )
    risk_accept = bool(
        risk_delta >= -0.005
        and joint.get("evidence_phrase_f1_proxy") is not None
        and joint["evidence_phrase_f1_proxy"] >= 0.30
        and (conditioning_delta is None or conditioning_delta >= 0.002)
    )
    decision = {
        "version": "B1-INNOVATION-VALIDATION",
        "factor": {
            "accepted": factor_accept,
            "delta_macro_f1_vs_doc": factor_delta,
            "delta_tail_macro_f1_vs_doc": tail_delta,
            "delta_head_macro_f1_vs_doc": head_delta,
            "delta_constraints_vs_plain_bottleneck": factor_constraints_delta,
            "recommended": factor_full["experiment"] if factor_accept else factor_baseline["experiment"],
        },
        "risk_evidence": {
            "accepted": risk_accept,
            "delta_risk_weighted_f1_vs_risk_only": risk_delta,
            "delta_partial_composite_vs_plain_multitask": conditioning_delta,
            "evidence_phrase_f1_proxy": joint.get("evidence_phrase_f1_proxy"),
            "recommended": joint["experiment"] if risk_accept else risk_only["experiment"],
        },
        "risk_results": risk_by_name,
    }
    _json_dump(decision, Path(artifact_root) / "B1_INNOVATION_DECISION.json")
    return decision


def environment_report() -> dict[str, Any]:
    return {
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "factor_labels": N_LABELS,
    }
