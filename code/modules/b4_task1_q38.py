"""B4 Task-1: Qwen3.8 Full64 evidence-conditioned risk detector.

The task is treated as two coupled verification problems rather than a single
four-way generation prompt:

1. score four operational risk cards from the document;
2. propose exact, fold-safe candidate spans and verify them for the provisional
   risk level;
3. score the same four risk cards again while exposing only the selected
   verbatim evidence.

All expensive language-model calls use the resumable A/B-margin harness in
``b4p_anchor_verifier``.  The candidate lexicon is learned from training-fold
gold spans only and therefore must be rebuilt for every outer fold.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

import b1_experiments as b1
import b1_innovation_experiments as inn
import b4p_anchor_verifier as b4
import qwen38_dual_task_experiments as q38


TASK1_RUNTIME_REVISION = "2026-08-24.q38-full64-official-evidence-v4"


@dataclass
class Task1Full64Config:
    model_name: str = "Qwen/Qwen3.8-27B"
    fold: int = 0
    n_splits: int = 3
    max_length: int = 1536
    context_chars: int = 5000
    candidate_max_chars: int = 180
    candidate_caps_for_audit: tuple[int, ...] = (64, 96, 128, 160, 192)
    validation_candidates_per_post: int = 64
    lexicon_max_unigrams: int = 700
    lexicon_max_bigrams: int = 500
    lexicon_positions_per_post: int = 28
    evidence_positive_variants_per_span: int = 1
    evidence_negatives_per_post: int = 3
    evidence_top_k: int = 3
    evidence_margin_threshold: float = 0.0
    evidence_length_penalty: float = 0.002
    conditioned_risk_blend_weight: float = 0.65
    evidence_conditioned_training_fraction: float = 0.65
    sft_epochs: float = 1.0
    sft_max_steps: int = -1
    learning_rate: float = 1.0e-4
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
    attention_implementation: str = "flash_attention_2"
    qwen35_fa2_position_guard: bool = True
    require_qwen35_fast_kernels: bool = True

    def b4_config(self) -> b4.B4PConfig:
        return b4.B4PConfig(
            seed=self.seed,
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


@dataclass
class EvidenceLexicon:
    unigram_scores: dict[str, float]
    bigram_scores: dict[str, float]
    training_rows: list[int]

    def to_json(self) -> dict[str, Any]:
        return {
            "unigram_scores": self.unigram_scores,
            "bigram_scores": self.bigram_scores,
            "training_rows": self.training_rows,
        }

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "EvidenceLexicon":
        return cls(
            unigram_scores={str(k): float(v) for k, v in value["unigram_scores"].items()},
            bigram_scores={str(k): float(v) for k, v in value["bigram_scores"].items()},
            training_rows=[int(item) for item in value["training_rows"]],
        )


@dataclass
class TokenProposalCache:
    probabilities: dict[int, np.ndarray]
    offsets: dict[int, np.ndarray]
    metrics: dict[str, Any]


def _proposal_candidates(
    probabilities: np.ndarray,
    offsets: np.ndarray,
    max_chars: int,
) -> list[q38.CandidateSpan]:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    offsets = np.asarray(offsets, dtype=np.int32)
    valid = np.flatnonzero(offsets[:, 1] > offsets[:, 0])
    if len(valid) == 0:
        return []
    raw: list[q38.CandidateSpan] = []
    # Contiguous token-head regions at several fixed thresholds provide
    # variable-length spans without tuning a threshold on the outer fold.
    for threshold in (0.20, 0.35, 0.50, 0.65):
        selected = valid[probabilities[valid] >= threshold]
        groups: list[list[int]] = []
        for token in selected:
            if groups and token <= groups[-1][-1] + 1:
                groups[-1].append(int(token))
            else:
                groups.append([int(token)])
        for group in groups:
            left, right = int(offsets[group[0], 0]), int(offsets[group[-1], 1])
            if 0 < right - left <= max_chars:
                raw.append(q38.CandidateSpan(left, right, f"token_run_{threshold:.2f}"))

    ranked_tokens = valid[np.argsort(probabilities[valid])[::-1]]
    centers: list[int] = []
    for token in ranked_tokens:
        if any(abs(int(token) - previous) <= 1 for previous in centers):
            continue
        centers.append(int(token))
        if len(centers) >= 32:
            break
    for center in centers:
        position = int(np.where(valid == center)[0][0])
        for width in (1, 2, 3, 4, 5, 7, 10, 14):
            for offset in (0, width // 2, width - 1):
                lo = max(0, position - offset)
                hi = min(len(valid), lo + width)
                lo = max(0, hi - width)
                left = int(offsets[valid[lo], 0])
                right = int(offsets[valid[hi - 1], 1])
                if 0 < right - left <= max_chars:
                    raw.append(q38.CandidateSpan(left, right, f"token_peak_{width}"))
    return raw


def _proposal_span_score(
    candidate: q38.CandidateSpan,
    probabilities: np.ndarray | None,
    offsets: np.ndarray | None,
) -> float:
    if probabilities is None or offsets is None:
        return 0.0
    probabilities = np.asarray(probabilities)
    offsets = np.asarray(offsets)
    overlap = (
        (offsets[:, 0] < candidate.right)
        & (offsets[:, 1] > candidate.left)
        & (offsets[:, 1] > offsets[:, 0])
    )
    values = probabilities[overlap]
    if len(values) == 0:
        return 0.0
    return float(2.5 * values.max() + values.mean())


_WORD_RE = re.compile(r"\b[\w']+\b")


def _word_spans(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0).lower(), m.start(), m.end()) for m in _WORD_RE.finditer(str(text))]


def _log_odds_scores(
    positive: Counter[str],
    background: Counter[str],
    minimum_positive_count: int,
    maximum_items: int,
) -> dict[str, float]:
    vocabulary = set(positive)
    positive_total = sum(positive.values())
    background_total = sum(background.values())
    alpha = 0.5
    width = max(1, len(vocabulary))
    scored: list[tuple[str, float]] = []
    for token in vocabulary:
        if positive[token] < minimum_positive_count:
            continue
        p = (positive[token] + alpha) / (positive_total + alpha * width)
        q = (background[token] + alpha) / (background_total + alpha * width)
        score = math.log(p / q)
        if score > -0.5:
            scored.append((token, score))
    scored.sort(key=lambda item: (item[1], positive[item[0]]), reverse=True)
    return {token: float(score) for token, score in scored[:maximum_items]}


def fit_evidence_lexicon(
    bundle: b1.DataBundle,
    annotations: Sequence[inn.EvidenceAnnotation],
    train_rows: Sequence[int],
    config: Task1Full64Config,
) -> EvidenceLexicon:
    positive_uni: Counter[str] = Counter()
    positive_bi: Counter[str] = Counter()
    background_uni: Counter[str] = Counter()
    background_bi: Counter[str] = Counter()
    for row in map(int, train_rows):
        document_tokens = [token for token, _, _ in _word_spans(bundle.texts[row])]
        background_uni.update(document_tokens)
        background_bi.update(" ".join(pair) for pair in zip(document_tokens, document_tokens[1:]))
        for left, right in annotations[row].spans:
            tokens = [token for token, _, _ in _word_spans(bundle.texts[row][left:right])]
            positive_uni.update(tokens)
            positive_bi.update(" ".join(pair) for pair in zip(tokens, tokens[1:]))

    unigram = _log_odds_scores(
        positive_uni, background_uni, minimum_positive_count=2,
        maximum_items=config.lexicon_max_unigrams,
    )
    bigram = _log_odds_scores(
        positive_bi, background_bi, minimum_positive_count=2,
        maximum_items=config.lexicon_max_bigrams,
    )
    # Stable clinical/action cues remain available when a rare fold happens to
    # contain only one example.  Their numerical value is below strong learned
    # cues, so they do not swamp the fold-specific lexicon.
    for cue in q38._RISK_CUES:
        unigram[cue] = max(unigram.get(cue, -99.0), 0.75)
    return EvidenceLexicon(unigram, bigram, [int(row) for row in train_rows])


def _candidate_lexical_score(
    text: str,
    candidate: q38.CandidateSpan,
    lexicon: EvidenceLexicon,
) -> float:
    tokens = [token for token, _, _ in _word_spans(text[candidate.left:candidate.right])]
    if not tokens:
        return -10.0
    scores = [lexicon.unigram_scores.get(token, -0.4) for token in tokens]
    bigrams = [lexicon.bigram_scores.get(" ".join(pair), -0.2) for pair in zip(tokens, tokens[1:])]
    positive_sum = sum(max(0.0, value) for value in scores + bigrams)
    peak = max(scores + bigrams, default=-0.4)
    concise = -0.08 * abs(len(tokens) - 5)
    return 1.8 * peak + 0.20 * positive_sum + concise


def generate_fold_safe_candidates(
    text: str,
    lexicon: EvidenceLexicon,
    config: Task1Full64Config,
    max_candidates: int | None = None,
    token_probabilities: np.ndarray | None = None,
    token_offsets: np.ndarray | None = None,
) -> list[q38.CandidateSpan]:
    """Generate exact spans using only a training-fold evidence lexicon."""
    text = str(text or "")
    cap = int(max_candidates or config.validation_candidates_per_post)
    broad = q38.generate_evidence_candidates(
        text,
        max_candidates=max(96, cap * 3),
        max_chars=config.candidate_max_chars,
    )
    words = _word_spans(text)
    position_scores: list[tuple[float, int]] = []
    for index, (token, _, _) in enumerate(words):
        score = lexicon.unigram_scores.get(token, -0.5)
        if index > 0:
            score = max(
                score,
                lexicon.bigram_scores.get(f"{words[index - 1][0]} {token}", -0.5),
            )
        if index + 1 < len(words):
            score = max(
                score,
                lexicon.bigram_scores.get(f"{token} {words[index + 1][0]}", -0.5),
            )
        position_scores.append((score, index))
    position_scores.sort(reverse=True)
    selected_positions: list[int] = []
    for score, index in position_scores:
        if score < 0.0:
            break
        if any(abs(index - previous) <= 1 for previous in selected_positions):
            continue
        selected_positions.append(index)
        if len(selected_positions) >= config.lexicon_positions_per_post:
            break

    raw = list(broad)
    if token_probabilities is not None and token_offsets is not None:
        raw.extend(
            _proposal_candidates(
                token_probabilities, token_offsets, config.candidate_max_chars
            )
        )
    for index in selected_positions:
        for width in (1, 2, 3, 4, 5, 7, 10, 14):
            for offset in (0, width // 2, width - 1):
                lo = max(0, index - offset)
                hi = min(len(words), lo + width)
                lo = max(0, hi - width)
                if hi <= lo:
                    continue
                left, right = words[lo][1], words[hi - 1][2]
                if right - left <= config.candidate_max_chars:
                    raw.append(q38.CandidateSpan(left, right, f"fold_lexicon_{width}"))

    dedup: dict[str, q38.CandidateSpan] = {}
    for item in raw:
        trimmed = q38._trim_span(text, item.left, item.right)
        if trimmed is None:
            continue
        left, right = trimmed
        normalized = b1.normalize_text(text[left:right])
        if len(normalized) < 3:
            continue
        candidate = q38.CandidateSpan(left, right, item.source)
        current = dedup.get(normalized)
        if current is None or (right - left) < (current.right - current.left):
            dedup[normalized] = candidate
    def ranking_score(
        item: q38.CandidateSpan,
        include_token_score: bool,
    ) -> float:
        score = _candidate_lexical_score(text, item, lexicon)
        if include_token_score:
            score += _proposal_span_score(item, token_probabilities, token_offsets)
            score += (
                8.0 if item.source.startswith("token_run") else (
                    2.0 if item.source.startswith("token_peak") else 0.0
                )
            )
        return score

    ranked = sorted(
        dedup.values(),
        key=lambda item: (
            ranking_score(item, include_token_score=True),
            -len(text[item.left:item.right]),
            -item.left,
        ),
        reverse=True,
    )

    # Source-balanced union.  A learned token head can be badly calibrated yet
    # still useful for discovering regions missed by deterministic candidates.
    # First freeze 90% of the budget from a token-independent ranking (including
    # broad document coverage); only the final 10% is allowed to be reordered
    # by the token head.  Thus adding a weak proposer cannot destroy the strong
    # heuristic/lexicon ceiling observed before the proposer was introduced.
    broad_unique: list[q38.CandidateSpan] = []
    broad_seen: set[str] = set()
    for item in broad:
        trimmed = q38._trim_span(text, item.left, item.right)
        if trimmed is None:
            continue
        left, right = trimmed
        normalized = b1.normalize_text(text[left:right])
        if not normalized or normalized in broad_seen:
            continue
        broad_seen.add(normalized)
        broad_unique.append(q38.CandidateSpan(left, right, item.source))
    deterministic = [
        item for item in dedup.values()
        if not item.source.startswith("token_")
    ]
    deterministic_ranked = sorted(
        deterministic,
        key=lambda item: (
            ranking_score(item, include_token_score=False),
            -len(text[item.left:item.right]),
            -item.left,
        ),
        reverse=True,
    )
    deterministic_budget = max(1, int(round(cap * 0.90)))
    broad_budget = min(
        len(broad_unique),
        max(1, int(round(deterministic_budget * 0.65))),
    )
    chosen = list(broad_unique[:broad_budget])
    chosen_norm = {
        b1.normalize_text(text[item.left:item.right]) for item in chosen
    }
    for item in deterministic_ranked:
        normalized = b1.normalize_text(text[item.left:item.right])
        if normalized in chosen_norm:
            continue
        chosen.append(item)
        chosen_norm.add(normalized)
        if len(chosen) >= deterministic_budget:
            break
    for item in ranked:
        normalized = b1.normalize_text(text[item.left:item.right])
        if normalized in chosen_norm:
            continue
        chosen.append(item)
        chosen_norm.add(normalized)
        if len(chosen) >= cap:
            break
    return chosen[:cap]


def audit_candidate_ceiling(
    bundle: b1.DataBundle,
    rows: Sequence[int],
    lexicon: EvidenceLexicon,
    config: Task1Full64Config,
    token_proposals: TokenProposalCache | None = None,
) -> pd.DataFrame:
    gold_all = q38._gold_phrases(bundle)
    records: list[dict[str, Any]] = []
    for cap in config.candidate_caps_for_audit:
        predicted = []
        counts = []
        for row in map(int, rows):
            probability = None if token_proposals is None else token_proposals.probabilities.get(row)
            offsets = None if token_proposals is None else token_proposals.offsets.get(row)
            candidates = generate_fold_safe_candidates(
                bundle.texts[row], lexicon, config, cap,
                token_probabilities=probability, token_offsets=offsets,
            )
            predicted.append(
                [bundle.texts[row][item.left:item.right].strip() for item in candidates]
            )
            counts.append(len(candidates))
        score = q38.official_like_phrase_f1(
            predicted, [gold_all[int(row)] for row in rows]
        )
        records.append(
            {
                "candidate_cap": cap,
                "candidate_recall_ceiling": score["recall"],
                "candidate_oracle_precision": score["precision"],
                "mean_candidates": float(np.mean(counts)),
            }
        )
    return pd.DataFrame(records)


def prepare_modernbert_proposer_fold(
    bundle: b1.DataBundle,
    annotations: Sequence[inn.EvidenceAnnotation],
    folds: np.ndarray,
    fold: int,
    output_dir: str | Path,
    epochs: int = 2,
) -> TokenProposalCache:
    """Train/resume the lightweight token proposer on the identical outer fold."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / f"fold_{fold}_token_proposals.npz"
    metrics_path = output_dir / f"fold_{fold}_metrics.json"
    if cache_path.exists() and metrics_path.exists():
        saved = np.load(cache_path, allow_pickle=True)
        rows = saved["rows"].astype(int)
        return TokenProposalCache(
            probabilities={int(row): saved["token_probabilities"][i] for i, row in enumerate(rows)},
            offsets={int(row): saved["token_offsets"][i] for i, row in enumerate(rows)},
            metrics=json.loads(metrics_path.read_text(encoding="utf-8")),
        )

    config = inn.RiskEvidenceConfig(
        name=f"TASK1_MODERNBERT_PROPOSER_FOLD{fold}",
        model_name="answerdotai/ModernBERT-base",
        max_length=512,
        epochs=epochs,
        n_splits=len(np.unique(folds)),
        use_ordinal=True,
        use_evidence=True,
        condition_risk_on_evidence=True,
        use_counterfactual=True,
        save_checkpoints=True,
        resume=False,
    )
    result = inn.train_risk_evidence_fold(
        bundle, annotations, folds, fold, config, output_dir
    )
    rows = np.asarray(result["idx"], dtype=int)
    probabilities = [result["token_probabilities"][int(row)] for row in rows]
    offsets = [result["token_offsets"][int(row)] for row in rows]
    risk_probability = inn.risk_probabilities(
        result["risk_logits"], result["ordinal_logits"], config.ordinal_blend
    )
    risk_prediction = risk_probability.argmax(axis=1)
    predicted_phrases: list[list[str]] = []
    for row in rows:
        spans = inn._token_predictions_to_spans(
            result["token_probabilities"][int(row)],
            result["token_offsets"][int(row)],
            0.5,
        )
        predicted_phrases.append(
            [bundle.texts[int(row)][left:right] for left, right in spans]
        )
    gold_all = q38._gold_phrases(bundle)
    phrase = q38.official_like_phrase_f1(
        predicted_phrases, [gold_all[int(row)] for row in rows]
    )
    risk_wf1 = float(
        f1_score(
            bundle.risk_ids[rows], risk_prediction,
            average="weighted", zero_division=0,
        )
    )
    raw_points = 0.4 * risk_wf1 + 0.3 * phrase["f1"]
    metrics = {
        "model": config.model_name,
        "fold": int(fold),
        "risk_weighted_f1": risk_wf1,
        "risk_macro_f1": float(
            f1_score(bundle.risk_ids[rows], risk_prediction, average="macro", zero_division=0)
        ),
        "evidence_token_ap": float(result["evidence_token_ap"]),
        "evidence_phrase_f1_official_like_fixed_0_5": phrase["f1"],
        "subtask1_raw_points_out_of_0_7": raw_points,
        "subtask1_normalized_score": raw_points / 0.7,
        "warning": "Same-fold ModernBERT proposer/baseline; fixed 0.5 evidence threshold.",
    }
    np.savez_compressed(
        cache_path,
        rows=rows,
        token_probabilities=np.asarray(probabilities, dtype=object),
        token_offsets=np.asarray(offsets, dtype=object),
        risk_probabilities=risk_probability,
    )
    b4.json_dump(metrics, metrics_path)
    return TokenProposalCache(
        probabilities={int(row): probabilities[i] for i, row in enumerate(rows)},
        offsets={int(row): offsets[i] for i, row in enumerate(rows)},
        metrics=metrics,
    )


def _render_evidence(evidence: Sequence[str] | None) -> str:
    clean = [str(item).strip() for item in (evidence or []) if str(item).strip()]
    if not clean:
        return "No concise evidence phrase has been selected."
    return "\n".join(f"- {item}" for item in clean)


def make_risk_prompt(
    text: str,
    risk_id: int,
    evidence: Sequence[str] | None,
    config: Task1Full64Config,
) -> str:
    card = q38.RISK_CARDS[int(risk_id)]
    return (
        f"[SYSTEM]\n{q38.RISK_SYSTEM}\n[/SYSTEM]\n"
        f"[RISK CARD]\n{card.render()}\n[/RISK CARD]\n\n"
        f"[POST]\n{q38._compact_post(text, config.context_chars)}\n[/POST]\n\n"
        f"[SELECTED VERBATIM EVIDENCE]\n{_render_evidence(evidence)}\n"
        "[/SELECTED VERBATIM EVIDENCE]\n\n"
        "The evidence may be empty or imperfect. Use it as grounding, but reject a level "
        "that contradicts the complete post.\n"
        f"Question: Does the author support risk level {card.name}?\n"
        "A = YES\nB = NO\nAnswer:"
    )


def make_evidence_prompt(
    text: str,
    provisional_risk: int,
    candidate: str,
    config: Task1Full64Config,
) -> str:
    card = q38.RISK_CARDS[int(provisional_risk)]
    return (
        f"[SYSTEM]\n{q38.EVIDENCE_SYSTEM}\n[/SYSTEM]\n"
        f"[PROVISIONAL RISK CARD]\n{card.render()}\n[/PROVISIONAL RISK CARD]\n\n"
        f"[POST]\n{q38._compact_post(text, config.context_chars)}\n[/POST]\n\n"
        f"[EXACT CANDIDATE PHRASE]\n{candidate}\n[/EXACT CANDIDATE PHRASE]\n\n"
        "A valid phrase must be verbatim, directly relevant, and concise enough to submit.\n"
        "Question: Is this exact candidate a textual reason for the provisional risk level?\n"
        "A = SUPPORTS\nB = DOES NOT SUPPORT\nAnswer:"
    )


def _containing_positive_variant(
    text: str,
    gold: tuple[int, int],
    candidates: Sequence[q38.CandidateSpan],
) -> q38.CandidateSpan | None:
    gold_length = max(1, gold[1] - gold[0])
    eligible = [
        item for item in candidates
        if item.left <= gold[0] and item.right >= gold[1]
        and (item.right - item.left) <= 3 * gold_length
        and (item.left, item.right) != gold
    ]
    return min(eligible, key=lambda item: item.right - item.left) if eligible else None


def build_joint_manifest(
    bundle: b1.DataBundle,
    annotations: Sequence[inn.EvidenceAnnotation],
    train_rows: Sequence[int],
    lexicon: EvidenceLexicon,
    config: Task1Full64Config,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed + 1009 * config.fold)
    records: list[dict[str, Any]] = []
    for row in map(int, train_rows):
        text = bundle.texts[row]
        gold_risk = int(bundle.risk_ids[row])
        gold_phrases = [text[left:right].strip() for left, right in annotations[row].spans]
        use_evidence = bool(rng.random() < config.evidence_conditioned_training_fraction)
        risk_evidence = gold_phrases if use_evidence else []
        for risk_id in range(len(q38.RISK_CARDS)):
            records.append(
                {
                    "pair_id": f"risk::{row}::{risk_id}::{int(use_evidence)}",
                    "row_idx": row,
                    "task": "risk_conditioned" if use_evidence else "risk_document",
                    "target": int(risk_id == gold_risk),
                    "prompt": make_risk_prompt(text, risk_id, risk_evidence, config),
                }
            )

        candidates = generate_fold_safe_candidates(
            text, lexicon, config,
            max_candidates=max(48, config.evidence_negatives_per_post * 12),
        )
        for number, ((left, right), phrase) in enumerate(zip(annotations[row].spans, gold_phrases)):
            if not phrase:
                continue
            records.append(
                {
                    "pair_id": f"evidence_exact_positive::{row}::{number}",
                    "row_idx": row,
                    "task": "evidence_exact_positive",
                    "target": 1,
                    "prompt": make_evidence_prompt(text, gold_risk, phrase, config),
                }
            )
            if number < config.evidence_positive_variants_per_span:
                variant = _containing_positive_variant(text, (left, right), candidates)
                if variant is not None:
                    variant_text = text[variant.left:variant.right].strip()
                    records.append(
                        {
                            "pair_id": f"evidence_containing_positive::{row}::{number}",
                            "row_idx": row,
                            "task": "evidence_containing_positive",
                            "target": 1,
                            "prompt": make_evidence_prompt(text, gold_risk, variant_text, config),
                        }
                    )

        negatives = [
            item for item in candidates
            if not q38._overlaps_any((item.left, item.right), annotations[row].spans)
        ]
        count = min(config.evidence_negatives_per_post, len(negatives))
        selected = list(negatives[: max(0, count - 1)])
        remainder = negatives[len(selected):]
        if len(selected) < count and remainder:
            selected.append(remainder[int(rng.integers(len(remainder)))])
        for number, item in enumerate(selected):
            records.append(
                {
                    "pair_id": f"evidence_hard_negative::{row}::{number}",
                    "row_idx": row,
                    "task": "evidence_hard_negative",
                    "target": 0,
                    "prompt": make_evidence_prompt(
                        text, gold_risk, text[item.left:item.right].strip(), config
                    ),
                }
            )
    frame = pd.DataFrame(records)
    return frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)


def train_fold(
    bundle: b1.DataBundle,
    annotations: Sequence[inn.EvidenceAnnotation],
    folds: np.ndarray,
    config: Task1Full64Config,
    output_dir: str | Path,
    overwrite: bool = False,
) -> tuple[Path, EvidenceLexicon, pd.DataFrame]:
    output_dir = Path(output_dir)
    fold_dir = output_dir / f"fold_{config.fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_rows = np.flatnonzero(np.asarray(folds) != config.fold)
    lexicon_path = fold_dir / "evidence_lexicon.json"
    if lexicon_path.exists() and not overwrite:
        lexicon = EvidenceLexicon.from_json(json.loads(lexicon_path.read_text(encoding="utf-8")))
        if lexicon.training_rows != train_rows.astype(int).tolist():
            raise AssertionError("Saved evidence lexicon uses different training rows")
    else:
        lexicon = fit_evidence_lexicon(bundle, annotations, train_rows, config)
        b4.json_dump(lexicon.to_json(), lexicon_path)

    manifest_path = fold_dir / "risk_evidence_pair_manifest.csv"
    if manifest_path.exists() and not overwrite:
        manifest = pd.read_csv(manifest_path)
    else:
        manifest = build_joint_manifest(bundle, annotations, train_rows, lexicon, config)
        manifest.to_csv(manifest_path, index=False)
    b4.json_dump(
        {
            "runtime_revision": TASK1_RUNTIME_REVISION,
            "config": asdict(config),
            "risk_cards": [asdict(card) for card in q38.RISK_CARDS],
            "training_rows": train_rows.tolist(),
        },
        fold_dir / "frozen_task1_spec.json",
    )
    adapter = b4.train_verifier_adapter(
        q38.PromptABDataset(manifest), config.b4_config(), fold_dir / "adapter",
        overwrite=overwrite,
    )
    return adapter, lexicon, manifest


def build_risk_prompt_frame(
    bundle: b1.DataBundle,
    rows: Sequence[int],
    config: Task1Full64Config,
    evidence_by_row: dict[int, list[str]] | None,
    stage: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in map(int, rows):
        evidence = None if evidence_by_row is None else evidence_by_row.get(row, [])
        for risk_id in range(len(q38.RISK_CARDS)):
            records.append(
                {
                    "pair_id": f"risk-{stage}::{bundle.row_ids[row]}::{risk_id}",
                    "query_row_idx": row,
                    "risk_id": risk_id,
                    "prompt": make_risk_prompt(bundle.texts[row], risk_id, evidence, config),
                }
            )
    return pd.DataFrame(records)


def build_evidence_prompt_frame(
    bundle: b1.DataBundle,
    rows: Sequence[int],
    provisional_risk: np.ndarray,
    lexicon: EvidenceLexicon,
    config: Task1Full64Config,
    token_proposals: TokenProposalCache | None = None,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in map(int, rows):
        probability = None if token_proposals is None else token_proposals.probabilities.get(row)
        offsets = None if token_proposals is None else token_proposals.offsets.get(row)
        candidates = generate_fold_safe_candidates(
            bundle.texts[row], lexicon, config,
            token_probabilities=probability, token_offsets=offsets,
        )
        for number, item in enumerate(candidates):
            candidate = bundle.texts[row][item.left:item.right].strip()
            records.append(
                {
                    "pair_id": f"evidence-valid::{bundle.row_ids[row]}::{number}",
                    "query_row_idx": row,
                    "left": item.left,
                    "right": item.right,
                    "source": item.source,
                    "candidate": candidate,
                    "prompt": make_evidence_prompt(
                        bundle.texts[row], int(provisional_risk[row]), candidate, config
                    ),
                }
            )
    return pd.DataFrame(records)


def _risk_matrix(frame: pd.DataFrame, margins: np.ndarray, n_rows: int) -> np.ndarray:
    matrix = np.full((n_rows, len(q38.RISK_CARDS)), np.nan, dtype=np.float32)
    for row, risk_id, margin in zip(
        frame.query_row_idx.astype(int), frame.risk_id.astype(int), np.asarray(margins)
    ):
        matrix[row, risk_id] = margin
    return matrix


def select_evidence(
    frame: pd.DataFrame,
    margins: np.ndarray,
    config: Task1Full64Config,
) -> tuple[dict[int, list[str]], pd.DataFrame]:
    scored = frame.drop(columns=["prompt"], errors="ignore").copy()
    scored["margin"] = np.asarray(margins, dtype=np.float32)
    scored["selection_score"] = (
        scored.margin
        - config.evidence_length_penalty * scored.candidate.astype(str).str.len()
    )
    scored["selected"] = False
    output: dict[int, list[str]] = {}
    for row, group in scored.groupby("query_row_idx", sort=False):
        accepted = group[group.margin >= config.evidence_margin_threshold].sort_values(
            "selection_score", ascending=False
        )
        chosen: list[int] = []
        spans: list[tuple[int, int]] = []
        for index, item in accepted.iterrows():
            span = (int(item.left), int(item.right))
            if any(span[0] < right and span[1] > left for left, right in spans):
                continue
            chosen.append(index)
            spans.append(span)
            if len(chosen) >= config.evidence_top_k:
                break
        scored.loc[chosen, "selected"] = True
        output[int(row)] = scored.loc[chosen, "candidate"].astype(str).tolist()
    return output, scored


def _risk_metrics(
    gold: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    return {
        "weighted_f1": float(f1_score(gold, predicted, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(gold, predicted, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(gold, predicted, labels=np.arange(4)).tolist(),
    }


def evaluate_fold(
    bundle: b1.DataBundle,
    folds: np.ndarray,
    adapter_path: str | Path,
    lexicon: EvidenceLexicon,
    config: Task1Full64Config,
    output_dir: str | Path,
    token_proposals: TokenProposalCache | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_rows = np.flatnonzero(np.asarray(folds) == config.fold)
    b4_cfg = config.b4_config()
    model, tokenizer = b4.load_quantized_causal_model(
        config.model_name,
        adapter_path=adapter_path,
        training=False,
        attention_implementation=config.attention_implementation,
        qwen35_fa2_position_guard=config.qwen35_fa2_position_guard,
        require_qwen35_fast_kernels=config.require_qwen35_fast_kernels,
    )

    document_frame = build_risk_prompt_frame(
        bundle, valid_rows, config, evidence_by_row=None, stage="document"
    )
    document_margins = b4.score_prompts_cached(
        model, tokenizer, document_frame, output_dir / "risk_document_scores",
        b4_cfg, config.score_batch_size, use_chat_template=True,
    )
    document_matrix = _risk_matrix(document_frame, document_margins, len(bundle.texts))
    document_prediction = np.full(len(bundle.texts), -1, dtype=np.int64)
    document_prediction[valid_rows] = np.argmax(document_matrix[valid_rows], axis=1)

    evidence_frame = build_evidence_prompt_frame(
        bundle, valid_rows, document_prediction, lexicon, config,
        token_proposals=token_proposals,
    )
    ceiling = q38.candidate_recall_ceiling(evidence_frame, bundle, valid_rows)
    evidence_margins = b4.score_prompts_cached(
        model, tokenizer, evidence_frame, output_dir / "evidence_scores",
        b4_cfg, config.score_batch_size, use_chat_template=True,
    )
    selected, evidence_audit = select_evidence(evidence_frame, evidence_margins, config)

    conditioned_frame = build_risk_prompt_frame(
        bundle, valid_rows, config, evidence_by_row=selected, stage="conditioned"
    )
    conditioned_margins = b4.score_prompts_cached(
        model, tokenizer, conditioned_frame, output_dir / "risk_conditioned_scores",
        b4_cfg, config.score_batch_size, use_chat_template=True,
    )
    conditioned_matrix = _risk_matrix(conditioned_frame, conditioned_margins, len(bundle.texts))
    conditioned_prediction = np.full(len(bundle.texts), -1, dtype=np.int64)
    conditioned_prediction[valid_rows] = np.argmax(conditioned_matrix[valid_rows], axis=1)
    blended_matrix = (
        (1.0 - config.conditioned_risk_blend_weight) * document_matrix
        + config.conditioned_risk_blend_weight * conditioned_matrix
    )
    blended_prediction = np.full(len(bundle.texts), -1, dtype=np.int64)
    blended_prediction[valid_rows] = np.argmax(blended_matrix[valid_rows], axis=1)

    gold_risk = bundle.risk_ids[valid_rows]
    document_metrics = _risk_metrics(gold_risk, document_prediction[valid_rows])
    conditioned_metrics = _risk_metrics(gold_risk, conditioned_prediction[valid_rows])
    blended_metrics = _risk_metrics(gold_risk, blended_prediction[valid_rows])
    risk_variants = {
        "document_only": document_metrics,
        "evidence_conditioned": conditioned_metrics,
        "fixed_margin_blend": blended_metrics,
    }
    selected_risk_variant = max(
        risk_variants, key=lambda name: risk_variants[name]["weighted_f1"]
    )
    # The fixed blend is the predeclared primary result.  The best variant is
    # shown only as a screening diagnostic and may be frozen for Folds 1/2 in a
    # later decision; it must not silently inflate the Fold-0 gate.
    primary_risk_variant = "fixed_margin_blend"
    primary_risk_metrics = risk_variants[primary_risk_variant]

    gold_evidence_all = q38._gold_phrases(bundle)
    predicted_evidence = [selected.get(int(row), []) for row in valid_rows]
    gold_evidence = [gold_evidence_all[int(row)] for row in valid_rows]
    phrase = q38.official_like_phrase_f1(predicted_evidence, gold_evidence)
    raw_points = 0.4 * primary_risk_metrics["weighted_f1"] + 0.3 * phrase["f1"]
    metrics = {
        "runtime_revision": TASK1_RUNTIME_REVISION,
        "model": config.model_name,
        "fold": int(config.fold),
        "n_validation": int(len(valid_rows)),
        "risk_variants": risk_variants,
        "primary_risk_variant": primary_risk_variant,
        "screening_selected_risk_variant": selected_risk_variant,
        "risk_weighted_f1": primary_risk_metrics["weighted_f1"],
        "risk_macro_f1": primary_risk_metrics["macro_f1"],
        "evidence_phrase_precision_official_like": phrase["precision"],
        "evidence_phrase_recall_official_like": phrase["recall"],
        "evidence_phrase_f1_official_like": phrase["f1"],
        "candidate_recall_ceiling": float(ceiling),
        "mean_predicted_evidence_phrases": float(np.mean([len(items) for items in predicted_evidence])),
        "subtask1_raw_points_out_of_0_7": float(raw_points),
        "subtask1_normalized_score": float(raw_points / 0.7),
        "warning": (
            "The best risk variant is a Fold-0 screening choice. It must be fixed and "
            "confirmed on Folds 1/2 before final use. Phrase-F1 is official-like until "
            "the organizer scorer is available."
        ),
    }
    evidence_audit.to_csv(output_dir / "evidence_candidate_audit.csv", index=False)
    pd.DataFrame(
        {
            "row_id": bundle.row_ids[valid_rows],
            "gold_risk": [b1.RISK_LABELS[index] for index in gold_risk],
            "document_risk": [b1.RISK_LABELS[index] for index in document_prediction[valid_rows]],
            "conditioned_risk": [b1.RISK_LABELS[index] for index in conditioned_prediction[valid_rows]],
            "blended_risk": [b1.RISK_LABELS[index] for index in blended_prediction[valid_rows]],
            "predicted_evidence": [json.dumps(items, ensure_ascii=False) for items in predicted_evidence],
            "gold_evidence": [json.dumps(items, ensure_ascii=False) for items in gold_evidence],
        }
    ).to_csv(output_dir / "validation_predictions.csv", index=False)
    np.savez_compressed(
        output_dir / "q38_task1_fold_outputs.npz",
        row_ids=bundle.row_ids,
        folds=np.asarray(folds),
        valid_rows=valid_rows,
        document_risk_margins=document_matrix,
        conditioned_risk_margins=conditioned_matrix,
        blended_risk_margins=blended_matrix,
        document_risk_predictions=document_prediction,
        conditioned_risk_predictions=conditioned_prediction,
        blended_risk_predictions=blended_prediction,
    )
    b4.json_dump(metrics, output_dir / "q38_task1_fold_metrics.json")
    del model, tokenizer
    b4.unload_model()
    return metrics


def continuation_gate(
    challenger: dict[str, Any],
    paired_baseline: dict[str, Any] | None,
    online_reference: float = 0.7577,
) -> dict[str, Any]:
    paired_score = None
    paired_delta = None
    if paired_baseline:
        paired_score = paired_baseline.get("subtask1_normalized_score")
        if paired_score is not None:
            paired_delta = float(challenger["subtask1_normalized_score"] - paired_score)
    passed = bool(
        challenger["candidate_recall_ceiling"] >= 0.84
        and challenger["risk_weighted_f1"] >= 0.70
        and challenger["evidence_phrase_f1_official_like"] >= 0.45
        and (paired_delta is None or paired_delta >= 0.005)
    )
    return {
        "passed": passed,
        "decision": "RUN_TASK1_FOLDS_1_2" if passed else "STOP_AND_DIAGNOSE_TASK1",
        "paired_baseline_normalized_score": paired_score,
        "paired_delta_normalized_score": paired_delta,
        "online_reference_unpaired": float(online_reference),
        "rule": {
            "candidate_recall_ceiling_min": 0.84,
            "risk_weighted_f1_min": 0.70,
            "evidence_phrase_f1_min": 0.45,
            "paired_normalized_delta_min": 0.005,
        },
    }
