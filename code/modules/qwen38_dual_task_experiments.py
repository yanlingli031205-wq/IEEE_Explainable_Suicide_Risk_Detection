"""One-fold Qwen3.8-27B challenger for Lenormand Factor and Risk/Evidence.

The module deliberately reuses B4-P's A/B margin objective and resumable NF4
QLoRA trainer.  It contains no remote API calls.  Factor evaluation is directly
comparable with the existing Qwen3-14B fold; Task 1 reports both the official-
like phrase score and an explicit warning when only an online reference is
available.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import Dataset

import b1_experiments as b1
import b1_innovation_experiments as inn
import b4p_anchor_verifier as b4


QWEN38_MODEL = "Qwen/Qwen3.8-27B"
Q38_RUNTIME_REVISION = "2026-08-24.official-evidence-scorer-v3"


@dataclass(frozen=True)
class RiskCard:
    name: str
    definition: str
    present: str
    absent: str
    boundary: str

    def render(self) -> str:
        return (
            f"Risk level: {self.name}\n"
            f"Operational definition: {self.definition}\n"
            f"Count as this level when: {self.present}\n"
            f"Do not count as this level when: {self.absent}\n"
            f"Most important boundary: {self.boundary}"
        )


RISK_CARDS: tuple[RiskCard, ...] = (
    RiskCard(
        "Indicator",
        "Distress or vulnerability relevant to suicide-risk monitoring, without explicit self-directed suicidal ideation or suicidal action.",
        "the post shows clinically relevant distress, warning signs, or vulnerability but does not support a higher level",
        "the post explicitly supports suicidal thoughts, suicidal behavior, or an actual attempt",
        "Indirect distress is not automatically Ideation; require self-directed death or suicide content for the higher level.",
    ),
    RiskCard(
        "Ideation",
        "Self-directed thoughts, wishes, intent, or plans about death or suicide, without a reported enacted suicidal behavior or attempt.",
        "the author wants to die, thinks about suicide, expresses suicidal intent, or describes a plan as thought/intention",
        "the text only shows distress, or it reports an enacted suicidal behavior/attempt that belongs at a higher level",
        "A contemplated method is Ideation unless the post says the author has begun acting or has attempted.",
    ),
    RiskCard(
        "Behavior",
        "Enacted suicidal or self-harm behavior, preparation, or an interrupted/aborted act, short of a reported suicide attempt.",
        "the author reports concrete preparation, an initiated/interrupted act, or relevant self-harm behavior beyond thoughts alone",
        "the text contains only ideation/planning in thought, or clearly reports an actual suicide attempt",
        "Behavior requires an action; separate it from Ideation and from a completed non-fatal Attempt.",
    ),
    RiskCard(
        "Attempt",
        "A reported suicide attempt by the author: an act carried out with at least some intent to die that did not result in death.",
        "the author says they attempted suicide or describes carrying out a potentially lethal self-directed act with suicidal intent",
        "the text reports only thoughts, a plan, preparation, non-suicidal self-injury, or another person's attempt",
        "Do not infer Attempt from method access or strong intent without an enacted attempt.",
    ),
)


RISK_SYSTEM = (
    "You are a deterministic research annotation verifier for suicide-risk text. "
    "Use only the supplied operational card and the post. Do not diagnose, do not "
    "invent missing facts, do not explain, and output exactly A or B."
)

EVIDENCE_SYSTEM = (
    "You are a deterministic evidence verifier. Decide whether the exact candidate "
    "phrase, in its post context, is a concise textual reason for the supplied gold "
    "risk level. The candidate must be grounded in the post. Do not explain and "
    "output exactly A or B."
)


@dataclass
class Qwen38RiskEvidenceConfig:
    model_name: str = QWEN38_MODEL
    fold: int = 0
    max_length: int = 1536
    context_chars: int = 5200
    candidate_max_chars: int = 420
    validation_candidates_per_post: int = 112
    negative_candidates_per_train_post: int = 3
    evidence_top_k: int = 3
    evidence_margin_threshold: float = 0.0
    sft_epochs: float = 1.0
    sft_max_steps: int = -1
    learning_rate: float = 7.5e-5
    gradient_accumulation: int = 32
    lora_r: int = 16
    lora_alpha: int = 32
    lora_last_n_layers: int | None = 16
    lora_target_leaves: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
    )
    gradient_checkpointing: bool = False
    score_batch_size: int = 2
    seed: int = 42

    def b4_config(self) -> b4.B4PConfig:
        return b4.B4PConfig(
            seed=self.seed,
            n_splits=3,
            verifier_model=self.model_name,
            max_length=self.max_length,
            include_retrieval=False,
            sft_epochs=self.sft_epochs,
            sft_max_steps=self.sft_max_steps,
            sft_learning_rate=self.learning_rate,
            sft_batch_size=1,
            sft_gradient_accumulation=self.gradient_accumulation,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_last_n_layers=self.lora_last_n_layers,
            lora_target_leaves=self.lora_target_leaves,
            gradient_checkpointing=self.gradient_checkpointing,
            verifier_score_batch_size=self.score_batch_size,
            score_chunk_size=256,
        )


class PromptABDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        required = {"prompt", "target", "pair_id"}
        if not required.issubset(frame.columns):
            raise KeyError(f"Prompt frame is missing {sorted(required - set(frame.columns))}")
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        return {
            "prompt": str(row.prompt),
            "target": int(row.target),
            "pair_id": str(row.pair_id),
        }


@dataclass(frozen=True)
class CandidateSpan:
    left: int
    right: int
    source: str


_RISK_CUES = {
    "suicide", "suicidal", "die", "dying", "death", "kill", "killing", "dead",
    "attempt", "overdose", "jump", "hanging", "hang", "cut", "cutting", "gun",
    "pills", "bridge", "worthless", "hopeless", "goodbye", "burden", "end",
    "selfharm", "harm", "hurt", "hurting", "blade", "blades", "helium", "rope",
    "commit", "committed", "committing", "live", "alive", "living", "breathe",
    "breathing", "pain", "life", "plan", "planned", "planning", "tried", "try",
    "trying", "want", "wanted", "wanna", "need", "needed", "wish", "wished",
    "rather", "should", "method", "exit", "leave", "leaving", "go", "gone",
    "over", "soon", "tomorrow", "note", "letter", "scar", "scars",
}


def _compact_post(text: str, limit: int) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    half = max(1, (limit - 35) // 2)
    return text[:half] + "\n[... middle omitted ...]\n" + text[-half:]


def _trim_span(text: str, left: int, right: int) -> tuple[int, int] | None:
    left, right = max(0, int(left)), min(len(text), int(right))
    while left < right and text[left].isspace():
        left += 1
    while right > left and text[right - 1].isspace():
        right -= 1
    return (left, right) if right > left else None


def _word_spans(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0).lower(), m.start(), m.end()) for m in re.finditer(r"\b[\w']+\b", text)]


def generate_evidence_candidates(
    text: str,
    max_candidates: int = 64,
    max_chars: int = 420,
) -> list[CandidateSpan]:
    """Generate exact, verbatim clause/window candidates without using gold."""
    text = str(text or "")
    raw: list[CandidateSpan] = []
    clause_spans = b1.split_clauses(text, max_clauses=48)
    for left, right in clause_spans:
        if right - left <= max_chars:
            raw.append(CandidateSpan(left, right, "clause"))
        else:
            cursor = left
            while cursor < right:
                raw.append(CandidateSpan(cursor, min(right, cursor + max_chars), "clause_chunk"))
                cursor += max(120, max_chars - 80)

    words = _word_spans(text)
    for index, (word, _, _) in enumerate(words):
        if word not in _RISK_CUES:
            continue
        for radius in (7, 15):
            lo, hi = max(0, index - radius), min(len(words), index + radius + 1)
            raw.append(CandidateSpan(words[lo][1], words[hi - 1][2], f"cue_window_{radius}"))
        # Gold evidence in this task is usually only 3--8 tokens.  Short
        # cue-anchored n-grams keep the submitted phrase within the official
        # 3x length cap; sentence-only candidates have a poor oracle ceiling.
        for width in (1, 2, 3, 5, 8, 12):
            for cue_offset in (0, width // 2, width - 1):
                lo = max(0, index - cue_offset)
                hi = min(len(words), lo + width)
                lo = max(0, hi - width)
                raw.append(CandidateSpan(words[lo][1], words[hi - 1][2], f"cue_ngram_{width}"))

    # Add short adjacent-clause compositions; gold annotations sometimes cross
    # one punctuation boundary even when neither isolated clause is sufficient.
    for first, second in zip(clause_spans, clause_spans[1:]):
        if second[1] - first[0] <= max_chars:
            raw.append(CandidateSpan(first[0], second[1], "adjacent_clauses"))

    dedup: dict[str, CandidateSpan] = {}
    for item in raw:
        trimmed = _trim_span(text, item.left, item.right)
        if trimmed is None:
            continue
        left, right = trimmed
        phrase = text[left:right]
        normalized = b1.normalize_text(phrase)
        if len(normalized) < 4:
            continue
        current = dedup.get(normalized)
        if current is None or (right - left) < (current.right - current.left):
            dedup[normalized] = CandidateSpan(left, right, item.source)

    def priority(item: CandidateSpan) -> tuple[float, float, int]:
        phrase = text[item.left:item.right]
        tokens = [m.group(0).lower() for m in re.finditer(r"\b[\w']+\b", phrase)]
        cue_count = sum(token in _RISK_CUES for token in tokens)
        source_bonus = (
            3.0 if item.source.startswith("cue_ngram")
            else (1.0 if item.source.startswith("cue_window") else 0.0)
        )
        concise = -abs(len(phrase) - 120) / 420.0
        return (3.0 * cue_count + source_bonus + concise, -len(phrase), -item.left)

    candidates = sorted(dedup.values(), key=priority, reverse=True)
    if len(candidates) > max_candidates:
        # Preserve broad document coverage as well as high-cue candidates.
        top_n = max_candidates * 2 // 3
        chosen = candidates[:top_n]
        remainder = sorted(candidates[top_n:], key=lambda item: item.left)
        if remainder:
            positions = np.linspace(0, len(remainder) - 1, max_candidates - top_n).round().astype(int)
            chosen.extend(remainder[index] for index in positions)
        candidates = chosen
    return candidates


def make_risk_prompt(text: str, risk_id: int, context_chars: int = 5200) -> str:
    card = RISK_CARDS[int(risk_id)]
    return (
        f"[SYSTEM]\n{RISK_SYSTEM}\n[/SYSTEM]\n"
        f"[RISK CARD]\n{card.render()}\n[/RISK CARD]\n\n"
        f"[POST]\n{_compact_post(text, context_chars)}\n[/POST]\n\n"
        f"Question: Does the post support risk level {card.name}?\n"
        "A = YES\nB = NO\nAnswer:"
    )


def make_evidence_prompt(
    text: str,
    risk_id: int,
    candidate: str,
    context_chars: int = 5200,
) -> str:
    card = RISK_CARDS[int(risk_id)]
    return (
        f"[SYSTEM]\n{EVIDENCE_SYSTEM}\n[/SYSTEM]\n"
        f"[RISK CARD]\n{card.render()}\n[/RISK CARD]\n\n"
        f"[POST]\n{_compact_post(text, context_chars)}\n[/POST]\n\n"
        f"[EXACT CANDIDATE PHRASE]\n{candidate}\n[/EXACT CANDIDATE PHRASE]\n\n"
        "Question: Is this candidate a concise, sufficient textual reason for assigning the supplied risk level?\n"
        "A = SUPPORTS\nB = DOES NOT SUPPORT\nAnswer:"
    )


def _overlaps_any(span: tuple[int, int], gold: Sequence[tuple[int, int]]) -> bool:
    return any(span[0] < right and span[1] > left for left, right in gold)


def build_risk_evidence_manifest(
    bundle: b1.DataBundle,
    annotations: Sequence[inn.EvidenceAnnotation],
    train_indices: Sequence[int],
    config: Qwen38RiskEvidenceConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed + config.fold)
    records: list[dict[str, Any]] = []
    for row in map(int, train_indices):
        text, gold_risk = bundle.texts[row], int(bundle.risk_ids[row])
        for risk_id in range(len(RISK_CARDS)):
            records.append(
                {
                    "pair_id": f"risk::{row}::{risk_id}",
                    "row_idx": row,
                    "task": "risk",
                    "target": int(risk_id == gold_risk),
                    "prompt": make_risk_prompt(text, risk_id, config.context_chars),
                }
            )

        annotation = annotations[row]
        for number, (left, right) in enumerate(annotation.spans):
            candidate = text[left:right].strip()
            if candidate:
                records.append(
                    {
                        "pair_id": f"evidence-positive::{row}::{number}",
                        "row_idx": row,
                        "task": "evidence",
                        "target": 1,
                        "prompt": make_evidence_prompt(text, gold_risk, candidate, config.context_chars),
                    }
                )

        candidates = generate_evidence_candidates(
            text,
            max_candidates=max(20, config.negative_candidates_per_train_post * 6),
            max_chars=config.candidate_max_chars,
        )
        negatives = [
            candidate for candidate in candidates
            if not _overlaps_any((candidate.left, candidate.right), annotation.spans)
        ]
        if negatives:
            count = min(config.negative_candidates_per_train_post, len(negatives))
            # Candidate order already prioritizes risk-like clauses.  Mix the
            # strongest within-post hard negatives with one random negative.
            selected = negatives[: max(1, count - 1)]
            remaining = [item for item in negatives if item not in selected]
            if len(selected) < count and remaining:
                selected.append(remaining[int(rng.integers(len(remaining)))])
            for number, candidate_span in enumerate(selected):
                candidate = text[candidate_span.left:candidate_span.right].strip()
                records.append(
                    {
                        "pair_id": f"evidence-negative::{row}::{number}",
                        "row_idx": row,
                        "task": "evidence",
                        "target": 0,
                        "prompt": make_evidence_prompt(text, gold_risk, candidate, config.context_chars),
                    }
                )
    frame = pd.DataFrame(records)
    frame = frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)
    return frame


def train_risk_evidence_adapter(
    bundle: b1.DataBundle,
    annotations: Sequence[inn.EvidenceAnnotation],
    folds: np.ndarray,
    config: Qwen38RiskEvidenceConfig,
    output_dir: str | Path,
    overwrite: bool = False,
) -> tuple[Path, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    b4.json_dump(
        {
            "config": asdict(config),
            "risk_cards": [asdict(card) for card in RISK_CARDS],
            "risk_system": RISK_SYSTEM,
            "evidence_system": EVIDENCE_SYSTEM,
        },
        output_dir / "frozen_task1_spec.json",
    )
    manifest_path = output_dir / "risk_evidence_pair_manifest.csv"
    train_indices = np.flatnonzero(np.asarray(folds) != config.fold)
    if manifest_path.exists() and not overwrite:
        manifest = pd.read_csv(manifest_path)
    else:
        manifest = build_risk_evidence_manifest(bundle, annotations, train_indices, config)
        manifest.to_csv(manifest_path, index=False)
    adapter = b4.train_verifier_adapter(
        PromptABDataset(manifest), config.b4_config(), output_dir / "adapter", overwrite=overwrite
    )
    return adapter, manifest


def build_risk_validation_prompts(
    bundle: b1.DataBundle,
    rows: Sequence[int],
    config: Qwen38RiskEvidenceConfig,
) -> pd.DataFrame:
    records = []
    for row in map(int, rows):
        for risk_id in range(len(RISK_CARDS)):
            records.append(
                {
                    "pair_id": f"risk-valid::{bundle.row_ids[row]}::{risk_id}",
                    "query_row_idx": row,
                    "risk_id": risk_id,
                    "prompt": make_risk_prompt(bundle.texts[row], risk_id, config.context_chars),
                }
            )
    return pd.DataFrame(records)


def build_evidence_validation_prompts(
    bundle: b1.DataBundle,
    rows: Sequence[int],
    predicted_risks: np.ndarray,
    config: Qwen38RiskEvidenceConfig,
) -> pd.DataFrame:
    records = []
    for row in map(int, rows):
        candidates = generate_evidence_candidates(
            bundle.texts[row],
            max_candidates=config.validation_candidates_per_post,
            max_chars=config.candidate_max_chars,
        )
        for number, candidate in enumerate(candidates):
            phrase = bundle.texts[row][candidate.left:candidate.right].strip()
            records.append(
                {
                    "pair_id": f"evidence-valid::{bundle.row_ids[row]}::{number}",
                    "query_row_idx": row,
                    "left": candidate.left,
                    "right": candidate.right,
                    "source": candidate.source,
                    "candidate": phrase,
                    "prompt": make_evidence_prompt(
                        bundle.texts[row], int(predicted_risks[row]), phrase, config.context_chars
                    ),
                }
            )
    return pd.DataFrame(records)


def _risk_matrix(prompt_frame: pd.DataFrame, margins: np.ndarray, n_rows: int) -> np.ndarray:
    matrix = np.full((n_rows, len(RISK_CARDS)), np.nan, dtype=np.float32)
    for row, risk_id, score in zip(
        prompt_frame.query_row_idx.astype(int),
        prompt_frame.risk_id.astype(int),
        np.asarray(margins, dtype=np.float32),
    ):
        matrix[row, risk_id] = score
    return matrix


def select_evidence_phrases(
    frame: pd.DataFrame,
    margins: np.ndarray,
    config: Qwen38RiskEvidenceConfig,
) -> tuple[dict[int, list[str]], pd.DataFrame]:
    scored = frame.drop(columns=["prompt"], errors="ignore").copy()
    scored["margin"] = np.asarray(margins, dtype=np.float32)
    scored["selection_score"] = scored["margin"] - 0.0015 * scored["candidate"].astype(str).str.len()
    selected_by_row: dict[int, list[str]] = {}
    scored["selected"] = False
    for row, group in scored.groupby("query_row_idx", sort=False):
        ranked = group.sort_values("selection_score", ascending=False)
        accepted = ranked[ranked.margin >= config.evidence_margin_threshold]
        chosen_indices: list[int] = []
        chosen_spans: list[tuple[int, int]] = []
        for index, item in accepted.iterrows():
            span = (int(item.left), int(item.right))
            # Do not submit nested/near-duplicate phrases from the same clause.
            if any(span[0] < right and span[1] > left for left, right in chosen_spans):
                continue
            chosen_indices.append(index)
            chosen_spans.append(span)
            if len(chosen_indices) >= config.evidence_top_k:
                break
        scored.loc[chosen_indices, "selected"] = True
        selected_by_row[int(row)] = scored.loc[chosen_indices, "candidate"].astype(str).tolist()
    return selected_by_row, scored


def official_like_phrase_f1(
    predicted: Sequence[Sequence[str]],
    gold: Sequence[Sequence[str]],
) -> dict[str, float]:
    """Published Subtask-1b Phrase-F1, with extra micro diagnostics.

    The competition page specifies all of the following:

    * case-insensitive normalization;
    * ``pred in gold`` **or** ``gold in pred``;
    * predicted token length no greater than three times the matched Gold;
    * one-to-one predicted/Gold matching; and
    * final Precision, Recall and F1 averaged over posts.

    The organizers have published the rule but not executable scorer code, so
    the historical function name is retained for API compatibility.  Empty /
    empty posts receive 1, matching the supplied Task-1b notebook.
    """
    if len(predicted) != len(gold):
        raise ValueError(f"predicted/gold length mismatch: {len(predicted)} != {len(gold)}")

    def normalize(value: Any) -> str:
        return " ".join(str(value).strip().casefold().split())

    def token_length(value: str) -> int:
        return len(value.split())

    def is_match(prediction: str, target: str) -> bool:
        if not prediction or not target:
            return False
        if token_length(prediction) > 3 * max(1, token_length(target)):
            return False
        return prediction in target or target in prediction

    def maximum_matches(predictions: Sequence[str], targets: Sequence[str]) -> int:
        adjacency = [
            [g_idx for g_idx, target in enumerate(targets) if is_match(prediction, target)]
            for prediction in predictions
        ]
        gold_owner = [-1] * len(targets)

        def augment(pred_idx: int, seen_gold: set[int]) -> bool:
            for gold_idx in adjacency[pred_idx]:
                if gold_idx in seen_gold:
                    continue
                seen_gold.add(gold_idx)
                owner = gold_owner[gold_idx]
                if owner < 0 or augment(owner, seen_gold):
                    gold_owner[gold_idx] = pred_idx
                    return True
            return False

        return sum(augment(pred_idx, set()) for pred_idx in range(len(predictions)))

    per_post_precision: list[float] = []
    per_post_recall: list[float] = []
    per_post_f1: list[float] = []
    tp = fp = fn = empty_empty = 0
    for predicted_row, gold_row in zip(predicted, gold):
        pred_norm = [normalize(item) for item in predicted_row]
        pred_norm = [item for item in pred_norm if item and item != "none"]
        gold_norm = [normalize(item) for item in gold_row]
        gold_norm = [item for item in gold_norm if item and item != "none"]

        if not pred_norm and not gold_norm:
            precision = recall = f1 = 1.0
            matched = 0
            empty_empty += 1
        elif not pred_norm or not gold_norm:
            precision = recall = f1 = 0.0
            matched = 0
        else:
            matched = maximum_matches(pred_norm, gold_norm)
            precision = matched / len(pred_norm)
            recall = matched / len(gold_norm)
            f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

        per_post_precision.append(float(precision))
        per_post_recall.append(float(recall))
        per_post_f1.append(float(f1))
        tp += matched
        fp += len(pred_norm) - matched
        fn += len(gold_norm) - matched

    micro_precision = tp / max(1, tp + fp)
    micro_recall = tp / max(1, tp + fn)
    micro_f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    return {
        "precision": float(np.mean(per_post_precision)) if per_post_precision else 0.0,
        "recall": float(np.mean(per_post_recall)) if per_post_recall else 0.0,
        "f1": float(np.mean(per_post_f1)) if per_post_f1 else 0.0,
        "micro_precision_diagnostic": float(micro_precision),
        "micro_recall_diagnostic": float(micro_recall),
        "micro_f1_diagnostic": float(micro_f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "n_posts": int(len(per_post_f1)),
        "empty_empty_posts": int(empty_empty),
    }


def _gold_phrases(bundle: b1.DataBundle) -> list[list[str]]:
    column = b1._find_column(
        bundle.frame,
        ["evidence for suicide risk level", "evidence", "risk_evidence"],
    )
    return [inn.parse_evidence_phrases(value) for value in bundle.frame[column].tolist()]


def candidate_recall_ceiling(
    frame: pd.DataFrame,
    bundle: b1.DataBundle,
    rows: Sequence[int],
) -> float:
    predicted = []
    gold_all = _gold_phrases(bundle)
    for row in map(int, rows):
        predicted.append(frame.loc[frame.query_row_idx == row, "candidate"].astype(str).tolist())
    gold = [gold_all[int(row)] for row in rows]
    return official_like_phrase_f1(predicted, gold)["recall"]


def evaluate_risk_evidence_fold(
    bundle: b1.DataBundle,
    folds: np.ndarray,
    adapter_path: str | Path,
    config: Qwen38RiskEvidenceConfig,
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_rows = np.flatnonzero(np.asarray(folds) == config.fold)
    model, tokenizer = b4.load_quantized_causal_model(
        config.model_name, adapter_path=adapter_path, training=False
    )
    b4_cfg = config.b4_config()
    risk_prompts = build_risk_validation_prompts(bundle, valid_rows, config)
    risk_margins = b4.score_prompts_cached(
        model, tokenizer, risk_prompts, output_dir / "risk_scores", b4_cfg,
        config.score_batch_size, use_chat_template=True,
    )
    risk_matrix = _risk_matrix(risk_prompts, risk_margins, len(bundle.texts))
    risk_predictions = np.full(len(bundle.texts), -1, dtype=np.int64)
    risk_predictions[valid_rows] = np.nanargmax(risk_matrix[valid_rows], axis=1)
    risk_wf1 = float(
        f1_score(bundle.risk_ids[valid_rows], risk_predictions[valid_rows], average="weighted", zero_division=0)
    )
    risk_macro = float(
        f1_score(bundle.risk_ids[valid_rows], risk_predictions[valid_rows], average="macro", zero_division=0)
    )

    evidence_prompts = build_evidence_validation_prompts(
        bundle, valid_rows, risk_predictions, config
    )
    ceiling = candidate_recall_ceiling(evidence_prompts, bundle, valid_rows)
    evidence_margins = b4.score_prompts_cached(
        model, tokenizer, evidence_prompts, output_dir / "evidence_scores", b4_cfg,
        config.score_batch_size, use_chat_template=True,
    )
    selected, scored = select_evidence_phrases(evidence_prompts, evidence_margins, config)
    gold_all = _gold_phrases(bundle)
    predicted_rows = [selected.get(int(row), []) for row in valid_rows]
    gold_rows = [gold_all[int(row)] for row in valid_rows]
    phrase = official_like_phrase_f1(predicted_rows, gold_rows)
    raw_points = 0.4 * risk_wf1 + 0.3 * phrase["f1"]
    normalized_score = raw_points / 0.7
    metrics = {
        "model": config.model_name,
        "fold": config.fold,
        "n_validation": len(valid_rows),
        "risk_weighted_f1": risk_wf1,
        "risk_macro_f1": risk_macro,
        "evidence_phrase_precision_official_like": phrase["precision"],
        "evidence_phrase_recall_official_like": phrase["recall"],
        "evidence_phrase_f1_official_like": phrase["f1"],
        "candidate_recall_ceiling": ceiling,
        "subtask1_raw_points_out_of_0_7": raw_points,
        "subtask1_normalized_score": normalized_score,
        "mean_predicted_evidence_phrases": float(np.mean([len(row) for row in predicted_rows])),
        "warning": "Fold-local challenger score; online 0.7577 is an external reference, not a paired test.",
    }
    scored.to_csv(output_dir / "evidence_candidate_audit.csv", index=False)
    pd.DataFrame(
        {
            "row_id": bundle.row_ids[valid_rows],
            "gold_risk": [b1.RISK_LABELS[index] for index in bundle.risk_ids[valid_rows]],
            "predicted_risk": [b1.RISK_LABELS[index] for index in risk_predictions[valid_rows]],
            "predicted_evidence": [json.dumps(items, ensure_ascii=False) for items in predicted_rows],
            "gold_evidence": [json.dumps(items, ensure_ascii=False) for items in gold_rows],
        }
    ).to_csv(output_dir / "validation_predictions.csv", index=False)
    np.savez_compressed(
        output_dir / "qwen38_task1_fold_logits.npz",
        row_ids=bundle.row_ids,
        folds=np.asarray(folds),
        valid_rows=valid_rows,
        risk_margins=risk_matrix,
        risk_predictions=risk_predictions,
    )
    b4.json_dump(metrics, output_dir / "qwen38_task1_fold_metrics.json")
    del model, tokenizer
    b4.unload_model()
    return metrics


def run_modernbert_same_fold_proxy(
    bundle: b1.DataBundle,
    annotations: Sequence[inn.EvidenceAnnotation],
    folds: np.ndarray,
    fold: int,
    output_dir: str | Path,
    epochs: int = 2,
) -> dict[str, Any]:
    """Reproducible same-fold encoder baseline; not the private online model."""
    config = inn.RiskEvidenceConfig(
        name="Q38_TASK1_MODERNBERT_PROXY",
        model_name="answerdotai/ModernBERT-base",
        max_length=512,
        epochs=epochs,
        n_splits=len(np.unique(folds)),
        use_ordinal=True,
        use_evidence=True,
        condition_risk_on_evidence=True,
        use_counterfactual=True,
        save_checkpoints=False,
        resume=False,
    )
    result = inn.train_risk_evidence_fold(
        bundle, annotations, folds, fold, config, Path(output_dir)
    )
    valid_rows = np.asarray(result["idx"], dtype=int)
    probabilities = inn.risk_probabilities(
        result["risk_logits"], result["ordinal_logits"], config.ordinal_blend
    )
    predictions = probabilities.argmax(axis=1)
    predicted_phrases: list[list[str]] = []
    for row in valid_rows:
        index = int(row)
        spans = inn._token_predictions_to_spans(
            result["token_probabilities"][index], result["token_offsets"][index], 0.5
        )
        predicted_phrases.append([bundle.texts[index][left:right] for left, right in spans])
    gold_all = _gold_phrases(bundle)
    phrase = official_like_phrase_f1(
        predicted_phrases, [gold_all[int(row)] for row in valid_rows]
    )
    risk_wf1 = float(
        f1_score(bundle.risk_ids[valid_rows], predictions, average="weighted", zero_division=0)
    )
    raw_points = 0.4 * risk_wf1 + 0.3 * phrase["f1"]
    metrics = {
        "model": config.model_name,
        "fold": int(fold),
        "risk_weighted_f1": risk_wf1,
        "risk_macro_f1": float(
            f1_score(bundle.risk_ids[valid_rows], predictions, average="macro", zero_division=0)
        ),
        "evidence_token_ap": float(result["evidence_token_ap"]),
        "evidence_phrase_f1_official_like_fixed_0_5": phrase["f1"],
        "subtask1_raw_points_out_of_0_7": raw_points,
        "subtask1_normalized_score": raw_points / 0.7,
        "warning": "Same-fold ModernBERT proxy, not Lenormand's private/online BERT baseline.",
    }
    b4.json_dump(metrics, Path(output_dir) / "modernbert_same_fold_metrics.json")
    return metrics


def factor_fold_metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    probabilities = b4.sigmoid(logits)
    predictions = logits >= 0
    support = targets.sum(axis=0)
    tail = support < 60
    ap = np.asarray([
        average_precision_score(targets[:, label], probabilities[:, label])
        if np.unique(targets[:, label]).size == 2 else np.nan
        for label in range(targets.shape[1])
    ])
    per_f1 = np.asarray([
        f1_score(targets[:, label], predictions[:, label], zero_division=0)
        for label in range(targets.shape[1])
    ])
    return {
        "macro_ap": float(np.nanmean(ap)),
        "tail_macro_ap": float(np.nanmean(ap[tail])),
        "zero_margin_macro_f1": float(np.mean(per_f1)),
        "zero_margin_tail_macro_f1": float(np.mean(per_f1[tail])),
    }


def load_existing_factor_fold(
    path: str | Path,
    bundle: b1.DataBundle,
    fold: int,
) -> dict[str, float]:
    saved = np.load(path, allow_pickle=True)
    row_ids = saved["row_ids"].astype(str)
    if row_ids.tolist() != bundle.row_ids.astype(str).tolist():
        raise AssertionError("Existing Factor OOF row order does not match training data")
    folds = saved["folds"].astype(int)
    logits = saved["verifier_logits"] if "verifier_logits" in saved else saved["logits"]
    valid = folds == fold
    return factor_fold_metrics(logits[valid], bundle.factor_binary[valid])


def run_qwen38_factor_fold(
    bundle: b1.DataBundle,
    folds: np.ndarray,
    semantic_cache: b4.SemanticCache,
    config: b4.B4PConfig,
    fold: int,
    output_dir: str | Path,
    overwrite: bool = False,
) -> tuple[dict[str, float], Path]:
    output_dir = Path(output_dir)
    adapter = b4.train_one_outer_fold(
        bundle, folds, semantic_cache, fold, config, output_dir, overwrite=overwrite
    )
    corpus = b4.training_corpus(bundle)
    valid_rows = np.flatnonzero(np.asarray(folds) == fold)
    train_rows = np.flatnonzero(np.asarray(folds) != fold)
    matrix, _ = b4.run_verifier_scoring(
        adapter, corpus, semantic_cache, corpus, bundle, semantic_cache, train_rows,
        config, output_dir / f"fold_{fold}" / "validation", query_rows=valid_rows,
        query_is_training_corpus=True,
    )
    metrics = factor_fold_metrics(matrix[valid_rows], bundle.factor_binary[valid_rows])
    metrics.update({"model": config.verifier_model, "fold": fold, "n_validation": len(valid_rows)})
    np.savez_compressed(
        output_dir / f"fold_{fold}" / "qwen38_factor_fold_logits.npz",
        row_ids=bundle.row_ids,
        folds=np.asarray(folds),
        logits=matrix,
        targets=bundle.factor_binary,
    )
    b4.json_dump(metrics, output_dir / f"fold_{fold}" / "qwen38_factor_fold_metrics.json")
    return metrics, adapter


def factor_continuation_gate(
    qwen38: dict[str, float],
    qwen14: dict[str, float],
) -> dict[str, Any]:
    # A 27B run must buy more than noise because it costs materially more.
    passed = (
        qwen38["macro_ap"] >= qwen14["macro_ap"] + 0.010
        and qwen38["zero_margin_macro_f1"] >= 0.610
        and qwen38["tail_macro_ap"] >= qwen14["tail_macro_ap"] - 0.015
    )
    return {
        "passed": bool(passed),
        "decision": "RUN_REMAINING_FACTOR_FOLDS" if passed else "KEEP_QWEN3_14B_FACTOR",
        "delta_macro_ap": qwen38["macro_ap"] - qwen14["macro_ap"],
        "delta_zero_margin_macro_f1": qwen38["zero_margin_macro_f1"] - qwen14["zero_margin_macro_f1"],
        "delta_tail_macro_ap": qwen38["tail_macro_ap"] - qwen14["tail_macro_ap"],
        "rule": "AP +0.010, zero-margin Macro-F1 >=0.610, tail AP degradation <=0.015",
    }


def task1_continuation_gate(
    qwen38: dict[str, Any],
    local_baseline: dict[str, Any] | None,
    online_reference: float = 0.7577,
) -> dict[str, Any]:
    local_delta = None
    if local_baseline is not None:
        local_delta = qwen38["subtask1_normalized_score"] - local_baseline["subtask1_normalized_score"]
    passed = (
        qwen38["risk_weighted_f1"] >= 0.76
        and qwen38["evidence_phrase_f1_official_like"] >= 0.55
        and qwen38["candidate_recall_ceiling"] >= 0.80
        and (local_delta is None or local_delta > 0.01)
    )
    return {
        "passed": bool(passed),
        "decision": "RUN_REMAINING_TASK1_FOLDS" if passed else "KEEP_CURRENT_TASK1_AND_REVISE_CANDIDATES",
        "fold_local_score": qwen38["subtask1_normalized_score"],
        "local_paired_delta": local_delta,
        "online_reference_unpaired": online_reference,
        "warning": "Do not subtract the online reference from a CV fold and call it an improvement.",
        "rule": "risk WF1>=.76, phrase F1>=.55, candidate ceiling>=.80, paired local delta>.01",
    }
