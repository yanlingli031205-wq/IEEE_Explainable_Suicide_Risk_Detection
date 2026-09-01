"""Lenormand B4-P: performance-first Anchor--Verifier--Calibrator harness.

The factor task is transformed into 24 label-conditioned binary verification
problems.  One shared QLoRA adapter learns all factors; a frozen local MoE
anchor scores the same A/B questions.  All retrieval, prompts, calibration and
thresholds are fold safe.

This module intentionally keeps model loading lazy so data/prompt/calibration
tests can run on CPU without installing the full Colab GPU stack.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import gc
import hashlib
import inspect
import json
import math
import os
import random
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
from torch.utils.data import Dataset

import b1_experiments as b1


FACTOR_LABELS = tuple(b1.FACTOR_LABELS)
LABEL_TO_ID = {name: index for index, name in enumerate(FACTOR_LABELS)}
B4P_RUNTIME_REVISION = "2026-08-21.qwen38-full64-kernels-v4"


@dataclass(frozen=True)
class LabelCard:
    factor_id: str
    name: str
    definition: str
    include: str
    exclude: str
    confusions: tuple[str, ...] = ()

    def render(self) -> str:
        confusions = "; ".join(self.confusions) if self.confusions else "none specified"
        return (
            f"Factor ID: {self.factor_id}\n"
            f"Factor name: {self.name}\n"
            f"Operational definition: {self.definition}\n"
            f"Count as PRESENT when: {self.include}\n"
            f"Count as ABSENT when: {self.exclude}\n"
            f"Common boundary confusions: {confusions}"
        )


def _card(
    index: int,
    name: str,
    definition: str,
    include: str,
    exclude: str,
    *confusions: str,
) -> LabelCard:
    return LabelCard(f"F{index:02d}", name, definition, include, exclude, tuple(confusions))


# These cards operationalize the supplied taxonomy for annotation decisions.
# They are not diagnostic criteria and should be frozen before outer-CV runs.
LABEL_CARDS: dict[str, LabelCard] = {
    "cognitive deficits": _card(
        0, "cognitive deficits",
        "Difficulty with concentration, memory, judgement, clear thinking, planning, or problem solving.",
        "the author directly reports impaired cognitive functioning or inability to think clearly",
        "the post is merely confused in writing, uncertain about the future, or emotionally overwhelmed without a cognitive difficulty",
        "mental health issues", "emotion dysregulation",
    ),
    "coping strategy": _card(
        1, "coping strategy",
        "An intentional behaviour, activity, or mental strategy used to manage distress.",
        "the author describes something they do or deliberately try in order to tolerate, reduce, distract from, or handle distress",
        "an activity is only mentioned, is harmful without a coping function, or is advice not used by the author",
        "social support", "psychological capital", "substance use",
    ),
    "dysfunctional family": _card(
        2, "dysfunctional family",
        "Harmful family functioning involving abuse, neglect, rejection, chronic conflict, control, or serious instability.",
        "the author describes a sustained or serious harmful family relationship or environment",
        "there is only an ordinary temporary disagreement, a non-family relationship, or family structure without dysfunction",
        "interpersonal difficulty", "interpersonal violence", "poor social support",
    ),
    "emotion dysregulation": _card(
        3, "emotion dysregulation",
        "Intense, rapidly shifting, poorly controlled, or behaviourally overwhelming emotional reactions.",
        "the author cannot regulate affect, has explosive or extreme reactions, or emotion drives loss of control",
        "the post expresses sadness, fear, anger, or hopelessness without evidence of regulation difficulty",
        "hopelessness", "mental health issues", "low self-esteem",
    ),
    "exposure to others' suicide": _card(
        4, "exposure to others' suicide",
        "Exposure to another person's suicide, suicide attempt, suicidal ideation, or suicidal crisis.",
        "someone other than the author died by suicide, attempted suicide, or expressed suicidal behaviour",
        "the text concerns only the author's own suicidality or another person's non-suicidal death",
        "prior self-harm or suicidal thought/attempt", "stressful life event",
    ),
    "hopelessness": _card(
        5, "hopelessness",
        "A negative expectation that the future will not improve or that no solution or escape exists.",
        "the author expresses no way out, no future improvement, permanent defeat, helplessness, or futility",
        "the author is distressed, sad, or suicidal but does not express a negative future expectation or lack of solution",
        "low self-esteem", "emotion dysregulation", "meaning in life",
    ),
    "interpersonal difficulty": _card(
        6, "interpersonal difficulty",
        "Persistent or substantial difficulty forming, maintaining, understanding, or navigating relationships.",
        "the author reports serious recurring problems interacting with peers, friends, partners, or other people",
        "there is a single conflict adequately captured by another concrete event, or only loneliness/lack of support",
        "poor social support", "dysfunctional family", "stressful life event",
    ),
    "interpersonal violence": _card(
        7, "interpersonal violence",
        "Physical or sexual violence, assault, coercive physical harm, or a serious violent interpersonal threat.",
        "another person physically or sexually harmed, attacked, assaulted, or violently threatened the author",
        "the harm is only verbal conflict, emotional invalidation, self-harm, or a non-interpersonal accident",
        "traumatic experience", "dysfunctional family", "stressful life event",
    ),
    "low self-esteem": _card(
        8, "low self-esteem",
        "A negative evaluation of the author's own worth, adequacy, competence, or value.",
        "the author calls themselves worthless, useless, inadequate, a failure, unlovable, or fundamentally bad",
        "the author reports sadness, guilt about one act, social rejection, or hopelessness without negative self-worth",
        "hopelessness", "cognitive deficits", "mental health issues",
    ),
    "low socio-economic status": _card(
        9, "low socio-economic status",
        "Insufficient income, employment, housing, or basic material resources.",
        "the author reports poverty, homelessness, unemployment with deprivation, insecure housing, debt crisis, or inability to afford needs",
        "money is mentioned without deprivation, or job/school dissatisfaction has no material-resource consequence",
        "stressful life event", "poor school performance",
    ),
    "meaning in life": _card(
        10, "meaning in life",
        "A stated purpose, valued direction, significance, or reason that makes life worth continuing.",
        "the author identifies a purpose, valued role, spiritual meaning, or reason for living",
        "the author merely enjoys an activity, feels responsible, receives support, or denies meaning without a positive source of purpose",
        "sense of responsibility", "psychological capital", "hopelessness",
    ),
    "mental health issues": _card(
        11, "mental health issues",
        "A named mental disorder or substantial psychiatric symptoms affecting the author.",
        "the author reports a diagnosis, treatment, or clinically meaningful symptoms such as depression, anxiety, psychosis, panic, or disordered eating",
        "the post contains ordinary distress, one emotion, another person's condition, or informal self-description without substantial symptoms",
        "emotion dysregulation", "cognitive deficits", "physical health/characteristic",
    ),
    "physical health/characteristic": _card(
        12, "physical health/characteristic",
        "A physical illness, disability, chronic pain, sleep or bodily problem, or relevant physical characteristic affecting the author.",
        "the author reports their own physical condition, impairment, pain, disability, or bodily characteristic as relevant context",
        "the condition belongs only to another person or the post contains purely psychiatric symptoms",
        "mental health issues", "stressful life event",
    ),
    "poor school performance": _card(
        13, "poor school performance",
        "Failure, poor grades, inability to complete work, or serious underperformance in school or college.",
        "the author reports failing classes, low grades, missed academic requirements, or inability to study/complete coursework",
        "school is stressful, disliked, closed, or socially difficult without academic underperformance",
        "cognitive deficits", "stressful life event", "low socio-economic status",
    ),
    "poor social support": _card(
        14, "poor social support",
        "Needed emotional, informational, practical, professional, or emergency support is absent, unreliable, invalidating, or rejected.",
        "the author needs support but helpers are unavailable, dismissive, unreliable, inaccessible, or the author cannot accept available help",
        "the author is simply alone by choice, has relationship difficulty without a support need, or actually receives meaningful help",
        "social support", "interpersonal difficulty", "dysfunctional family",
    ),
    "prior self-harm or suicidal thought/attempt": _card(
        15, "prior self-harm or suicidal thought/attempt",
        "A history before the current moment of self-harm, suicidal thoughts, suicidal behaviour, or suicide attempt by the author.",
        "the author describes earlier self-harm, earlier suicidal ideation, or a previous suicide attempt",
        "the text only expresses current suicidality, a future plan, non-suicidal injury, or another person's history",
        "suicide means (with access)", "exposure to others' suicide",
    ),
    "psychological capital": _card(
        16, "psychological capital",
        "Positive psychological resources: hope, optimism, resilience, self-efficacy, or confidence in recovery.",
        "the author expects improvement, believes they can cope/recover, persists with hope, or expresses personal efficacy",
        "the author only receives support, uses a coping activity, or wishes vaguely for change without positive expectancy or efficacy",
        "coping strategy", "social support", "meaning in life",
    ),
    "sense of responsibility": _card(
        17, "sense of responsibility",
        "A felt duty or obligation toward self, family, dependants, pets, work, or other people that influences action.",
        "the author feels responsible for protecting, caring for, supporting, or staying for someone/something",
        "the author merely feels guilt, burden, affection, or receives support without an expressed duty",
        "meaning in life", "social support", "low self-esteem",
    ),
    "sexual orientation related issues": _card(
        18, "sexual orientation related issues",
        "Distress, rejection, discrimination, concealment, identity conflict, or relationship conflict tied to sexual orientation or gender identity.",
        "the text explicitly links a problem to sexuality, sexual orientation, gender identity, coming out, or related stigma",
        "the post concerns sexual activity, general relationships, or abuse without the identity-related link",
        "interpersonal difficulty", "dysfunctional family", "interpersonal violence",
    ),
    "social support": _card(
        19, "social support",
        "Meaningful emotional, informational, practical, professional, or emergency help is available or received.",
        "another person, community, professional, or service actually provides care, advice, companionship, protection, or practical help",
        "support is only requested, imagined, unavailable, invalidating, or mentioned as generic advice",
        "poor social support", "coping strategy", "psychological capital",
    ),
    "stressful life event": _card(
        20, "stressful life event",
        "A concrete disruptive occurrence or major change producing substantial stress.",
        "the author reports a loss, breakup, conflict, move, failure, job event, bereavement, crisis, or other identifiable stressful occurrence",
        "the post contains only chronic feelings/conditions or broad life dissatisfaction with no event",
        "traumatic experience", "interpersonal difficulty", "dysfunctional family",
    ),
    "substance use": _card(
        21, "substance use",
        "Use of alcohol, drugs, tobacco, or another substance in a risky, impairing, dependent, intoxicated, or withdrawal-related way.",
        "the author reports consumption, misuse, dependence, cravings, intoxication, withdrawal, or substance-related harm",
        "a substance is merely mentioned, prescribed medication is taken normally, or another person uses it without affecting the author",
        "coping strategy", "mental health issues",
    ),
    "suicide means (with access)": _card(
        22, "suicide means (with access)",
        "A concrete suicide method or potentially lethal means is available, obtainable, or possessed by the author.",
        "the author has, can readily obtain, is physically near, or is actively using a concrete means for suicide",
        "a method is named hypothetically or historically without current access, or there is vague intent without means",
        "prior self-harm or suicidal thought/attempt", "interpersonal violence",
    ),
    "traumatic experience": _card(
        23, "traumatic experience",
        "An overwhelming adverse experience with lasting traumatic psychological impact.",
        "the author describes abuse, assault, disaster, severe threat, or another overwhelming event that continues to affect them as trauma",
        "the event is stressful but not described as overwhelming/traumatic or has no lasting impact",
        "stressful life event", "interpersonal violence", "dysfunctional family",
    ),
}


assert tuple(LABEL_CARDS) == FACTOR_LABELS, "Label-card order must match the official taxonomy"


@dataclass
class B4PConfig:
    seed: int = 42
    n_splits: int = 3
    verifier_model: str = "Qwen/Qwen3-14B"
    anchor_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    retriever_model: str = "intfloat/e5-large-v2"
    max_length: int = 2048
    context_char_budget: int = 6500
    full_post_char_limit: int = 5500
    context_top_clauses: int = 8
    retrieval_example_chars: int = 650
    include_retrieval: bool = True
    verifier_use_chat_template: bool = True
    anchor_use_chat_template: bool = True
    # Decoder-only verification prompts put the operational question at the
    # end.  Keep that suffix whenever a prompt exceeds the token budget, and
    # use the same rule in SFT and scoring.
    prompt_truncation_side: str = "left"
    # Qwen3.8 has 16 full-attention and 48 Gated DeltaNet layers.  The latter
    # are accelerated by optional fla/causal-conv1d packages; this switch only
    # chooses the backend for the 16 full-attention layers.
    attention_implementation: str = "sdpa"
    # Transformers currently misidentifies Qwen3.5-family 3-D M-RoPE position
    # ids as a packed sequence on the FA2 path.  Enable the narrow guard only
    # for an explicitly requested, smoke-tested FA2 experiment.
    qwen35_fa2_position_guard: bool = False
    require_qwen35_fast_kernels: bool = False
    positives_floor: int = 64
    examples_cap_per_class: int = 512
    negative_ratio: float = 1.0
    hard_negative_fraction: float = 0.7
    sft_epochs: float = 1.0
    # Positive values are useful for a cheap architecture/PEFT smoke test.
    # Keep -1 for the normal epoch-controlled run.
    sft_max_steps: int = -1
    sft_learning_rate: float = 1.0e-4
    sft_batch_size: int = 1
    sft_gradient_accumulation: int = 32
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    # None adapts every language layer.  A positive value adapts only the last
    # N decoder layers, which is the practical 27B screening configuration.
    lora_last_n_layers: int | None = None
    lora_target_leaves: tuple[str, ...] | None = None
    gradient_checkpointing: bool = True
    verifier_score_batch_size: int = 8
    anchor_score_batch_size: int = 12
    score_chunk_size: int = 384
    embedding_batch_size: int = 64
    max_embedding_length: int = 512
    threshold_kappa_tail: float = 0.0
    threshold_kappa_mid: float = 2.0
    threshold_kappa_head: float = 2.0
    stack_l2: float = 0.02


@dataclass
class TextCorpus:
    frame: pd.DataFrame
    texts: list[str]
    row_ids: np.ndarray
    user_ids: np.ndarray
    post_ids: np.ndarray


@dataclass
class SemanticCache:
    row_ids: np.ndarray
    doc_embeddings: np.ndarray
    clause_embeddings: np.ndarray
    clause_rows: np.ndarray
    clause_left: np.ndarray
    clause_right: np.ndarray
    card_embeddings: np.ndarray


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_dump(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)


def stable_hash(value: Any, length: int = 16) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip().lower()


def load_test_data(root: str | Path, explicit: str | Path | None = None) -> TextCorpus:
    root = Path(root)
    candidates = [Path(explicit)] if explicit else []
    candidates.extend([root / "leaderboard.xlsx", root / "ieee" / "leaderboard.xlsx", root / "test.xlsx"])
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError("Could not find leaderboard.xlsx/test.xlsx")
    frame = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    columns = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in frame.columns}
    def column(*names: str) -> str:
        for name in names:
            key = re.sub(r"[^a-z0-9]", "", name.lower())
            if key in columns:
                return columns[key]
        raise KeyError(f"Could not find {names} in {frame.columns.tolist()}")
    row_col, user_col, text_col = column("row_id", "id"), column("anon_user_id", "user_id"), column("post", "text")
    try:
        post_col = column("post_id")
        post_ids = frame[post_col].to_numpy()
    except KeyError:
        post_ids = np.arange(len(frame))
    return TextCorpus(
        frame=frame,
        texts=frame[text_col].fillna("").astype(str).tolist(),
        row_ids=frame[row_col].astype(str).to_numpy(),
        user_ids=frame[user_col].astype(str).to_numpy(),
        post_ids=np.asarray(post_ids),
    )


def training_corpus(bundle: b1.DataBundle) -> TextCorpus:
    post_ids = (
        bundle.frame["post_id"].to_numpy()
        if "post_id" in bundle.frame.columns
        else np.arange(len(bundle.texts))
    )
    return TextCorpus(
        frame=bundle.frame,
        texts=bundle.texts,
        row_ids=bundle.row_ids,
        user_ids=bundle.user_ids,
        post_ids=np.asarray(post_ids),
    )


def validate_folds(bundle: b1.DataBundle, folds: np.ndarray) -> dict[str, Any]:
    folds = np.asarray(folds)
    if len(folds) != len(bundle.texts):
        raise ValueError("Fold vector length does not match training data")
    for user in np.unique(bundle.user_ids):
        if len(np.unique(folds[bundle.user_ids == user])) != 1:
            raise AssertionError(f"User leakage detected for {user}")
    hashes = np.asarray([hashlib.sha1(normalize_text(t).encode()).hexdigest() for t in bundle.texts])
    for digest in np.unique(hashes):
        if len(np.unique(folds[hashes == digest])) != 1:
            raise AssertionError(f"Exact-text leakage detected for {digest}")
    supports = []
    for fold in sorted(np.unique(folds)):
        supports.append(bundle.factor_binary[folds == fold].sum(axis=0).astype(int).tolist())
    return {
        "fold_sizes": [int(np.sum(folds == fold)) for fold in sorted(np.unique(folds))],
        "factor_support_by_fold": supports,
        "fold_hash": stable_hash(folds.tolist()),
    }


def card_registry_payload() -> dict[str, Any]:
    return {name: asdict(card) for name, card in LABEL_CARDS.items()}


def freeze_cards(path: str | Path) -> str:
    payload = {"version": "B4P-label-cards-v1", "cards": card_registry_payload()}
    payload["sha256"] = stable_hash(payload["cards"], length=64)
    json_dump(payload, path)
    return payload["sha256"]


def _mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


@torch.inference_mode()
def encode_texts(
    texts: Sequence[str],
    model_name: str,
    batch_size: int = 64,
    max_length: int = 512,
    prefix: str = "passage: ",
) -> np.ndarray:
    """Encode texts with normalized mean-pooled embeddings."""
    from transformers import AutoModel, AutoTokenizer

    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device).eval()
    outputs: list[np.ndarray] = []
    started = time.perf_counter()
    for left in range(0, len(texts), batch_size):
        batch = [prefix + str(text) for text in texts[left : left + batch_size]]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        hidden = model(**encoded).last_hidden_state
        pooled = torch.nn.functional.normalize(_mean_pool(hidden, encoded["attention_mask"]), dim=-1)
        outputs.append(pooled.float().cpu().numpy())
        if left and (left // batch_size) % 25 == 0:
            elapsed = time.perf_counter() - started
            done = min(left + batch_size, len(texts))
            eta = elapsed / done * (len(texts) - done)
            print(f"[B4-P embed] {done}/{len(texts)} | ETA {eta / 60:.1f} min")
    result = np.concatenate(outputs).astype(np.float32)
    del model, tokenizer, outputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _clause_inventory(corpus: TextCorpus, max_clauses: int = 48) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    texts: list[str] = []
    rows: list[int] = []
    lefts: list[int] = []
    rights: list[int] = []
    for row, post in enumerate(corpus.texts):
        spans = b1.split_clauses(post, max_clauses=max_clauses)
        for left, right in spans:
            clause = post[left:right].strip()
            if not clause:
                continue
            texts.append(clause)
            rows.append(row)
            lefts.append(left)
            rights.append(right)
    return texts, np.asarray(rows), np.asarray(lefts), np.asarray(rights)


def prepare_semantic_cache(
    corpus: TextCorpus,
    config: B4PConfig,
    cache_dir: str | Path,
    force: bool = False,
) -> SemanticCache:
    """Cache document, clause and label-card embeddings for one corpus."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = stable_hash(
        {
            "rows": corpus.row_ids.tolist(),
            "text_hashes": [stable_hash(normalize_text(text)) for text in corpus.texts],
            "retriever": config.retriever_model,
            "max_length": config.max_embedding_length,
            "cards": card_registry_payload(),
        }
    )
    path = cache_dir / f"semantic_cache_{fingerprint}.npz"
    if path.exists() and not force:
        saved = np.load(path, allow_pickle=True)
        if saved["row_ids"].astype(str).tolist() != corpus.row_ids.astype(str).tolist():
            raise AssertionError(f"Semantic cache row mismatch: {path}")
        print(f"[B4-P] resumed semantic cache: {path}")
        return SemanticCache(
            row_ids=saved["row_ids"],
            doc_embeddings=saved["doc_embeddings"],
            clause_embeddings=saved["clause_embeddings"],
            clause_rows=saved["clause_rows"],
            clause_left=saved["clause_left"],
            clause_right=saved["clause_right"],
            card_embeddings=saved["card_embeddings"],
        )

    started = time.perf_counter()
    doc_embeddings = encode_texts(
        corpus.texts,
        config.retriever_model,
        config.embedding_batch_size,
        config.max_embedding_length,
        prefix="passage: ",
    )
    clause_texts, clause_rows, clause_left, clause_right = _clause_inventory(corpus)
    clause_embeddings = encode_texts(
        clause_texts,
        config.retriever_model,
        config.embedding_batch_size,
        config.max_embedding_length,
        prefix="passage: ",
    )
    card_queries = [
        f"{card.name}. {card.definition} Present when {card.include}"
        for card in LABEL_CARDS.values()
    ]
    card_embeddings = encode_texts(
        card_queries,
        config.retriever_model,
        config.embedding_batch_size,
        config.max_embedding_length,
        prefix="query: ",
    )
    np.savez_compressed(
        path,
        row_ids=corpus.row_ids,
        doc_embeddings=doc_embeddings.astype(np.float16),
        clause_embeddings=clause_embeddings.astype(np.float16),
        clause_rows=clause_rows,
        clause_left=clause_left,
        clause_right=clause_right,
        card_embeddings=card_embeddings.astype(np.float16),
    )
    print(f"[B4-P] semantic cache built in {(time.perf_counter() - started) / 60:.1f} min: {path}")
    return SemanticCache(
        row_ids=corpus.row_ids,
        doc_embeddings=doc_embeddings.astype(np.float16),
        clause_embeddings=clause_embeddings.astype(np.float16),
        clause_rows=clause_rows,
        clause_left=clause_left,
        clause_right=clause_right,
        card_embeddings=card_embeddings.astype(np.float16),
    )


def label_conditioned_context(
    corpus: TextCorpus,
    cache: SemanticCache,
    row: int,
    label: int,
    config: B4PConfig,
) -> tuple[str, list[tuple[int, int, float]]]:
    """Return the full short post or a relevance-ranked, order-preserving clause view."""
    post = corpus.texts[row]
    if len(post) <= config.full_post_char_limit:
        return post.strip(), [(0, len(post), 1.0)]
    clause_indices = np.flatnonzero(cache.clause_rows == row)
    if not len(clause_indices):
        return post[-config.context_char_budget :].strip(), []
    scores = (
        cache.clause_embeddings[clause_indices].astype(np.float32)
        @ cache.card_embeddings[label].astype(np.float32)
    )
    ranked = clause_indices[np.argsort(scores)[::-1]]
    selected = set(ranked[: config.context_top_clauses].tolist())
    # Preserve discourse framing and final crisis language.
    selected.update(clause_indices[:2].tolist())
    selected.update(clause_indices[-2:].tolist())
    ordered = sorted(selected, key=lambda idx: int(cache.clause_left[idx]))
    pieces: list[str] = []
    audit: list[tuple[int, int, float]] = []
    used = 0
    for idx in ordered:
        left, right = int(cache.clause_left[idx]), int(cache.clause_right[idx])
        piece = post[left:right].strip()
        if not piece:
            continue
        if pieces and used + len(piece) > config.context_char_budget:
            continue
        pieces.append(piece)
        used += len(piece)
        local = int(np.where(clause_indices == idx)[0][0])
        audit.append((left, right, float(scores[local])))
    return "\n…\n".join(pieces), audit


class FoldRetriever:
    """Exact-cosine fold-safe retrieval over the tiny training collection."""

    def __init__(self, bundle: b1.DataBundle, corpus: TextCorpus, cache: SemanticCache):
        self.bundle = bundle
        self.corpus = corpus
        self.embeddings = cache.doc_embeddings.astype(np.float32)
        self.users = corpus.user_ids.astype(str)
        self.hashes = np.asarray([stable_hash(normalize_text(text), 40) for text in corpus.texts])

    def eligible_mask(
        self,
        allowed_indices: np.ndarray,
        query_user: str,
        query_hash: str,
        query_train_index: int | None = None,
    ) -> np.ndarray:
        mask = np.zeros(len(self.corpus.texts), dtype=bool)
        mask[np.asarray(allowed_indices, dtype=int)] = True
        mask &= self.users != str(query_user)
        mask &= self.hashes != query_hash
        if query_train_index is not None:
            mask[int(query_train_index)] = False
        return mask

    def retrieve(
        self,
        query_embedding: np.ndarray,
        query_user: str,
        query_hash: str,
        label: int,
        allowed_indices: np.ndarray,
        query_train_index: int | None = None,
    ) -> dict[str, Any]:
        eligible = self.eligible_mask(allowed_indices, query_user, query_hash, query_train_index)
        scores = self.embeddings @ query_embedding.astype(np.float32)
        positive = eligible & (self.bundle.factor_binary[:, label] == 1)
        negative = eligible & (self.bundle.factor_binary[:, label] == 0)
        def best(mask: np.ndarray) -> tuple[int | None, float | None]:
            indices = np.flatnonzero(mask)
            if not len(indices):
                return None, None
            winner = int(indices[np.argmax(scores[indices])])
            return winner, float(scores[winner])
        pos_index, pos_score = best(positive)
        neg_index, neg_score = best(negative)
        return {
            "positive_index": pos_index,
            "positive_score": pos_score,
            "negative_index": neg_index,
            "negative_score": neg_score,
        }


def _compact_example(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= limit:
        return text
    half = max(1, (limit - 5) // 2)
    return text[:half] + " […] " + text[-half:]


SYSTEM_INSTRUCTION = (
    "You are a deterministic annotation verifier for research text coding. "
    "Judge only whether the post supports the supplied operational factor. "
    "Do not infer a clinical diagnosis, do not explain your reasoning, and output exactly A or B."
)


def make_verification_prompt(
    query_corpus: TextCorpus,
    query_cache: SemanticCache,
    query_row: int,
    label: int,
    train_corpus: TextCorpus,
    retriever: FoldRetriever,
    allowed_train_indices: np.ndarray,
    config: B4PConfig,
    query_train_index: int | None = None,
) -> tuple[str, dict[str, Any]]:
    card = LABEL_CARDS[FACTOR_LABELS[label]]
    context, spans = label_conditioned_context(query_corpus, query_cache, query_row, label, config)
    query_hash = stable_hash(normalize_text(query_corpus.texts[query_row]), 40)
    retrieval = retriever.retrieve(
        query_cache.doc_embeddings[query_row].astype(np.float32),
        str(query_corpus.user_ids[query_row]),
        query_hash,
        label,
        allowed_train_indices,
        query_train_index=query_train_index,
    )
    examples = ""
    if config.include_retrieval:
        pos = retrieval["positive_index"]
        neg = retrieval["negative_index"]
        positive_text = (
            _compact_example(train_corpus.texts[pos], config.retrieval_example_chars)
            if pos is not None else "No eligible positive example in this fold."
        )
        negative_text = (
            _compact_example(train_corpus.texts[neg], config.retrieval_example_chars)
            if neg is not None else "No eligible negative example in this fold."
        )
        examples = (
            "\n[CONTRASTIVE TRAINING REFERENCES]\n"
            "Known PRESENT example:\n" + positive_text + "\n"
            "Known ABSENT but semantically similar example:\n" + negative_text + "\n"
            "[/CONTRASTIVE TRAINING REFERENCES]\n"
        )
    prompt = (
        SYSTEM_INSTRUCTION
        + "\n\n[POST]\n" + context + "\n[/POST]\n\n"
        + "[LABEL CARD]\n" + card.render() + "\n[/LABEL CARD]\n"
        + examples
        + f"\nQuestion: Does the post support {card.name}?\n"
        + "A = PRESENT\nB = ABSENT\nAnswer:"
    )
    audit = {
        "query_row_id": str(query_corpus.row_ids[query_row]),
        "label": card.name,
        "factor_id": card.factor_id,
        "selected_spans": spans,
        **retrieval,
    }
    return prompt, audit


def _lexical_boundary_score(text: str, card: LabelCard) -> float:
    stop = {
        "the", "a", "an", "or", "and", "to", "of", "with", "is", "are", "in", "for",
        "author", "reports", "describes", "person", "their", "they", "another", "only",
    }
    query = re.findall(r"[a-z']+", (card.name + " " + card.definition).lower())
    terms = {word for word in query if len(word) >= 4 and word not in stop}
    if not terms:
        return 0.0
    words = set(re.findall(r"[a-z']+", text.lower()))
    return float(len(terms & words) / math.sqrt(len(terms)))


def build_pair_manifest(
    bundle: b1.DataBundle,
    train_cache: SemanticCache,
    allowed_indices: np.ndarray,
    config: B4PConfig,
    fold: int | str,
) -> pd.DataFrame:
    """Balanced per-label SFT curriculum with real fold-local hard negatives."""
    rng = np.random.default_rng(config.seed + (int(fold) if str(fold).isdigit() else 991))
    allowed = np.asarray(allowed_indices, dtype=int)
    embeddings = train_cache.doc_embeddings.astype(np.float32)
    rows: list[dict[str, Any]] = []
    for label, name in enumerate(FACTOR_LABELS):
        card = LABEL_CARDS[name]
        positive = allowed[bundle.factor_binary[allowed, label] == 1]
        negative = allowed[bundle.factor_binary[allowed, label] == 0]
        if not len(positive) or not len(negative):
            raise ValueError(f"Fold {fold} has no positive/negative support for {name}")

        target_per_class = min(
            config.examples_cap_per_class,
            max(config.positives_floor, len(positive)),
        )
        positive_sample = rng.choice(
            positive,
            size=target_per_class,
            replace=len(positive) < target_per_class,
        )

        centroid = embeddings[positive].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-8
        semantic = embeddings[negative] @ centroid
        confusion_ids = [LABEL_TO_ID[item] for item in card.confusions if item in LABEL_TO_ID]
        confusion = (
            bundle.factor_binary[negative][:, confusion_ids].max(axis=1).astype(np.float32)
            if confusion_ids else np.zeros(len(negative), dtype=np.float32)
        )
        lexical = np.asarray(
            [_lexical_boundary_score(bundle.texts[idx], card) for idx in negative],
            dtype=np.float32,
        )
        if lexical.max(initial=0.0) > 0:
            lexical /= lexical.max()
        hard_score = 0.65 * semantic + 0.25 * confusion + 0.10 * lexical
        negative_count = max(1, int(round(target_per_class * config.negative_ratio)))
        hard_count = min(len(negative), int(round(negative_count * config.hard_negative_fraction)))
        ranked = negative[np.argsort(hard_score)[::-1]]
        hard_sample = ranked[:hard_count]
        remaining_pool = np.setdiff1d(negative, hard_sample, assume_unique=False)
        random_count = negative_count - hard_count
        if random_count:
            pool = remaining_pool if len(remaining_pool) else negative
            random_sample = rng.choice(pool, size=random_count, replace=len(pool) < random_count)
        else:
            random_sample = np.asarray([], dtype=int)

        for idx in positive_sample:
            rows.append({"row_idx": int(idx), "label_idx": label, "target": 1, "kind": "positive"})
        for idx in hard_sample:
            rows.append({"row_idx": int(idx), "label_idx": label, "target": 0, "kind": "hard_negative"})
        for idx in random_sample:
            rows.append({"row_idx": int(idx), "label_idx": label, "target": 0, "kind": "random_negative"})

    frame = pd.DataFrame(rows)
    frame = frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)
    frame["pair_id"] = [f"train::{fold}::{i:06d}" for i in range(len(frame))]
    return frame


def build_prompt_table(
    query_corpus: TextCorpus,
    query_cache: SemanticCache,
    train_corpus: TextCorpus,
    retriever: FoldRetriever,
    allowed_train_indices: np.ndarray,
    config: B4PConfig,
    query_rows: np.ndarray | None = None,
    train_targets: np.ndarray | None = None,
    query_is_training_corpus: bool = False,
) -> pd.DataFrame:
    """Materialize query-label prompts and a compact provenance audit."""
    if query_rows is None:
        query_rows = np.arange(len(query_corpus.texts))
    records: list[dict[str, Any]] = []
    allowed_set = set(np.asarray(allowed_train_indices, dtype=int).tolist())
    started = time.perf_counter()
    for counter, row in enumerate(np.asarray(query_rows, dtype=int)):
        for label, card in enumerate(LABEL_CARDS.values()):
            prompt, audit = make_verification_prompt(
                query_corpus,
                query_cache,
                int(row),
                label,
                train_corpus,
                retriever,
                allowed_train_indices,
                config,
                query_train_index=int(row) if query_is_training_corpus else None,
            )
            for key in ("positive_index", "negative_index"):
                index = audit[key]
                if index is not None:
                    if int(index) not in allowed_set:
                        raise AssertionError(f"Retriever escaped fold mask: {index}")
                    if str(train_corpus.user_ids[index]) == str(query_corpus.user_ids[row]):
                        raise AssertionError("Same-user retrieval leakage")
                    if normalize_text(train_corpus.texts[index]) == normalize_text(query_corpus.texts[row]):
                        raise AssertionError("Exact-duplicate retrieval leakage")
            record = {
                "pair_id": f"{query_corpus.row_ids[row]}::{card.factor_id}",
                "query_row_idx": int(row),
                "label_idx": label,
                "label": card.name,
                "prompt": prompt,
                "positive_index": audit["positive_index"],
                "negative_index": audit["negative_index"],
                "context_spans": json.dumps(audit["selected_spans"]),
            }
            if train_targets is not None:
                record["target"] = int(train_targets[row, label])
            records.append(record)
        if counter and counter % 150 == 0:
            elapsed = time.perf_counter() - started
            done = counter + 1
            eta = elapsed / done * (len(query_rows) - done)
            print(f"[B4-P prompts] {done}/{len(query_rows)} posts | ETA {eta / 60:.1f} min")
    return pd.DataFrame(records)


class VerificationSFTDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        train_corpus: TextCorpus,
        train_cache: SemanticCache,
        retriever: FoldRetriever,
        allowed_train_indices: np.ndarray,
        config: B4PConfig,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.train_corpus = train_corpus
        self.train_cache = train_cache
        self.retriever = retriever
        self.allowed = np.asarray(allowed_train_indices, dtype=int)
        self.config = config

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.manifest.iloc[index]
        row, label = int(record.row_idx), int(record.label_idx)
        prompt, _ = make_verification_prompt(
            self.train_corpus,
            self.train_cache,
            row,
            label,
            self.train_corpus,
            self.retriever,
            self.allowed,
            self.config,
            query_train_index=row,
        )
        return {
            "prompt": prompt,
            "target": int(record.target),
            "pair_id": str(record.pair_id),
        }


def resolve_answer_token_ids(tokenizer: Any) -> tuple[int, int, str, str]:
    for present, absent in (("A", "B"), (" A", " B")):
        a = tokenizer.encode(present, add_special_tokens=False)
        b = tokenizer.encode(absent, add_special_tokens=False)
        if len(a) == len(b) == 1 and a[0] != b[0]:
            return int(a[0]), int(b[0]), present, absent
    raise ValueError("Tokenizer does not encode A/B as distinct single tokens; add sequence scoring")


def render_chat_prompts(tokenizer: Any, prompts: Sequence[str]) -> list[str]:
    if not getattr(tokenizer, "chat_template", None):
        return list(prompts)
    rendered_prompts: list[str] = []
    for prompt in prompts:
        system_content = SYSTEM_INSTRUCTION
        user_content = prompt
        if prompt.startswith("[SYSTEM]\n") and "\n[/SYSTEM]\n" in prompt:
            system_content, user_content = prompt[len("[SYSTEM]\n") :].split(
                "\n[/SYSTEM]\n", 1
            )
        elif prompt.startswith(SYSTEM_INSTRUCTION):
            user_content = prompt[len(SYSTEM_INSTRUCTION) :].lstrip()
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        try:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        rendered_prompts.append(rendered)
    return rendered_prompts


class CompletionOnlyABCollator:
    def __init__(
        self,
        tokenizer: Any,
        max_length: int,
        use_chat_template: bool = True,
        truncation_side: str = "left",
    ):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.use_chat_template = bool(use_chat_template)
        self.truncation_side = str(truncation_side)
        if self.truncation_side not in {"left", "right"}:
            raise ValueError("truncation_side must be 'left' or 'right'")
        self.a_id, self.b_id, _, _ = resolve_answer_token_ids(tokenizer)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # AutoTokenizer defaults to right truncation.  Scoring historically
        # used left truncation, which made short-context experiments train and
        # evaluate on different prompt views.  Set it explicitly every call.
        self.tokenizer.truncation_side = self.truncation_side
        encoded_rows: list[list[int]] = []
        prompts = [row["prompt"] for row in rows]
        if self.use_chat_template:
            prompts = render_chat_prompts(self.tokenizer, prompts)
        for row, prompt in zip(rows, prompts):
            ids = self.tokenizer.encode(
                prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length - 1,
            )
            ids.append(self.a_id if int(row["target"]) == 1 else self.b_id)
            encoded_rows.append(ids)
        width = max(len(ids) for ids in encoded_rows)
        pad = int(self.tokenizer.pad_token_id)
        input_ids = torch.full((len(rows), width), pad, dtype=torch.long)
        attention = torch.zeros((len(rows), width), dtype=torch.long)
        labels = torch.full((len(rows), width), -100, dtype=torch.long)
        for i, ids in enumerate(encoded_rows):
            start = width - len(ids)
            input_ids[i, start:] = torch.tensor(ids)
            attention[i, start:] = 1
            labels[i, -1] = ids[-1]
        return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


def audit_verification_dataset_lengths(
    dataset: Dataset,
    tokenizer: Any,
    max_length: int,
    use_chat_template: bool = True,
    sample_size: int = 512,
) -> dict[str, Any]:
    """Audit raw rendered prompt lengths without persisting sensitive text."""
    if len(dataset) == 0:
        raise ValueError("Cannot audit an empty verification dataset")
    count = min(int(sample_size), len(dataset))
    indices = np.unique(np.linspace(0, len(dataset) - 1, count, dtype=int))
    lengths: list[int] = []
    for index in indices:
        prompt = str(dataset[int(index)]["prompt"])
        rendered = render_chat_prompts(tokenizer, [prompt])[0] if use_chat_template else prompt
        lengths.append(len(tokenizer.encode(rendered, add_special_tokens=True, truncation=False)))
    values = np.asarray(lengths, dtype=np.int32)
    result = {
        "sample_size": int(len(values)),
        "max_length": int(max_length),
        "mean_tokens": float(values.mean()),
        "p50_tokens": float(np.quantile(values, 0.50)),
        "p90_tokens": float(np.quantile(values, 0.90)),
        "p95_tokens": float(np.quantile(values, 0.95)),
        "max_tokens": int(values.max()),
        "over_limit_fraction": float(np.mean(values > (int(max_length) - 1))),
    }
    print("[B4-P] prompt length audit:", result)
    return result


def _quantization_config() -> Any:
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def qwen35_kernel_status() -> dict[str, Any]:
    """Report optional hybrid-kernel availability without loading the model."""
    from transformers.utils.import_utils import (
        is_causal_conv1d_available,
        is_flash_linear_attention_available,
    )
    return {
        "causal_conv1d": bool(is_causal_conv1d_available()),
        "flash_linear_attention": bool(is_flash_linear_attention_available()),
        "cuda": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def install_qwen35_fa2_position_guard() -> bool:
    """Guard FA2 against Qwen3.5/3.8 3-D M-RoPE packed-sequence detection.

    Packed text sequences use 2-D position ids.  Qwen3.5-family multimodal
    RoPE uses a leading three-component axis which must never be passed to the
    packed-sequence preparation path.  This is intentionally narrow and is
    activated only by an explicit config flag.
    """
    import functools
    import transformers.modeling_flash_attention_utils as fa_utils

    current = fa_utils._is_packed_sequence
    if getattr(current, "_qwen35_mrope_guard", False):
        return False

    @functools.wraps(current)
    def guarded(position_ids: Any, batch_size: int, *args: Any, **kwargs: Any) -> Any:
        if position_ids is not None and getattr(position_ids, "ndim", 0) > 2:
            return False
        return current(position_ids, batch_size, *args, **kwargs)

    guarded._qwen35_mrope_guard = True
    fa_utils._is_packed_sequence = guarded
    print("[B4-P] installed Qwen3.5/3.8 FA2 3-D M-RoPE position guard")
    return True


def load_quantized_causal_model(
    model_name: str,
    adapter_path: str | Path | None = None,
    training: bool = False,
    attention_implementation: str = "sdpa",
    qwen35_fa2_position_guard: bool = False,
    require_qwen35_fast_kernels: bool = False,
) -> tuple[Any, Any]:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    checkpoint_config = AutoConfig.from_pretrained(model_name)
    architectures = tuple(getattr(checkpoint_config, "architectures", None) or ())
    is_qwen35_family = str(getattr(checkpoint_config, "model_type", "")) == "qwen3_5"
    if attention_implementation == "flash_attention_2" and qwen35_fa2_position_guard:
        install_qwen35_fa2_position_guard()
    # Qwen3.8 is distributed as a native multimodal Qwen3.5-family checkpoint.
    # AutoModelForCausalLM is correct for text-only checkpoints, while the full
    # checkpoint must retain its language-model wrapper so the state dict maps
    # correctly.  We still feed text-only input_ids and never instantiate image
    # inputs in this competition.
    if any("ConditionalGeneration" in name for name in architectures):
        try:
            from transformers import AutoModelForMultimodalLM
            model_class = AutoModelForMultimodalLM
        except ImportError:
            try:
                from transformers import AutoModelForImageTextToText
                model_class = AutoModelForImageTextToText
            except ImportError:
                from transformers import AutoModelForVision2Seq
                model_class = AutoModelForVision2Seq
    else:
        model_class = AutoModelForCausalLM
    print(
        f"[B4-P] loading {model_name} with {model_class.__name__}; "
        f"architectures={list(architectures)}"
    )
    attention_backend: Any = attention_implementation
    if is_qwen35_family and any("ConditionalGeneration" in name for name in architectures):
        attention_backend = {
            "text_config": attention_implementation,
            "vision_config": "sdpa",
        }
    print(f"[B4-P] attention backend: {attention_backend}")
    kernel_status = qwen35_kernel_status() if is_qwen35_family else None
    if kernel_status is not None:
        print(f"[B4-P] Qwen3.5-family kernel status: {kernel_status}")
        if require_qwen35_fast_kernels and not (
            kernel_status["causal_conv1d"] and kernel_status["flash_linear_attention"]
        ):
            raise RuntimeError(
                "Qwen3.5-family fast kernels are required but unavailable; "
                "install causal-conv1d and flash-linear-attention, restart the runtime, and retry"
            )
    model = model_class.from_pretrained(
        model_name,
        quantization_config=_quantization_config(),
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation=attention_backend,
    )
    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=training)
    model.config.use_cache = not training
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None and hasattr(text_config, "use_cache"):
        text_config.use_cache = not training
    return model, tokenizer


def discover_language_lora_targets(
    model: Any,
    last_n_layers: int | None = None,
    target_leaves: Sequence[str] | None = None,
) -> list[str]:
    """Return exact language-tower projection names for Qwen/Qwen3.8.

    Qwen3.8 mixes full-attention and Gated DeltaNet blocks.  The legacy Qwen3
    target list misses the DeltaNet projections; suffix-only PEFT matching can
    also accidentally adapt the vision tower.  Exact, audited module names
    avoid both failure modes.
    """
    suffixes = {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
    }
    if target_leaves is not None:
        requested = set(map(str, target_leaves))
        unknown = requested - suffixes
        if unknown:
            raise ValueError(f"Unknown LoRA target leaves: {sorted(unknown)}")
        suffixes = requested
    vision_markers = ("visual", "vision", "image", "patch_embed", "merger")
    targets: list[str] = []
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        lowered = name.lower()
        if leaf not in suffixes or any(marker in lowered for marker in vision_markers):
            continue
        if not hasattr(module, "weight"):
            continue
        targets.append(name)
    targets = sorted(set(targets))
    if not targets:
        raise RuntimeError("No language LoRA projection modules were discovered")
    if last_n_layers is not None:
        if int(last_n_layers) <= 0:
            raise ValueError("lora_last_n_layers must be positive or None")
        layer_pattern = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
        indexed = [
            (int(match.group(1)), name)
            for name in targets
            if (match := layer_pattern.search(name)) is not None
        ]
        if not indexed:
            raise RuntimeError("Could not identify decoder layer indices for partial LoRA")
        maximum = max(index for index, _ in indexed)
        minimum = max(0, maximum - int(last_n_layers) + 1)
        targets = [name for index, name in indexed if index >= minimum]
        print(f"[B4-P] partial LoRA decoder layers {minimum}..{maximum}")
    print(f"[B4-P] discovered {len(targets)} exact language LoRA targets")
    print("[B4-P] LoRA target leaves:", sorted({name.rsplit('.', 1)[-1] for name in targets}))
    return targets


def train_verifier_adapter(
    dataset: VerificationSFTDataset,
    config: B4PConfig,
    output_dir: str | Path,
    overwrite: bool = False,
) -> Path:
    """Train exactly one shared QLoRA adapter for all 24 factors."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments

    output_dir = Path(output_dir)
    final_dir = output_dir / "adapter_final"
    if (final_dir / "adapter_config.json").exists() and not overwrite:
        print(f"[B4-P] resumed verifier adapter: {final_dir}")
        return final_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    torch.set_float32_matmul_precision("high")
    model, tokenizer = load_quantized_causal_model(
        config.verifier_model,
        training=True,
        attention_implementation=config.attention_implementation,
        qwen35_fa2_position_guard=config.qwen35_fa2_position_guard,
        require_qwen35_fast_kernels=config.require_qwen35_fast_kernels,
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=config.gradient_checkpointing
    )
    if config.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    lora_targets = discover_language_lora_targets(
        model,
        last_n_layers=config.lora_last_n_layers,
        target_leaves=config.lora_target_leaves,
    )
    lora = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    collator = CompletionOnlyABCollator(
        tokenizer,
        config.max_length,
        use_chat_template=config.verifier_use_chat_template,
        truncation_side=config.prompt_truncation_side,
    )
    prompt_length_audit = audit_verification_dataset_lengths(
        dataset,
        tokenizer,
        config.max_length,
        use_chat_template=config.verifier_use_chat_template,
    )
    # TrainingArguments has changed across Colab's transformers releases.
    # Use explicit warmup_steps and retain only arguments supported by the
    # installed signature instead of forcing a runtime restart/version pin.
    updates_per_epoch = max(
        1,
        math.ceil(
            len(dataset)
            / max(1, config.sft_batch_size * config.sft_gradient_accumulation)
        ),
    )
    warmup_steps = max(1, int(round(updates_per_epoch * config.sft_epochs * 0.05)))
    # A Colab Pro runtime can disappear without warning. Persist a resumable
    # optimizer/adapter checkpoint roughly eight times per epoch; the prior
    # fixed value (250) exceeded this experiment's 225 total update steps.
    checkpoint_steps = max(10, min(50, updates_per_epoch // 8))
    argument_values: dict[str, Any] = {
        "output_dir": str(output_dir / "checkpoints"),
        "num_train_epochs": config.sft_epochs,
        "per_device_train_batch_size": config.sft_batch_size,
        "gradient_accumulation_steps": config.sft_gradient_accumulation,
        "learning_rate": config.sft_learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_steps": warmup_steps,
        "weight_decay": 0.01,
        "optim": "paged_adamw_8bit",
        "bf16": True,
        "tf32": True,
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_steps": checkpoint_steps,
        "save_total_limit": 2,
        "report_to": "none",
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "seed": config.seed,
    }
    if config.gradient_checkpointing:
        argument_values["gradient_checkpointing"] = True
        argument_values["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if config.sft_max_steps > 0:
        argument_values["max_steps"] = int(config.sft_max_steps)
    supported = inspect.signature(TrainingArguments.__init__).parameters
    dropped = sorted(set(argument_values) - set(supported))
    if dropped:
        print(f"[B4-P] TrainingArguments compatibility: omitted unsupported {dropped}")
    arguments = TrainingArguments(
        **{key: value for key, value in argument_values.items() if key in supported}
    )
    trainer = Trainer(model=model, args=arguments, train_dataset=dataset, data_collator=collator)
    checkpoints = sorted(
        (output_dir / "checkpoints").glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )
    training_output = trainer.train(
        resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None
    )
    trainer_metrics = {
        str(key): float(value) if isinstance(value, (int, float, np.number)) else str(value)
        for key, value in dict(getattr(training_output, "metrics", {}) or {}).items()
    }
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    json_dump(
        {
            "config": asdict(config),
            "training_pairs": len(dataset),
            "updates_per_epoch": updates_per_epoch,
            "warmup_steps": warmup_steps,
            "checkpoint_steps": checkpoint_steps,
            "lora_target_count": len(lora_targets),
            "lora_target_leaves": sorted({name.rsplit('.', 1)[-1] for name in lora_targets}),
            "prompt_truncation_side": config.prompt_truncation_side,
            "prompt_length_audit": prompt_length_audit,
            "attention_implementation": config.attention_implementation,
            "kernel_status": qwen35_kernel_status(),
            "trainer_metrics": trainer_metrics,
        },
        output_dir / "train_manifest.json",
    )
    del trainer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return final_dir


@torch.inference_mode()
def _score_prompt_batch(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    max_length: int,
    use_chat_template: bool = False,
    truncation_side: str = "left",
) -> np.ndarray:
    a_id, b_id, _, _ = resolve_answer_token_ids(tokenizer)
    tokenizer.padding_side = "left"
    if truncation_side not in {"left", "right"}:
        raise ValueError("truncation_side must be 'left' or 'right'")
    tokenizer.truncation_side = truncation_side
    prepared = render_chat_prompts(tokenizer, prompts) if use_chat_template else list(prompts)
    encoded = tokenizer(
        prepared,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    logits = model(**encoded, use_cache=False).logits[:, -1, :]
    return (logits[:, a_id] - logits[:, b_id]).float().cpu().numpy()


def score_prompts_cached(
    model: Any,
    tokenizer: Any,
    prompt_frame: pd.DataFrame,
    cache_dir: str | Path,
    config: B4PConfig,
    batch_size: int,
    use_chat_template: bool = False,
) -> np.ndarray:
    """Score A-vs-B margins with resumable immutable chunks."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pair_ids = prompt_frame["pair_id"].astype(str).to_numpy()
    fingerprint = stable_hash(
        {
            "pairs": pair_ids.tolist(),
            "max_length": config.max_length,
            "model": getattr(model, "name_or_path", "loaded-model"),
            "use_chat_template": use_chat_template,
            "truncation_side": config.prompt_truncation_side,
            "attention_implementation": config.attention_implementation,
            "qwen35_fa2_position_guard": config.qwen35_fa2_position_guard,
        }
    )
    final_path = cache_dir / f"scores_{fingerprint}.npz"
    if final_path.exists():
        saved = np.load(final_path, allow_pickle=True)
        if saved["pair_ids"].astype(str).tolist() != pair_ids.tolist():
            raise AssertionError(f"Score-cache pair mismatch: {final_path}")
        print(f"[B4-P] resumed scores: {final_path}")
        return saved["logits"].astype(np.float32)

    logits = np.full(len(prompt_frame), np.nan, dtype=np.float32)
    started = time.perf_counter()
    for left in range(0, len(prompt_frame), config.score_chunk_size):
        right = min(len(prompt_frame), left + config.score_chunk_size)
        chunk_path = cache_dir / f"part_{fingerprint}_{left:06d}_{right:06d}.npz"
        expected_ids = pair_ids[left:right]
        if chunk_path.exists():
            saved = np.load(chunk_path, allow_pickle=True)
            if saved["pair_ids"].astype(str).tolist() != expected_ids.tolist():
                raise AssertionError(f"Chunk pair mismatch: {chunk_path}")
            logits[left:right] = saved["logits"]
            continue
        chunk_logits: list[np.ndarray] = []
        cursor = left
        current_batch = int(batch_size)
        while cursor < right:
            stop = min(right, cursor + current_batch)
            try:
                scores = _score_prompt_batch(
                    model,
                    tokenizer,
                    prompt_frame["prompt"].iloc[cursor:stop].tolist(),
                    config.max_length,
                    use_chat_template=use_chat_template,
                    truncation_side=config.prompt_truncation_side,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if current_batch <= 1:
                    raise
                current_batch = max(1, current_batch // 2)
                print(f"[B4-P] OOM backoff: score batch -> {current_batch}")
                continue
            chunk_logits.append(scores)
            cursor = stop
        values = np.concatenate(chunk_logits).astype(np.float32)
        logits[left:right] = values
        np.savez_compressed(chunk_path, pair_ids=expected_ids, logits=values)
        elapsed = time.perf_counter() - started
        done = right
        eta = elapsed / max(done, 1) * (len(prompt_frame) - done)
        print(f"[B4-P score] {done}/{len(prompt_frame)} | ETA {eta / 60:.1f} min")
    if not np.isfinite(logits).all():
        raise AssertionError("Incomplete score cache")
    np.savez_compressed(final_path, pair_ids=pair_ids, logits=logits)
    return logits


def unload_model(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pair_logits_to_matrix(prompt_frame: pd.DataFrame, pair_logits: np.ndarray, n_rows: int) -> np.ndarray:
    matrix = np.full((n_rows, len(FACTOR_LABELS)), np.nan, dtype=np.float32)
    for row, label, score in zip(
        prompt_frame["query_row_idx"].astype(int),
        prompt_frame["label_idx"].astype(int),
        np.asarray(pair_logits, dtype=np.float32),
    ):
        if np.isfinite(matrix[row, label]):
            raise AssertionError(f"Duplicate pair score for row={row}, label={label}")
        matrix[row, label] = score
    return matrix


def run_anchor_scoring(
    query_corpus: TextCorpus,
    query_cache: SemanticCache,
    train_corpus: TextCorpus,
    train_bundle: b1.DataBundle,
    train_cache: SemanticCache,
    allowed_train_indices: np.ndarray,
    config: B4PConfig,
    artifact_dir: str | Path,
    query_rows: np.ndarray | None = None,
    query_is_training_corpus: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Score all 24 labels with the frozen performance-first anchor."""
    artifact_dir = Path(artifact_dir)
    retriever = FoldRetriever(train_bundle, train_corpus, train_cache)
    prompts = build_prompt_table(
        query_corpus,
        query_cache,
        train_corpus,
        retriever,
        allowed_train_indices,
        config,
        query_rows=query_rows,
        train_targets=train_bundle.factor_binary if query_is_training_corpus else None,
        query_is_training_corpus=query_is_training_corpus,
    )
    audit_columns = [column for column in prompts.columns if column != "prompt"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    prompts[audit_columns].to_csv(artifact_dir / "prompt_audit.csv", index=False)
    model, tokenizer = load_quantized_causal_model(
        config.anchor_model,
        attention_implementation=config.attention_implementation,
        qwen35_fa2_position_guard=config.qwen35_fa2_position_guard,
        require_qwen35_fast_kernels=config.require_qwen35_fast_kernels,
    )
    logits = score_prompts_cached(
        model,
        tokenizer,
        prompts,
        artifact_dir / "score_cache",
        config,
        config.anchor_score_batch_size,
        use_chat_template=config.anchor_use_chat_template,
    )
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    matrix = pair_logits_to_matrix(prompts, logits, len(query_corpus.texts))
    return matrix, prompts[audit_columns]


def run_verifier_scoring(
    adapter_path: str | Path,
    query_corpus: TextCorpus,
    query_cache: SemanticCache,
    train_corpus: TextCorpus,
    train_bundle: b1.DataBundle,
    train_cache: SemanticCache,
    allowed_train_indices: np.ndarray,
    config: B4PConfig,
    artifact_dir: str | Path,
    query_rows: np.ndarray | None = None,
    query_is_training_corpus: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    artifact_dir = Path(artifact_dir)
    retriever = FoldRetriever(train_bundle, train_corpus, train_cache)
    prompts = build_prompt_table(
        query_corpus,
        query_cache,
        train_corpus,
        retriever,
        allowed_train_indices,
        config,
        query_rows=query_rows,
        train_targets=train_bundle.factor_binary if query_is_training_corpus else None,
        query_is_training_corpus=query_is_training_corpus,
    )
    audit_columns = [column for column in prompts.columns if column != "prompt"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    prompts[audit_columns].to_csv(artifact_dir / "prompt_audit.csv", index=False)
    model, tokenizer = load_quantized_causal_model(
        config.verifier_model,
        adapter_path=adapter_path,
        attention_implementation=config.attention_implementation,
        qwen35_fa2_position_guard=config.qwen35_fa2_position_guard,
        require_qwen35_fast_kernels=config.require_qwen35_fast_kernels,
    )
    logits = score_prompts_cached(
        model,
        tokenizer,
        prompts,
        artifact_dir / "score_cache",
        config,
        config.verifier_score_batch_size,
        use_chat_template=config.verifier_use_chat_template,
    )
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    matrix = pair_logits_to_matrix(prompts, logits, len(query_corpus.texts))
    return matrix, prompts[audit_columns]


def train_one_outer_fold(
    bundle: b1.DataBundle,
    folds: np.ndarray,
    train_cache: SemanticCache,
    fold: int,
    config: B4PConfig,
    artifact_root: str | Path,
    overwrite: bool = False,
) -> Path:
    artifact_root = Path(artifact_root)
    train_indices = np.flatnonzero(folds != fold)
    fold_dir = artifact_root / f"fold_{fold}"
    manifest_path = fold_dir / "sft_pair_manifest.csv"
    if manifest_path.exists() and not overwrite:
        manifest = pd.read_csv(manifest_path)
    else:
        manifest = build_pair_manifest(bundle, train_cache, train_indices, config, fold)
        fold_dir.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(manifest_path, index=False)
    corpus = training_corpus(bundle)
    retriever = FoldRetriever(bundle, corpus, train_cache)
    dataset = VerificationSFTDataset(manifest, corpus, train_cache, retriever, train_indices, config)
    return train_verifier_adapter(dataset, config, fold_dir / "verifier", overwrite=overwrite)


def benchmark_models(
    prompts: Sequence[str],
    config: B4PConfig,
    artifact_root: str | Path,
    run_anchor: bool = True,
    run_verifier_base: bool = True,
) -> pd.DataFrame:
    """Short throughput benchmark used before authorizing multi-hour runs."""
    artifact_root = Path(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    sample = list(prompts[: min(192, len(prompts))])
    rows: list[dict[str, Any]] = []
    jobs = []
    if run_verifier_base:
        jobs.append((
            "verifier_base", config.verifier_model, config.verifier_score_batch_size,
            config.verifier_use_chat_template,
        ))
    if run_anchor:
        jobs.append(("anchor", config.anchor_model, config.anchor_score_batch_size, config.anchor_use_chat_template))
    for name, model_name, batch, use_chat in jobs:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model, tokenizer = load_quantized_causal_model(
            model_name,
            attention_implementation=config.attention_implementation,
            qwen35_fa2_position_guard=config.qwen35_fa2_position_guard,
            require_qwen35_fast_kernels=config.require_qwen35_fast_kernels,
        )
        load_seconds = time.perf_counter() - started
        started_score = time.perf_counter()
        _ = []
        current = batch
        for left in range(0, len(sample), current):
            _.append(_score_prompt_batch(
                model, tokenizer, sample[left : left + current], config.max_length,
                use_chat_template=use_chat,
            ))
        score_seconds = time.perf_counter() - started_score
        rows.append(
            {
                "model": name,
                "model_name": model_name,
                "pairs": len(sample),
                "load_minutes": load_seconds / 60,
                "score_minutes": score_seconds / 60,
                "pairs_per_second": len(sample) / max(score_seconds, 1e-6),
                "projected_13080_pair_minutes": 13080 / max(len(sample) / score_seconds, 1e-6) / 60,
                "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0,
            }
        )
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    frame.to_csv(artifact_root / "throughput_benchmark.csv", index=False)
    return frame


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), -30, 30)
    return (1.0 / (1.0 + np.exp(-values))).astype(np.float32)


def safe_macro_ap(targets: np.ndarray, probabilities: np.ndarray) -> float:
    scores = []
    for label in range(targets.shape[1]):
        if np.unique(targets[:, label]).size < 2:
            continue
        scores.append(average_precision_score(targets[:, label], probabilities[:, label]))
    return float(np.mean(scores)) if scores else float("nan")


def label_strata(targets: np.ndarray) -> np.ndarray:
    support = targets.sum(axis=0)
    return np.where(support < 60, "tail", np.where(support < 200, "mid", "head"))


def _threshold_kappa_for_label(targets: np.ndarray, label: int, config: B4PConfig) -> float:
    support = int(targets[:, label].sum())
    if support < 60:
        return config.threshold_kappa_tail
    if support < 200:
        return config.threshold_kappa_mid
    return config.threshold_kappa_head


def fit_factor_thresholds(
    probabilities: np.ndarray,
    targets: np.ndarray,
    config: B4PConfig,
) -> np.ndarray:
    grid = np.linspace(0.02, 0.98, 97)
    global_scores = [
        f1_score(targets, probabilities >= threshold, average="macro", zero_division=0)
        for threshold in grid
    ]
    global_threshold = float(grid[int(np.argmax(global_scores))])
    result = np.full(targets.shape[1], global_threshold, dtype=np.float32)
    for label in range(targets.shape[1]):
        support = int(targets[:, label].sum())
        if support == 0 or support == len(targets):
            continue
        scores = [
            f1_score(targets[:, label], probabilities[:, label] >= threshold, zero_division=0)
            for threshold in grid
        ]
        local = float(grid[int(np.argmax(scores))])
        kappa = _threshold_kappa_for_label(targets, label, config)
        weight = support / (support + kappa) if kappa > 0 else 1.0
        result[label] = weight * local + (1.0 - weight) * global_threshold
    return result


def strict_crossfit_predictions(
    probabilities: np.ndarray,
    targets: np.ndarray,
    folds: np.ndarray,
    config: B4PConfig,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.zeros_like(targets, dtype=np.int8)
    thresholds = []
    for fold in sorted(np.unique(folds)):
        train, valid = folds != fold, folds == fold
        fold_thresholds = fit_factor_thresholds(probabilities[train], targets[train], config)
        predictions[valid] = (probabilities[valid] >= fold_thresholds).astype(np.int8)
        thresholds.append(fold_thresholds)
    return predictions, np.asarray(thresholds)


def factor_metric_bundle(
    targets: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0
    )
    strata = np.where(support < 60, "tail", np.where(support < 200, "mid", "head"))
    metrics = {
        "macro_ap": safe_macro_ap(targets, probabilities),
        "macro_f1": float(np.mean(f1)),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "tail_macro_f1": float(np.mean(f1[strata == "tail"])),
        "mid_macro_f1": float(np.mean(f1[strata == "mid"])),
        "head_macro_f1": float(np.mean(f1[strata == "head"])),
        "mean_predicted_labels": float(predictions.sum(axis=1).mean()),
    }
    table = pd.DataFrame(
        {
            "label": FACTOR_LABELS,
            "stratum": strata,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap": [
                average_precision_score(targets[:, label], probabilities[:, label])
                if np.unique(targets[:, label]).size == 2 else np.nan
                for label in range(targets.shape[1])
            ],
        }
    )
    return metrics, table


def evaluate_oof_logits(
    logits: np.ndarray,
    targets: np.ndarray,
    folds: np.ndarray,
    config: B4PConfig,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    if not np.isfinite(logits).all():
        raise ValueError("OOF logits contain missing values")
    probabilities = sigmoid(logits)
    predictions, thresholds = strict_crossfit_predictions(probabilities, targets, folds, config)
    metrics, table = factor_metric_bundle(targets, probabilities, predictions)
    return metrics, table, thresholds


def diagnostic_fold_metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Threshold-free gate plus clearly labeled uncalibrated A/B F1."""
    probability = sigmoid(logits)
    return {
        "macro_ap": safe_macro_ap(targets, probability),
        "ab_zero_margin_macro_f1": float(
            f1_score(targets, logits >= 0, average="macro", zero_division=0)
        ),
    }


def _fit_nonnegative_logistic(
    features: np.ndarray,
    target: np.ndarray,
    l2: float,
) -> tuple[np.ndarray, float]:
    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if np.unique(target).size < 2:
        return np.zeros(features.shape[1], dtype=np.float32), float(np.log((target.mean() + 1e-4) / (1 - target.mean() + 1e-4)))
    prior = np.clip(target.mean(), 1e-4, 1 - 1e-4)
    initial = np.concatenate([np.full(features.shape[1], 0.25), [math.log(prior / (1 - prior))]])
    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        weight, bias = parameters[:-1], parameters[-1]
        margin = features @ weight + bias
        probability = 1.0 / (1.0 + np.exp(-np.clip(margin, -30, 30)))
        loss = np.logaddexp(0, margin).mean() - np.mean(target * margin) + l2 * np.sum(weight**2)
        residual = probability - target
        grad_weight = features.T @ residual / len(target) + 2 * l2 * weight
        grad_bias = residual.mean()
        return float(loss), np.concatenate([grad_weight, [grad_bias]])
    fitted = minimize(
        lambda p: objective(p)[0],
        initial,
        jac=lambda p: objective(p)[1],
        method="L-BFGS-B",
        bounds=[(0.0, None)] * features.shape[1] + [(None, None)],
        options={"maxiter": 300},
    )
    if not fitted.success:
        print(f"[B4-P stack] optimizer warning: {fitted.message}")
    return fitted.x[:-1].astype(np.float32), float(fitted.x[-1])


def crossfit_nonnegative_calibrator(
    components: Mapping[str, np.ndarray],
    targets: np.ndarray,
    folds: np.ndarray,
    config: B4PConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    names = list(components)
    feature_cube = np.stack([np.asarray(components[name], dtype=np.float32) for name in names], axis=-1)
    if feature_cube.shape[:2] != targets.shape or not np.isfinite(feature_cube).all():
        raise ValueError("Calibrator components are incomplete or misaligned")
    output = np.full(targets.shape, np.nan, dtype=np.float32)
    records: list[dict[str, Any]] = []
    for fold in sorted(np.unique(folds)):
        train, valid = folds != fold, folds == fold
        for label, label_name in enumerate(FACTOR_LABELS):
            mean = feature_cube[train, label].mean(axis=0)
            scale = feature_cube[train, label].std(axis=0)
            scale = np.where(scale < 1e-4, 1.0, scale)
            x_train = (feature_cube[train, label] - mean) / scale
            x_valid = (feature_cube[valid, label] - mean) / scale
            weight, bias = _fit_nonnegative_logistic(x_train, targets[train, label], config.stack_l2)
            output[valid, label] = x_valid @ weight + bias
            record = {
                "fold": int(fold), "label": label_name, "bias": bias,
                "mean": mean.tolist(), "scale": scale.tolist(),
            }
            record.update({f"weight::{name}": float(weight[i]) for i, name in enumerate(names)})
            records.append(record)
    return output, records


def fit_final_calibrator(
    components_oof: Mapping[str, np.ndarray],
    targets: np.ndarray,
    components_test: Mapping[str, np.ndarray],
    config: B4PConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    names = list(components_oof)
    if set(names) != set(components_test):
        raise ValueError("OOF/test calibrator component names differ")
    train_cube = np.stack([components_oof[name] for name in names], axis=-1).astype(np.float32)
    test_cube = np.stack([components_test[name] for name in names], axis=-1).astype(np.float32)
    output = np.zeros(test_cube.shape[:2], dtype=np.float32)
    records: list[dict[str, Any]] = []
    for label, label_name in enumerate(FACTOR_LABELS):
        mean = train_cube[:, label].mean(axis=0)
        scale = train_cube[:, label].std(axis=0)
        scale = np.where(scale < 1e-4, 1.0, scale)
        weight, bias = _fit_nonnegative_logistic(
            (train_cube[:, label] - mean) / scale,
            targets[:, label],
            config.stack_l2,
        )
        output[:, label] = ((test_cube[:, label] - mean) / scale) @ weight + bias
        record = {"label": label_name, "bias": bias, "mean": mean.tolist(), "scale": scale.tolist()}
        record.update({f"weight::{name}": float(weight[i]) for i, name in enumerate(names)})
        records.append(record)
    return output, records


def evaluate_b4p_oof(
    components: Mapping[str, np.ndarray],
    bundle: b1.DataBundle,
    folds: np.ndarray,
    config: B4PConfig,
    artifact_dir: str | Path,
) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name, logits in components.items():
        metrics, table, thresholds = evaluate_oof_logits(logits, bundle.factor_binary, folds, config)
        metrics = {"experiment": name, **metrics}
        summaries.append(metrics)
        table.to_csv(artifact_dir / f"per_label_{name}.csv", index=False)
        np.save(artifact_dir / f"thresholds_{name}.npy", thresholds)
    stack_logits, weights = crossfit_nonnegative_calibrator(
        components, bundle.factor_binary, folds, config
    )
    stack_metrics, stack_table, stack_thresholds = evaluate_oof_logits(
        stack_logits, bundle.factor_binary, folds, config
    )
    stack_metrics = {"experiment": "B4P_NONNEGATIVE_STACK", **stack_metrics}
    summaries.append(stack_metrics)
    stack_table.to_csv(artifact_dir / "per_label_B4P_NONNEGATIVE_STACK.csv", index=False)
    pd.DataFrame(weights).to_csv(artifact_dir / "crossfit_stack_weights.csv", index=False)
    np.savez_compressed(
        artifact_dir / "B4P_OOF_STACK.npz",
        row_ids=bundle.row_ids,
        logits=stack_logits,
        probabilities=sigmoid(stack_logits),
        thresholds=stack_thresholds,
        folds=folds,
        targets=bundle.factor_binary,
    )
    summary_frame = pd.DataFrame(summaries).sort_values("macro_f1", ascending=False)
    summary_frame.to_csv(artifact_dir / "B4P_OOF_SUMMARY.csv", index=False)
    decision = {
        "version": "B4-P",
        "evaluation": "strict user-and-duplicate-grouped outer OOF",
        "components": list(components),
        "stack": stack_metrics,
        "core_gate": {
            "target_macro_f1": 0.60,
            "stretch_macro_f1": 0.67,
            "passed": bool(stack_metrics["macro_f1"] >= 0.60),
        },
        "config": asdict(config),
    }
    json_dump(decision, artifact_dir / "B4P_DECISION.json")
    return decision


def write_factor_predictions(
    test_corpus: TextCorpus,
    test_probabilities: np.ndarray,
    thresholds: np.ndarray,
    output_dir: str | Path,
    baseline_submission: str | Path | None = None,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = test_probabilities >= np.asarray(thresholds)[None, :]
    factor_lists = [
        [FACTOR_LABELS[label] for label in np.flatnonzero(row)]
        for row in predictions
    ]
    factor_frame = pd.DataFrame(
        {"row_id": test_corpus.row_ids.astype(str), "factors": [repr(items) for items in factor_lists]}
    )
    factor_path = output_dir / "B4P_factor_predictions.csv"
    factor_frame.to_csv(factor_path, index=False)
    probability_frame = pd.DataFrame(test_probabilities, columns=[f"p::{name}" for name in FACTOR_LABELS])
    probability_frame.insert(0, "row_id", test_corpus.row_ids.astype(str))
    probability_path = output_dir / "B4P_factor_probabilities.csv"
    probability_frame.to_csv(probability_path, index=False)
    outputs = {"factor_predictions": str(factor_path), "factor_probabilities": str(probability_path)}
    if baseline_submission:
        baseline_path = Path(baseline_submission)
        if baseline_path.exists():
            baseline = pd.read_csv(baseline_path)
            if "row_id" not in baseline.columns:
                raise KeyError("Baseline submission has no row_id column")
            baseline["row_id"] = baseline["row_id"].astype(str)
            merged = baseline.drop(columns=["factors"], errors="ignore").merge(
                factor_frame, on="row_id", how="left", validate="one_to_one"
            )
            if merged["factors"].isna().any() or len(merged) != len(test_corpus.texts):
                raise AssertionError("Baseline/test row IDs do not align")
            required = ["row_id", "risk_level", "evidence", "factors"]
            missing = [column for column in required if column not in merged.columns]
            if missing:
                raise KeyError(f"Merged submission is missing official fields: {missing}")
            merged = merged[required]
            submission_path = output_dir / "Lenormand.csv"
            merged.to_csv(submission_path, index=False)
            outputs["merged_submission"] = str(submission_path)
    return outputs


def finalize_test_predictions(
    components_oof: Mapping[str, np.ndarray],
    components_test: Mapping[str, np.ndarray],
    bundle: b1.DataBundle,
    folds: np.ndarray,
    test_corpus: TextCorpus,
    config: B4PConfig,
    artifact_dir: str | Path,
    baseline_submission: str | Path | None = None,
) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir)
    stack_oof, _ = crossfit_nonnegative_calibrator(components_oof, bundle.factor_binary, folds, config)
    test_logits, final_weights = fit_final_calibrator(
        components_oof, bundle.factor_binary, components_test, config
    )
    thresholds = fit_factor_thresholds(sigmoid(stack_oof), bundle.factor_binary, config)
    pd.DataFrame(final_weights).to_csv(artifact_dir / "final_stack_weights.csv", index=False)
    np.save(artifact_dir / "final_thresholds.npy", thresholds)
    outputs = write_factor_predictions(
        test_corpus,
        sigmoid(test_logits),
        thresholds,
        artifact_dir,
        baseline_submission=baseline_submission,
    )
    decision = {
        "outputs": outputs,
        "components": list(components_oof),
        "mean_predicted_labels": float((sigmoid(test_logits) >= thresholds).sum(axis=1).mean()),
        "thresholds": {name: float(thresholds[i]) for i, name in enumerate(FACTOR_LABELS)},
    }
    json_dump(decision, artifact_dir / "B4P_TEST_DECISION.json")
    return decision


def load_aligned_component(
    path: str | Path,
    expected_row_ids: Sequence[str],
    probability_epsilon: float = 1e-5,
) -> np.ndarray:
    """Load an optional B3/ModernBERT NPZ as logits with strict row alignment."""
    path = Path(path)
    saved = np.load(path, allow_pickle=True)
    keys = set(saved.files)
    if "row_ids" not in keys:
        raise KeyError(f"Optional component has no row_ids: {path}")
    actual = saved["row_ids"].astype(str).tolist()
    expected = np.asarray(expected_row_ids).astype(str).tolist()
    if actual != expected:
        raise AssertionError(f"Optional component row order mismatch: {path}")
    if "logits" in keys:
        matrix = saved["logits"].astype(np.float32)
    elif "probabilities" in keys:
        probability = np.clip(saved["probabilities"].astype(np.float32), probability_epsilon, 1 - probability_epsilon)
        matrix = np.log(probability / (1 - probability)).astype(np.float32)
    else:
        raise KeyError(f"Optional component must contain logits or probabilities: {path}")
    expected_shape = (len(expected), len(FACTOR_LABELS))
    if matrix.shape != expected_shape or not np.isfinite(matrix).all():
        raise ValueError(f"Optional component shape/values invalid: {matrix.shape}, expected {expected_shape}")
    return matrix
