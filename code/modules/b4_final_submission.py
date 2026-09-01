"""Lenormand B4 final three-fold test inference and submission utilities.

This module contains no large-model training code.  It only:

* scores the leaderboard/test corpus with already trained fold adapters;
* refits the small B4-E2 logistic candidate calibrator on all three OOF folds;
* ensembles the frozen Factor and Task-1 systems; and
* writes/audits the official ``Lenormand.csv`` submission.

Every expensive scoring function is resumable through the chunk caches already
implemented in :mod:`b4p_anchor_verifier`.
"""

from __future__ import annotations

import ast
import gc
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import b1_experiments as b1
import b1_innovation_experiments as inn
import b4p_anchor_verifier as b4
import b4_q38_full_oof as q38oof
import b4_task1_q38 as t1
import b4e_candidate_meta as b4e2
import b4e_evidence_set as b4e


FINAL_RUNTIME_REVISION = "2026-08-25.no-retrain-threefold-test-v1"
OFFICIAL_COLUMNS = ("row_id", "risk_level", "evidence", "factors")
OFFICIAL_RISKS = ("Indicator", "Ideation", "Behavior", "Attempt")


def sha256(path: str | Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_dataclass_config(
    path: str | Path,
    cls: type,
    tuple_fields: Sequence[str] = (),
) -> Any:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for field in tuple_fields:
        if field in payload and payload[field] is not None:
            payload[field] = tuple(payload[field])
    return cls(**payload)


def _aligned_npz_matrix(
    path: str | Path,
    expected_row_ids: Sequence[str],
    key: str = "logits",
) -> np.ndarray:
    path = Path(path)
    saved = np.load(path, allow_pickle=True)
    actual = saved["row_ids"].astype(str).tolist()
    expected = np.asarray(expected_row_ids).astype(str).tolist()
    if actual != expected:
        raise AssertionError(f"row order mismatch: {path}")
    matrix = np.asarray(saved[key], dtype=np.float32)
    if matrix.shape[0] != len(expected) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid matrix in {path}: {matrix.shape}")
    return matrix


# ---------------------------------------------------------------------------
# Factor: Q14/Q38 three-fold test scoring and fixed probability blend
# ---------------------------------------------------------------------------


def score_factor_test_fold(
    adapter_path: str | Path,
    test_corpus: b4.TextCorpus,
    test_cache: b4.SemanticCache,
    train_corpus: b4.TextCorpus,
    train_bundle: b1.DataBundle,
    train_cache: b4.SemanticCache,
    allowed_train_indices: np.ndarray,
    config: b4.B4PConfig,
    output_dir: str | Path,
) -> np.ndarray:
    """Score one frozen Factor adapter, resuming cached logits when possible."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "factor_test_logits.npz"
    if result_path.exists():
        matrix = _aligned_npz_matrix(result_path, test_corpus.row_ids)
        expected = (len(test_corpus.texts), len(b4.FACTOR_LABELS))
        if matrix.shape != expected:
            raise ValueError(f"cached factor shape {matrix.shape} != {expected}")
        print(f"[final factor resume] {result_path}")
        return matrix

    adapter_path = Path(adapter_path)
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(adapter_path)
    matrix, _ = b4.run_verifier_scoring(
        adapter_path,
        test_corpus,
        test_cache,
        train_corpus,
        train_bundle,
        train_cache,
        np.asarray(allowed_train_indices, dtype=int),
        config,
        output_dir / "SCORING",
    )
    expected = (len(test_corpus.texts), len(b4.FACTOR_LABELS))
    if matrix.shape != expected or not np.isfinite(matrix).all():
        raise ValueError(f"incomplete factor test matrix: {matrix.shape}")
    np.savez_compressed(
        result_path,
        row_ids=test_corpus.row_ids.astype(str),
        logits=matrix.astype(np.float32),
    )
    return matrix.astype(np.float32)


def final_factor_predictions(
    bundle: b1.DataBundle,
    folds: np.ndarray,
    test_corpus: b4.TextCorpus,
    q14_oof_path: str | Path,
    q38_oof_path: str | Path,
    q14_test_fold_paths: Sequence[str | Path],
    q38_test_fold_paths: Sequence[str | Path],
    threshold_config: b4.B4PConfig,
    output_dir: str | Path,
    q38_weight: float = 0.75,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Deploy the predeclared 25% Q14 + 75% Q38 probability blend."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    q14_saved = np.load(q14_oof_path, allow_pickle=True)
    if q14_saved["row_ids"].astype(str).tolist() != bundle.row_ids.astype(str).tolist():
        raise AssertionError("Q14 OOF rows do not align with training data")
    if not np.array_equal(q14_saved["folds"].astype(int), np.asarray(folds, dtype=int)):
        raise AssertionError("Q14 OOF folds changed")
    q14_key = "verifier_logits" if "verifier_logits" in q14_saved.files else "logits"
    q14_oof = np.asarray(q14_saved[q14_key], dtype=np.float32)
    q38_oof = _aligned_npz_matrix(q38_oof_path, bundle.row_ids)
    if q14_oof.shape != bundle.factor_binary.shape or q38_oof.shape != bundle.factor_binary.shape:
        raise ValueError("Factor OOF shapes do not match targets")

    oof_probability = (
        (1.0 - q38_weight) * b4.sigmoid(q14_oof)
        + q38_weight * b4.sigmoid(q38_oof)
    )
    thresholds = b4.fit_factor_thresholds(
        oof_probability, bundle.factor_binary, threshold_config
    )

    q14_folds = [
        _aligned_npz_matrix(path, test_corpus.row_ids) for path in q14_test_fold_paths
    ]
    q38_folds = [
        _aligned_npz_matrix(path, test_corpus.row_ids) for path in q38_test_fold_paths
    ]
    if len(q14_folds) != 3 or len(q38_folds) != 3:
        raise ValueError("Final Factor ensemble requires exactly three folds per family")
    q14_probability = np.mean(
        np.stack([b4.sigmoid(matrix) for matrix in q14_folds]), axis=0
    )
    q38_probability = np.mean(
        np.stack([b4.sigmoid(matrix) for matrix in q38_folds]), axis=0
    )
    probability = (1.0 - q38_weight) * q14_probability + q38_weight * q38_probability
    predictions = probability >= thresholds[None, :]
    factor_lists = [
        [b4.FACTOR_LABELS[index] for index in np.flatnonzero(row)]
        for row in predictions
    ]
    frame = pd.DataFrame({
        "row_id": test_corpus.row_ids.astype(str),
        "factors": [repr(items) for items in factor_lists],
    })
    frame.to_csv(output_dir / "factor_predictions.csv", index=False)
    probability_frame = pd.DataFrame(
        probability, columns=[f"p::{label}" for label in b4.FACTOR_LABELS]
    )
    probability_frame.insert(0, "row_id", test_corpus.row_ids.astype(str))
    probability_frame.to_csv(output_dir / "factor_probabilities.csv", index=False)
    np.save(output_dir / "factor_thresholds.npy", thresholds)
    decision = {
        "system": "Q14_25_Q38_75_FIXED_PROBABILITY_BLEND",
        "q38_weight": float(q38_weight),
        "fold_ensemble": "mean_probability_within_family",
        "threshold_fit": "all three-fold OOF with frozen support-adaptive rule",
        "mean_predicted_labels": float(predictions.sum(axis=1).mean()),
        "minimum_predicted_labels": int(predictions.sum(axis=1).min()),
        "maximum_predicted_labels": int(predictions.sum(axis=1).max()),
        "thresholds": {
            label: float(thresholds[index])
            for index, label in enumerate(b4.FACTOR_LABELS)
        },
    }
    b4.json_dump(decision, output_dir / "factor_test_decision.json")
    return frame, decision


# ---------------------------------------------------------------------------
# Task 1: frozen ModernBERT proposer + frozen Q38 adapters
# ---------------------------------------------------------------------------


def _empty_annotation() -> inn.EvidenceAnnotation:
    return inn.EvidenceAnnotation(
        spans=(), has_evidence=False, matched_phrases=0,
        total_phrases=0, exact_phrases=0, fuzzy_phrases=0,
    )


@torch.no_grad()
def prepare_modernbert_proposer_test(
    test_corpus: b4.TextCorpus,
    checkpoint_path: str | Path,
    fold: int,
    output_path: str | Path,
    batch_size: int = 32,
) -> t1.TokenProposalCache:
    """Apply an existing fold-specific ModernBERT proposer to test posts."""

    output_path = Path(output_path)
    metrics_path = output_path.with_suffix(".json")
    if output_path.exists():
        saved = np.load(output_path, allow_pickle=True)
        if saved["row_ids"].astype(str).tolist() != test_corpus.row_ids.astype(str).tolist():
            raise AssertionError(f"test proposer row mismatch: {output_path}")
        rows = saved["rows"].astype(int)
        return t1.TokenProposalCache(
            probabilities={int(row): saved["token_probabilities"][i] for i, row in enumerate(rows)},
            offsets={int(row): saved["token_offsets"][i] for i, row in enumerate(rows)},
            metrics=(json.loads(metrics_path.read_text()) if metrics_path.exists() else {}),
        )

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    config = inn.RiskEvidenceConfig(
        name=f"TASK1_MODERNBERT_PROPOSER_FOLD{fold}",
        model_name="answerdotai/ModernBERT-base",
        max_length=512,
        epochs=2,
        eval_batch_size=batch_size,
        n_splits=3,
        use_ordinal=True,
        use_evidence=True,
        condition_risk_on_evidence=True,
        use_counterfactual=True,
        save_checkpoints=True,
        resume=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_bundle = SimpleNamespace(
        texts=list(test_corpus.texts),
        risk_ids=np.zeros(len(test_corpus.texts), dtype=np.int64),
    )
    annotations = [_empty_annotation() for _ in test_corpus.texts]
    dataset = inn.RiskEvidenceDataset(
        dummy_bundle, annotations, np.arange(len(test_corpus.texts))
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.type == "cuda",
        collate_fn=inn.RiskEvidenceCollator(tokenizer, config.max_length),
    )
    model = inn.RiskEvidenceJointModel(config).to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True),
        strict=True,
    )
    model.eval()
    probabilities: dict[int, np.ndarray] = {}
    offsets: dict[int, np.ndarray] = {}
    for raw_batch in loader:
        batch = b1.move_batch(raw_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(batch)
        scores = torch.sigmoid(output["evidence_logits"]).cpu().numpy()
        valid = batch["token_valid"].cpu().numpy().astype(bool)
        batch_offsets = batch["offset_mapping"].cpu().numpy()
        for local, row in enumerate(batch["idx"].cpu().numpy()):
            keep = valid[local]
            probabilities[int(row)] = scores[local][keep].astype(np.float32)
            offsets[int(row)] = batch_offsets[local][keep].astype(np.int32)
    rows = np.arange(len(test_corpus.texts), dtype=int)
    if set(probabilities) != set(rows.tolist()):
        raise AssertionError("ModernBERT test proposer is incomplete")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        row_ids=test_corpus.row_ids.astype(str),
        rows=rows,
        token_probabilities=np.asarray([probabilities[i] for i in rows], dtype=object),
        token_offsets=np.asarray([offsets[i] for i in rows], dtype=object),
    )
    metrics = {
        "fold": int(fold),
        "rows": int(len(rows)),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    del model, loader, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return t1.TokenProposalCache(probabilities, offsets, metrics)


def run_task1_test_fold(
    test_corpus: b4.TextCorpus,
    adapter_path: str | Path,
    lexicon: t1.EvidenceLexicon,
    config: t1.Task1Full64Config,
    token_proposals: t1.TokenProposalCache,
    output_dir: str | Path,
) -> dict[str, str]:
    """Run one frozen Task-1 fold on test with fully resumable score chunks."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "q38_task1_test_outputs.npz"
    audit_path = output_dir / "evidence_candidate_audit.csv"
    prediction_path = output_dir / "test_predictions.csv"
    if npz_path.exists() and audit_path.exists() and prediction_path.exists():
        saved = np.load(npz_path, allow_pickle=True)
        if saved["row_ids"].astype(str).tolist() != test_corpus.row_ids.astype(str).tolist():
            raise AssertionError(f"Task1 test row mismatch: {npz_path}")
        print(f"[final Task1 resume] fold={config.fold} {output_dir}")
        return {
            "outputs": str(npz_path), "audit": str(audit_path),
            "predictions": str(prediction_path),
        }

    adapter_path = Path(adapter_path)
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(adapter_path)
    rows = np.arange(len(test_corpus.texts), dtype=int)
    b4_config = config.b4_config()
    model, tokenizer = b4.load_quantized_causal_model(
        config.model_name,
        adapter_path=adapter_path,
        training=False,
        attention_implementation=config.attention_implementation,
        qwen35_fa2_position_guard=config.qwen35_fa2_position_guard,
        require_qwen35_fast_kernels=config.require_qwen35_fast_kernels,
    )

    document_frame = t1.build_risk_prompt_frame(
        test_corpus, rows, config, evidence_by_row=None, stage="test-document"
    )
    document_margins = b4.score_prompts_cached(
        model, tokenizer, document_frame,
        output_dir / "risk_document_scores", b4_config,
        config.score_batch_size, use_chat_template=True,
    )
    document_matrix = t1._risk_matrix(
        document_frame, document_margins, len(test_corpus.texts)
    )
    document_prediction = np.argmax(document_matrix, axis=1)

    evidence_frame = t1.build_evidence_prompt_frame(
        test_corpus, rows, document_prediction, lexicon, config,
        token_proposals=token_proposals,
    )
    evidence_margins = b4.score_prompts_cached(
        model, tokenizer, evidence_frame,
        output_dir / "evidence_scores", b4_config,
        config.score_batch_size, use_chat_template=True,
    )
    selected, evidence_audit = t1.select_evidence(
        evidence_frame, evidence_margins, config
    )
    evidence_audit["adapter_fold"] = int(config.fold)

    conditioned_frame = t1.build_risk_prompt_frame(
        test_corpus, rows, config,
        evidence_by_row=selected, stage="test-conditioned",
    )
    conditioned_margins = b4.score_prompts_cached(
        model, tokenizer, conditioned_frame,
        output_dir / "risk_conditioned_scores", b4_config,
        config.score_batch_size, use_chat_template=True,
    )
    conditioned_matrix = t1._risk_matrix(
        conditioned_frame, conditioned_margins, len(test_corpus.texts)
    )
    conditioned_prediction = np.argmax(conditioned_matrix, axis=1)
    blended_matrix = (
        (1.0 - config.conditioned_risk_blend_weight) * document_matrix
        + config.conditioned_risk_blend_weight * conditioned_matrix
    )
    blended_prediction = np.argmax(blended_matrix, axis=1)

    evidence_audit.to_csv(audit_path, index=False)
    pd.DataFrame({
        "row_id": test_corpus.row_ids.astype(str),
        "document_risk": [b1.RISK_LABELS[i] for i in document_prediction],
        "conditioned_risk": [b1.RISK_LABELS[i] for i in conditioned_prediction],
        "blended_risk": [b1.RISK_LABELS[i] for i in blended_prediction],
        "predicted_evidence": [
            json.dumps(selected.get(int(row), []), ensure_ascii=False)
            for row in rows
        ],
    }).to_csv(prediction_path, index=False)
    np.savez_compressed(
        npz_path,
        row_ids=test_corpus.row_ids.astype(str),
        document_risk_margins=document_matrix.astype(np.float32),
        conditioned_risk_margins=conditioned_matrix.astype(np.float32),
        blended_risk_margins=blended_matrix.astype(np.float32),
    )
    del model, tokenizer
    b4.unload_model()
    return {
        "outputs": str(npz_path), "audit": str(audit_path),
        "predictions": str(prediction_path),
    }


# ---------------------------------------------------------------------------
# B4-E2 all-OOF refit and test decoding
# ---------------------------------------------------------------------------


def fit_all_oof_meta_calibrator(
    fold_artifacts: Mapping[int, tuple[str | Path, str | Path]],
    bundle: b1.DataBundle,
    folds: np.ndarray,
    config: b4e2.CandidateMetaConfig,
    output_dir: str | Path,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    """Fit the final small logistic calibrator on Fold0+Fold1+Fold2 OOF rows."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    row_to_index = {str(row): i for i, row in enumerate(bundle.row_ids)}
    for fold in sorted(fold_artifacts):
        audit_path, prediction_path = map(Path, fold_artifacts[fold])
        audit, validation = b4e.prepare_artifacts(
            pd.read_csv(audit_path), pd.read_csv(prediction_path), bundle
        )
        indices = np.asarray([row_to_index[str(row)] for row in validation.row_id])
        if not np.all(np.asarray(folds)[indices] == int(fold)):
            raise AssertionError(f"OOF meta artifact is not fold {fold}")
        expected = set(bundle.row_ids[np.asarray(folds) == int(fold)].astype(str))
        if set(validation.row_id.astype(str)) != expected:
            raise AssertionError(f"OOF meta coverage mismatch for fold {fold}")
        table = b4e2.build_candidate_meta_table(
            audit, validation, bundle, include_targets=True
        )
        table["outer_fold"] = int(fold)
        tables.append(table)
        manifests.append({
            "fold": int(fold),
            "rows": int(len(validation)),
            "candidates": int(len(table)),
            "positive_candidates": int(table.candidate_target.sum()),
            "audit_sha256": sha256(audit_path),
            "prediction_sha256": sha256(prediction_path),
        })
    if set(fold_artifacts) != set(np.unique(folds).astype(int).tolist()):
        raise AssertionError("Final meta refit requires every outer fold")
    candidate_table = pd.concat(tables, ignore_index=True)
    if set(candidate_table.row_id.astype(str)) != set(bundle.row_ids.astype(str)):
        raise AssertionError("Final meta candidate rows do not cover all training posts")
    model = b4e2.fit_meta_model(
        candidate_table, bundle.row_ids.astype(str).tolist(), config
    )
    model_path = output_dir / "B4E2_FINAL_ALL_OOF_META_CALIBRATOR.joblib"
    config_path = output_dir / "B4E2_FINAL_FROZEN_CONFIG.json"
    joblib.dump(model, model_path)
    config_path.write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(manifests).to_csv(
        output_dir / "B4E2_FINAL_OOF_MANIFEST.csv", index=False
    )
    decision = {
        "runtime_revision": FINAL_RUNTIME_REVISION,
        "model": "sklearn LogisticRegression candidate relevance calibrator",
        "training_scope": "all three outer-OOF candidate tables",
        "training_posts": int(len(bundle.texts)),
        "candidate_rows": int(len(candidate_table)),
        "positive_candidate_rate": float(candidate_table.candidate_target.mean()),
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        "config": asdict(config),
        "fold_manifest": manifests,
    }
    b4.json_dump(decision, output_dir / "B4E2_FINAL_REFIT_DECISION.json")
    return model, candidate_table, decision


def _aggregate_candidate_audits(
    fold_audits: Sequence[pd.DataFrame],
    length_penalty: float,
) -> pd.DataFrame:
    table = pd.concat(fold_audits, ignore_index=True)
    table["query_row_idx"] = table.query_row_idx.astype(int)
    table["left"] = table.left.astype(int)
    table["right"] = table.right.astype(int)
    table["margin"] = pd.to_numeric(table.margin, errors="coerce")
    table = table[np.isfinite(table.margin)].copy()
    table["normalized_candidate"] = table.candidate.map(b4e._normalize)
    table["candidate_tokens"] = table.normalized_candidate.str.split().map(len).astype(int)
    keys = ["query_row_idx", "left", "right", "normalized_candidate"]
    records: list[dict[str, Any]] = []
    for _, group in table.groupby(keys, sort=False, dropna=False):
        representative = group.loc[group.margin.idxmax()].to_dict()
        representative["margin"] = float(group.margin.mean())
        representative["fold_support"] = int(group.adapter_fold.nunique())
        representative["candidate_tokens"] = int(
            len(str(representative["normalized_candidate"]).split())
        )
        representative["selection_score"] = (
            representative["margin"]
            - float(length_penalty) * len(str(representative["candidate"]))
        )
        representative["selected"] = False
        records.append(representative)
    output = pd.DataFrame(records)
    return output.sort_values(
        ["query_row_idx", "margin"], ascending=[True, False], kind="mergesort"
    ).reset_index(drop=True)


def assemble_task1_test_predictions(
    test_corpus: b4.TextCorpus,
    fold_dirs: Sequence[str | Path],
    model: Any,
    meta_config: b4e2.CandidateMetaConfig,
    task1_config: t1.Task1Full64Config,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ensemble risk margins, candidate margins, and apply final B4-E2."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    document_matrices: list[np.ndarray] = []
    conditioned_matrices: list[np.ndarray] = []
    audits: list[pd.DataFrame] = []
    for fold, fold_dir_value in enumerate(fold_dirs):
        fold_dir = Path(fold_dir_value)
        saved = np.load(fold_dir / "q38_task1_test_outputs.npz", allow_pickle=True)
        if saved["row_ids"].astype(str).tolist() != test_corpus.row_ids.astype(str).tolist():
            raise AssertionError(f"Task1 test fold rows changed: {fold_dir}")
        document_matrices.append(saved["document_risk_margins"].astype(np.float32))
        conditioned_matrices.append(saved["conditioned_risk_margins"].astype(np.float32))
        audit = pd.read_csv(fold_dir / "evidence_candidate_audit.csv")
        audit["adapter_fold"] = int(fold)
        audits.append(audit)
    if len(document_matrices) != 3:
        raise ValueError("Final Task1 ensemble requires exactly three folds")
    document = np.mean(np.stack(document_matrices), axis=0)
    conditioned = np.mean(np.stack(conditioned_matrices), axis=0)
    blended = (
        (1.0 - task1_config.conditioned_risk_blend_weight) * document
        + task1_config.conditioned_risk_blend_weight * conditioned
    )
    document_prediction = document.argmax(axis=1)
    conditioned_prediction = conditioned.argmax(axis=1)
    blended_prediction = blended.argmax(axis=1)

    risk_frame = pd.DataFrame({
        "row_id": test_corpus.row_ids.astype(str),
        "user_id": test_corpus.user_ids.astype(str),
        "document_risk": [b1.RISK_LABELS[i] for i in document_prediction],
        "conditioned_risk": [b1.RISK_LABELS[i] for i in conditioned_prediction],
        "blended_risk": [b1.RISK_LABELS[i] for i in blended_prediction],
    })
    row_id_by_index = dict(enumerate(test_corpus.row_ids.astype(str)))

    # Preserve the candidate-set distribution seen by the validated OOF
    # calibrator: calibrate each fold's 64-candidate table independently first.
    fold_candidate_tables: list[pd.DataFrame] = []
    for fold, audit in enumerate(audits):
        audit = audit.copy()
        audit["row_id"] = audit.query_row_idx.astype(int).map(row_id_by_index)
        if audit.row_id.isna().any():
            raise AssertionError(f"Task1 fold {fold} contains invalid row indices")
        audit["normalized_candidate"] = audit.candidate.map(b4e._normalize)
        audit["candidate_tokens"] = (
            audit.normalized_candidate.str.split().map(len).astype(int)
        )
        raw_selected, _ = t1.select_evidence(
            audit, audit.margin.to_numpy(), task1_config
        )
        fold_validation = risk_frame.copy()
        fold_validation["baseline_evidence"] = [
            raw_selected.get(index, []) for index in range(len(test_corpus.texts))
        ]
        candidate_table = b4e2.build_candidate_meta_table(
            audit, fold_validation, test_corpus, include_targets=False
        )
        candidate_table["adapter_fold"] = int(fold)
        candidate_table["meta_probability"] = model.predict_proba(
            candidate_table[list(b4e2.MODEL_FEATURES)]
        )[:, 1]
        fold_candidate_tables.append(candidate_table)

    calibrated = pd.concat(fold_candidate_tables, ignore_index=True)
    calibrated.to_csv(
        output_dir / "task1_test_candidate_meta_by_fold.csv", index=False
    )
    calibrated["normalized_candidate"] = calibrated.candidate.map(b4e._normalize)
    keys = ["row_id", "left", "right", "normalized_candidate"]
    records: list[dict[str, Any]] = []
    for _, group in calibrated.groupby(keys, sort=False, dropna=False):
        representative = group.loc[group.meta_probability.idxmax()].to_dict()
        representative["meta_probability"] = float(group.meta_probability.mean())
        representative["margin"] = float(group.margin.mean())
        representative["fold_support"] = int(group.adapter_fold.nunique())
        records.append(representative)
    candidate_table = pd.DataFrame(records).sort_values(
        ["row_id", "meta_probability"],
        ascending=[True, False], kind="mergesort",
    ).reset_index(drop=True)
    validation = risk_frame.copy()
    validation["baseline_evidence"] = [[] for _ in range(len(validation))]
    probabilities = candidate_table.meta_probability.to_numpy(dtype=float)
    evidence_by_row = b4e2.decode_meta_probabilities(
        candidate_table, validation, probabilities, meta_config
    )
    candidate_table.to_csv(
        output_dir / "task1_test_candidate_meta_audit.csv", index=False
    )
    evidence_lists = [
        list(evidence_by_row[str(row_id)]) for row_id in test_corpus.row_ids.astype(str)
    ]
    frame = pd.DataFrame({
        "row_id": test_corpus.row_ids.astype(str),
        "risk_level": [b1.RISK_LABELS[i] for i in blended_prediction],
        "evidence_list": [json.dumps(items, ensure_ascii=False) for items in evidence_lists],
        "evidence": [serialize_evidence(items) for items in evidence_lists],
    })
    frame.to_csv(output_dir / "task1_test_predictions.csv", index=False)
    risk_counts = frame.risk_level.value_counts().reindex(OFFICIAL_RISKS, fill_value=0)
    decision = {
        "system": "three-fold Qwen3.8-27B risk/evidence + all-OOF B4-E2",
        "fold_ensemble": "mean A/B margins",
        "candidate_ensemble": (
            "foldwise B4-E2 calibration, then exact-span mean available-fold "
            "meta probability"
        ),
        "rows": int(len(frame)),
        "risk_counts": {name: int(risk_counts[name]) for name in OFFICIAL_RISKS},
        "mean_evidence_phrases": float(np.mean([len(items) for items in evidence_lists])),
        "indicator_nonempty_evidence": int(sum(
            bool(items) and risk == "Indicator"
            for items, risk in zip(evidence_lists, frame.risk_level)
        )),
        "mean_candidate_fold_support": float(candidate_table.fold_support.mean()),
    }
    b4.json_dump(decision, output_dir / "task1_test_decision.json")
    return frame, decision


def serialize_evidence(items: Sequence[str]) -> str:
    """Official semicolon serialization while preserving verbatim substrings."""

    pieces: list[str] = []
    seen: set[str] = set()
    for item in items:
        # A semicolon is the official delimiter.  Splitting an extracted span
        # at an existing semicolon preserves literal text on either side.
        for piece in str(item).split(";"):
            piece = piece.strip()
            normalized = b4e._normalize(piece)
            if piece and normalized not in seen:
                pieces.append(piece)
                seen.add(normalized)
    return "; ".join(pieces)


# ---------------------------------------------------------------------------
# Official submission audit
# ---------------------------------------------------------------------------


def build_and_audit_submission(
    test_corpus: b4.TextCorpus,
    task1_frame: pd.DataFrame,
    factor_frame: pd.DataFrame,
    output_path: str | Path,
) -> dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expected_rows = test_corpus.row_ids.astype(str).tolist()
    task1 = task1_frame.copy()
    factor = factor_frame.copy()
    task1["row_id"] = task1.row_id.astype(str)
    factor["row_id"] = factor.row_id.astype(str)
    if task1.row_id.tolist() != expected_rows or factor.row_id.tolist() != expected_rows:
        raise AssertionError("Task1/Factor predictions do not preserve leaderboard row order")
    submission = task1[["row_id", "risk_level", "evidence"]].merge(
        factor[["row_id", "factors"]], on="row_id", how="inner", validate="one_to_one"
    )
    submission = submission[list(OFFICIAL_COLUMNS)]
    if len(submission) != len(expected_rows) or submission.row_id.duplicated().any():
        raise AssertionError("submission row coverage/uniqueness failed")
    if not set(submission.risk_level).issubset(OFFICIAL_RISKS):
        raise ValueError("submission contains an invalid risk level")

    allowed_factors = set(b4.FACTOR_LABELS)
    factor_lists: list[list[str]] = []
    for value in submission.factors:
        parsed = ast.literal_eval(str(value))
        if not isinstance(parsed, list) or not set(parsed).issubset(allowed_factors):
            raise ValueError(f"invalid Factor list: {value}")
        if len(parsed) != len(set(parsed)):
            raise ValueError(f"duplicate Factor label: {value}")
        factor_lists.append(parsed)

    text_by_row = dict(zip(test_corpus.row_ids.astype(str), test_corpus.texts))
    verbatim_failures: list[dict[str, str]] = []
    indicator_nonempty = 0
    evidence_counts: list[int] = []
    for row in submission.itertuples(index=False):
        phrases = [item.strip() for item in str(row.evidence or "").split(";") if item.strip()]
        evidence_counts.append(len(phrases))
        if row.risk_level == "Indicator" and phrases:
            indicator_nonempty += 1
        for phrase in phrases:
            if phrase not in text_by_row[row.row_id]:
                verbatim_failures.append({"row_id": row.row_id, "phrase": phrase})
    if indicator_nonempty:
        raise AssertionError(f"{indicator_nonempty} Indicator rows have non-empty evidence")
    if verbatim_failures:
        pd.DataFrame(verbatim_failures).to_csv(
            output_path.with_name("VERBATIM_FAILURES.csv"), index=False
        )
        raise AssertionError(f"{len(verbatim_failures)} evidence phrases are not verbatim")

    submission["evidence"] = submission.evidence.fillna("").astype(str)
    submission.to_csv(output_path, index=False)
    reread = pd.read_csv(output_path, keep_default_na=False)
    if reread.columns.tolist() != list(OFFICIAL_COLUMNS):
        raise AssertionError("serialized submission columns changed")
    if reread.row_id.astype(str).tolist() != expected_rows:
        raise AssertionError("serialized submission row order changed")
    audit = {
        "runtime_revision": FINAL_RUNTIME_REVISION,
        "official_columns": list(OFFICIAL_COLUMNS),
        "rows": int(len(submission)),
        "unique_rows": int(submission.row_id.nunique()),
        "risk_counts": {
            name: int((submission.risk_level == name).sum()) for name in OFFICIAL_RISKS
        },
        "mean_evidence_phrases": float(np.mean(evidence_counts)),
        "mean_factors": float(np.mean([len(items) for items in factor_lists])),
        "indicator_nonempty_evidence": int(indicator_nonempty),
        "verbatim_failures": int(len(verbatim_failures)),
        "submission_path": str(output_path),
        "submission_sha256": sha256(output_path),
        "status": "READY_TO_UPLOAD",
    }
    b4.json_dump(audit, output_path.with_name("FINAL_SUBMISSION_AUDIT.json"))
    return audit
