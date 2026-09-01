"""Deploy the confirmed B16.1 seven-label latent Factor route on test data.

The accepted experiment is a three-fold ensemble, not a newly refitted
full-data adapter.  Each frozen Qwen3.8-27B fold adapter is paired with the
fixed layer-63/C=0.001 probe fitted on that fold's outer-training rows.  Only
the seven preregistered labels are scored; all other Factor cells and all
Task-1 cells are inherited verbatim from the current public-best submission.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

import b1_experiments as b1
import b4p_anchor_verifier as b4
import b7_top8_sprint as b7
import b16_factor_latent_readout as b16
import b161_factor_latent_route as b161


B161_TEST_RUNTIME_REVISION = "2026-08-31.factor-latent-test-ensemble-v1"


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


def sha256(path: str | Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def build_test_prompt_frame(
    bundle: b1.DataBundle,
    folds: np.ndarray,
    test_corpus: b4.TextCorpus,
    train_cache: b4.SemanticCache,
    test_cache: b4.SemanticCache,
    verifier_config: b4.B4PConfig,
    fold: int,
    route_ids: np.ndarray,
) -> pd.DataFrame:
    train_corpus = b4.training_corpus(bundle)
    allowed = np.flatnonzero(np.asarray(folds, dtype=int) != int(fold))
    retriever = b4.FoldRetriever(bundle, train_corpus, train_cache)
    frame = b4.build_prompt_table(
        test_corpus,
        test_cache,
        train_corpus,
        retriever,
        allowed,
        verifier_config,
        query_rows=np.arange(len(test_corpus.texts), dtype=int),
        train_targets=None,
        query_is_training_corpus=False,
    )
    keep = frame.label_idx.astype(int).isin(set(map(int, route_ids)))
    reduced = frame.loc[keep].reset_index(drop=True)
    expected = len(test_corpus.texts) * len(route_ids)
    if len(reduced) != expected:
        raise AssertionError(f"Expected {expected} routed test prompts, got {len(reduced)}")
    return reduced


def _load_q38_test_probability(
    path: str | Path,
    row_ids: Sequence[str],
) -> np.ndarray:
    return b7.load_probability_npz(
        path,
        row_ids,
        preferred_keys=("logits", "verifier_logits", "probabilities"),
    )


def score_test_fold(
    bundle: b1.DataBundle,
    folds: np.ndarray,
    test_corpus: b4.TextCorpus,
    train_cache: b4.SemanticCache,
    test_cache: b4.SemanticCache,
    verifier_config: b4.B4PConfig,
    adapter_path: str | Path,
    probe_path: str | Path,
    q38_test_logits_path: str | Path,
    output_dir: str | Path,
    fold: int,
    route_ids: np.ndarray,
    selected_layer: int = 63,
    extraction_batch_size: int = 2,
    extraction_chunk_size: int = 128,
) -> dict[str, Any]:
    """Score one frozen fold; all expensive hidden chunks are resumable."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "B161_TEST_FOLD_OUTPUTS.npz"
    if result_path.exists():
        saved = np.load(result_path, allow_pickle=True)
        if saved["row_ids"].astype(str).tolist() != test_corpus.row_ids.astype(str).tolist():
            raise AssertionError(f"B16.1 test row mismatch: {result_path}")
        if not np.array_equal(saved["route_ids"].astype(int), np.asarray(route_ids, dtype=int)):
            raise AssertionError(f"B16.1 test route mismatch: {result_path}")
        probability = np.asarray(saved["route_probability"], dtype=np.float32)
        if probability.shape != (len(test_corpus.texts), len(route_ids)):
            raise ValueError(f"Invalid cached B16.1 test shape: {probability.shape}")
        print(f"[B16.1 test resume] fold={fold} {result_path}")
        return {
            "fold": int(fold),
            "probability": probability,
            "q38_reproduction_mae": float(saved["q38_reproduction_mae"][0]),
            "result_path": str(result_path),
        }

    frame = build_test_prompt_frame(
        bundle,
        folds,
        test_corpus,
        train_cache,
        test_cache,
        verifier_config,
        fold,
        route_ids,
    )
    frame.drop(columns=["prompt"]).to_csv(
        output_dir / "B161_TEST_PROMPT_AUDIT.csv", index=False
    )
    extraction_config = b16.FactorLatentConfig(
        fold=int(fold),
        selected_layers=(int(selected_layer),),
        c_grid=(0.001,),
        blend_alphas=(1.0,),
        extraction_batch_size=int(extraction_batch_size),
        extraction_chunk_size=int(extraction_chunk_size),
    )
    extraction = b16.extract_factor_hidden(
        frame,
        adapter_path,
        verifier_config,
        output_dir,
        extraction_config,
    )
    margins = np.asarray(extraction["margins"], dtype=np.float32)
    hidden = np.asarray(extraction["hidden"], dtype=np.float16)
    query = frame.query_row_idx.astype(int).to_numpy()
    label = frame.label_idx.astype(int).to_numpy()
    pair_indices = b161._canonical_pair_indices(
        query,
        label,
        np.arange(len(test_corpus.texts), dtype=int),
        np.asarray(route_ids, dtype=int),
    )
    features = b161._pair_features(hidden, margins, label)[pair_indices]
    probe = joblib.load(probe_path)
    scores = b161._score_route(
        probe, features, len(test_corpus.texts), len(route_ids)
    )
    probability = b4.sigmoid(scores).astype(np.float32)
    route_margins = margins[pair_indices].reshape(len(test_corpus.texts), len(route_ids))
    q38_probability = _load_q38_test_probability(
        q38_test_logits_path, test_corpus.row_ids
    )[:, route_ids]
    reproduction_mae = float(np.mean(np.abs(b4.sigmoid(route_margins) - q38_probability)))
    if reproduction_mae > 0.01:
        raise AssertionError(
            f"B16.1 test prompts do not reproduce Q38 fold {fold}: MAE={reproduction_mae:.5f}"
        )
    np.savez_compressed(
        result_path,
        row_ids=test_corpus.row_ids.astype(str),
        route_ids=np.asarray(route_ids, dtype=np.int16),
        route_probability=probability,
        q38_route_probability=q38_probability.astype(np.float32),
        q38_reproduction_mae=np.asarray([reproduction_mae], dtype=np.float32),
        selected_layer=np.asarray([selected_layer], dtype=np.int16),
    )
    return {
        "fold": int(fold),
        "probability": probability,
        "q38_reproduction_mae": reproduction_mae,
        "result_path": str(result_path),
    }


def load_route_deployment_state(
    bundle: b1.DataBundle,
    anchor: Mapping[str, np.ndarray],
    b16_fold0_output: str | Path,
    b161_root: str | Path,
    route_ids: np.ndarray,
) -> dict[str, Any]:
    """Reconstruct strict OOF route probabilities and fold thresholds."""
    folds = np.asarray(anchor["folds"], dtype=int)
    targets = np.asarray(anchor["targets"], dtype=np.int8)
    route_ids = np.asarray(route_ids, dtype=int)
    probability = np.asarray(anchor["anchor_probability"], dtype=np.float32).copy()
    prediction = np.zeros_like(targets, dtype=np.int8)
    thresholds: dict[int, np.ndarray] = {}
    threshold_config = b4.B4PConfig(
        threshold_kappa_tail=0.0,
        threshold_kappa_mid=2.0,
        threshold_kappa_head=2.0,
        seed=42,
    )

    fold0 = np.load(b16_fold0_output, allow_pickle=True)
    if fold0["row_ids"].astype(str).tolist() != bundle.row_ids.astype(str).tolist():
        raise AssertionError("B16 Fold-0 rows do not align")
    if int(fold0["chosen_layer"][0]) != 63 or abs(float(fold0["chosen_c"][0]) - 0.001) > 1e-12:
        raise AssertionError("B16 Fold-0 probe is not the frozen layer63/C=.001 probe")
    train0 = np.flatnonzero(folds != 0)
    valid0 = np.asarray(fold0["valid_rows"], dtype=int)
    train0_probability = np.asarray(anchor["anchor_probability"], dtype=np.float32)[train0].copy()
    inner0 = b4.sigmoid(np.asarray(fold0["inner_probe_scores"], dtype=np.float32))
    valid0_probe = b4.sigmoid(np.asarray(fold0["valid_probe_scores"], dtype=np.float32))
    train0_probability[:, route_ids] = inner0[:, route_ids]
    probability[valid0[:, None], route_ids] = valid0_probe[:, route_ids]
    thresholds[0] = b4.fit_factor_thresholds(
        train0_probability, targets[train0], threshold_config
    )
    prediction[valid0] = (probability[valid0] >= thresholds[0][None, :]).astype(np.int8)

    b161_root = Path(b161_root)
    for fold in (1, 2):
        saved = np.load(b161_root / f"fold_{fold}/B161_FOLD_OUTPUTS.npz", allow_pickle=True)
        if saved["row_ids"].astype(str).tolist() != bundle.row_ids.astype(str).tolist():
            raise AssertionError(f"B16.1 Fold-{fold} rows do not align")
        if not np.array_equal(saved["route_ids"].astype(int), route_ids):
            raise AssertionError(f"B16.1 Fold-{fold} route differs")
        valid = np.asarray(saved["valid_rows"], dtype=int)
        probability[valid] = np.asarray(saved["route_probability"], dtype=np.float32)
        prediction[valid] = np.asarray(saved["route_prediction"], dtype=np.int8)
        thresholds[fold] = np.asarray(saved["route_thresholds"], dtype=np.float32)

    if not np.isfinite(probability).all():
        raise ValueError("B16.1 OOF deployment probabilities are incomplete")
    metrics, _ = b4.factor_metric_bundle(targets, probability, prediction)
    return {
        "oof_probability": probability,
        "oof_prediction": prediction,
        "thresholds": thresholds,
        "metrics": metrics,
    }


def ensemble_route(
    fold_probabilities: Mapping[int, np.ndarray],
    thresholds: Mapping[int, np.ndarray],
    route_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    route_ids = np.asarray(route_ids, dtype=int)
    if set(fold_probabilities) != {0, 1, 2} or set(thresholds) != {0, 1, 2}:
        raise ValueError("B16.1 test ensemble requires exactly Folds 0, 1, and 2")
    probabilities = np.stack(
        [np.asarray(fold_probabilities[fold], dtype=np.float32) for fold in (0, 1, 2)]
    )
    signed_margins = np.stack(
        [
            b7.logit(probabilities[fold])
            - b7.logit(np.asarray(thresholds[fold], dtype=np.float32)[route_ids])[None, :]
            for fold in (0, 1, 2)
        ]
    )
    mean_margin = signed_margins.mean(axis=0)
    return {
        "fold_probabilities": probabilities.astype(np.float32),
        "fold_signed_margins": signed_margins.astype(np.float32),
        "mean_probability": probabilities.mean(axis=0).astype(np.float32),
        "mean_signed_margin": mean_margin.astype(np.float32),
        "prediction": (mean_margin >= 0).astype(np.int8),
    }


def _factor_matrix(frame: pd.DataFrame) -> np.ndarray:
    label_to_id = {label: index for index, label in enumerate(b1.FACTOR_LABELS)}
    output = np.zeros((len(frame), len(b1.FACTOR_LABELS)), dtype=np.int8)
    for row, raw in enumerate(frame.factors.astype(str)):
        values = ast.literal_eval(raw)
        if not isinstance(values, list):
            raise ValueError(f"Factor cell is not a list at row {row}")
        for value in values:
            if value not in label_to_id:
                raise ValueError(f"Unknown Factor {value!r} at row {row}")
            output[row, label_to_id[value]] = 1
    return output


def build_submission(
    source_submission: str | Path,
    test_frame: pd.DataFrame,
    route_prediction: np.ndarray,
    route_ids: np.ndarray,
    output_dir: str | Path,
) -> dict[str, Any]:
    source_submission = Path(source_submission)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(source_submission, dtype=str, keep_default_na=False)
    before = _factor_matrix(source)
    after = before.copy()
    route_ids = np.asarray(route_ids, dtype=int)
    route_prediction = np.asarray(route_prediction, dtype=np.int8)
    if route_prediction.shape != (len(source), len(route_ids)):
        raise ValueError(f"Invalid routed test prediction shape: {route_prediction.shape}")
    after[:, route_ids] = route_prediction
    candidate = source.copy()
    candidate["factors"] = [
        repr([b1.FACTOR_LABELS[index] for index in np.flatnonzero(row)])
        for row in after
    ]
    if not np.array_equal(
        candidate[["row_id", "risk_level", "evidence"]].astype(str).to_numpy(),
        source[["row_id", "risk_level", "evidence"]].astype(str).to_numpy(),
    ):
        raise AssertionError("B16.1 changed a Task-1 cell")
    path = output_dir / "Lenormand.csv"
    candidate.to_csv(path, index=False)
    reread = pd.read_csv(path, dtype=str, keep_default_na=False)
    audit = b7.audit_submission(reread, test_frame, source)
    changed = before != after
    route_change_rows = []
    for row in np.flatnonzero(np.any(changed[:, route_ids], axis=1)):
        route_change_rows.append(
            {
                "row_id": str(source.row_id.iloc[row]),
                "removed": repr(
                    [
                        b1.FACTOR_LABELS[label]
                        for label in route_ids
                        if before[row, label] and not after[row, label]
                    ]
                ),
                "added": repr(
                    [
                        b1.FACTOR_LABELS[label]
                        for label in route_ids
                        if after[row, label] and not before[row, label]
                    ]
                ),
            }
        )
    pd.DataFrame(route_change_rows).to_csv(
        output_dir / "B161_ROUTE_CHANGES.csv", index=False
    )
    audit.update(
        {
            "runtime_revision": B161_TEST_RUNTIME_REVISION,
            "candidate": "B161_THREEFOLD_LATENT_ROUTE",
            "source_submission": str(source_submission),
            "source_sha256": sha256(source_submission),
            "sha256": sha256(path),
            "path": str(path),
            "task1_cells_identical": True,
            "nonroute_factor_cells_identical": bool(
                np.array_equal(
                    before[:, np.setdiff1d(np.arange(before.shape[1]), route_ids)],
                    after[:, np.setdiff1d(np.arange(after.shape[1]), route_ids)],
                )
            ),
            "changed_route_rows": int(np.any(changed[:, route_ids], axis=1).sum()),
            "changed_route_cells": int(changed[:, route_ids].sum()),
            "mean_factors_before": float(before.sum(axis=1).mean()),
            "mean_factors_after": float(after.sum(axis=1).mean()),
            "status": "READY_TO_UPLOAD",
        }
    )
    if not audit["nonroute_factor_cells_identical"]:
        raise AssertionError("B16.1 changed a non-routed Factor cell")
    json_dump(audit, output_dir / "AUDIT.json")
    return audit

