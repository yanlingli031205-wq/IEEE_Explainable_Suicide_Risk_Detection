"""B16: hidden-state readout gate for the Qwen3.8-27B Factor verifier.

The existing verifier reduces every post/label judgment to one A-vs-B logit.
B16 tests whether the last decoder states contain useful label evidence that
the language-model head discards.  The readout is deliberately small: one
shared, label-aware logistic probe trained on outer-train rows.  Validation
posts, users, labels, and thresholds never enter probe fitting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import b1_experiments as b1
import b4p_anchor_verifier as b4
import b15_latent_readout as b15


B16_RUNTIME_REVISION = "2026-08-30.factor-hidden-readout-gate-v1"


@dataclass
class FactorLatentConfig:
    fold: int = 0
    selected_layers: tuple[int, ...] = (47, 63)
    c_grid: tuple[float, ...] = (0.001, 0.01, 0.1)
    blend_alphas: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
    inner_splits: int = 3
    extraction_batch_size: int = 2
    extraction_chunk_size: int = 128
    seed: int = 42
    gate_macro_f1: float = 0.008
    gate_macro_ap: float = 0.004
    gate_tail_floor: float = -0.005


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


def load_full64_config(path: str | Path) -> b4.B4PConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = payload.get("config", payload)
    allowed = {item.name for item in fields(b4.B4PConfig)}
    payload = {key: value for key, value in payload.items() if key in allowed}
    for key in (
        "lora_target_leaves",
        "candidate_caps_for_audit",
    ):
        if key in payload and payload[key] is not None:
            payload[key] = tuple(payload[key])
    config = b4.B4PConfig(**payload)
    if not config.verifier_use_chat_template:
        raise ValueError("B16 v1 requires the Full64 chat-template verifier")
    return config


def load_factor_anchor(
    bundle: b1.DataBundle,
    q14_path: str | Path,
    q38_path: str | Path,
    q38_weight: float = 0.75,
) -> dict[str, np.ndarray]:
    q14_path, q38_path = Path(q14_path), Path(q38_path)
    q14, q38 = np.load(q14_path, allow_pickle=True), np.load(q38_path, allow_pickle=True)
    for saved, path in ((q14, q14_path), (q38, q38_path)):
        if saved["row_ids"].astype(str).tolist() != bundle.row_ids.astype(str).tolist():
            raise AssertionError(f"Factor OOF row mismatch: {path}")
        if not np.array_equal(saved["targets"].astype(np.int8), bundle.factor_binary.astype(np.int8)):
            raise AssertionError(f"Factor OOF target mismatch: {path}")
    folds = q14["folds"].astype(int)
    if not np.array_equal(folds, q38["folds"].astype(int)):
        raise AssertionError("Q14/Q38 fold mismatch")
    q14_key = "verifier_logits" if "verifier_logits" in q14.files else "logits"
    p14 = b4.sigmoid(np.asarray(q14[q14_key], dtype=np.float32))
    p38 = b4.sigmoid(np.asarray(q38["logits"], dtype=np.float32))
    probability = (1.0 - float(q38_weight)) * p14 + float(q38_weight) * p38
    return {
        "folds": folds,
        "targets": bundle.factor_binary.astype(np.int8),
        "q14_probability": p14.astype(np.float32),
        "q38_probability": p38.astype(np.float32),
        "anchor_probability": probability.astype(np.float32),
    }


def build_all_prompts(
    bundle: b1.DataBundle,
    folds: np.ndarray,
    semantic_cache: b4.SemanticCache,
    verifier_config: b4.B4PConfig,
    fold: int,
) -> pd.DataFrame:
    corpus = b4.training_corpus(bundle)
    allowed = np.flatnonzero(np.asarray(folds, dtype=int) != int(fold))
    retriever = b4.FoldRetriever(bundle, corpus, semantic_cache)
    frame = b4.build_prompt_table(
        corpus,
        semantic_cache,
        corpus,
        retriever,
        allowed,
        verifier_config,
        query_rows=np.arange(len(bundle.texts), dtype=int),
        train_targets=bundle.factor_binary,
        query_is_training_corpus=True,
    )
    expected = len(bundle.texts) * len(b1.FACTOR_LABELS)
    if len(frame) != expected:
        raise AssertionError(f"Expected {expected} factor prompts, got {len(frame)}")
    if frame.label.drop_duplicates().astype(str).tolist() != list(b1.FACTOR_LABELS):
        raise AssertionError("Prompt label order differs from b1.FACTOR_LABELS")
    return frame


def extract_factor_hidden(
    prompt_frame: pd.DataFrame,
    adapter_path: str | Path,
    verifier_config: b4.B4PConfig,
    output_dir: str | Path,
    config: FactorLatentConfig,
) -> dict[str, Any]:
    model, tokenizer = b4.load_quantized_causal_model(
        verifier_config.verifier_model,
        adapter_path=adapter_path,
        training=False,
        attention_implementation=verifier_config.attention_implementation,
        qwen35_fa2_position_guard=verifier_config.qwen35_fa2_position_guard,
        require_qwen35_fast_kernels=verifier_config.require_qwen35_fast_kernels,
    )
    latent_config = b15.LatentReadoutConfig(
        fold=int(config.fold),
        selected_layers=tuple(map(int, config.selected_layers)),
        max_length=int(verifier_config.max_length),
        extraction_batch_size=int(config.extraction_batch_size),
        extraction_chunk_size=int(config.extraction_chunk_size),
    )
    result = b15.extract_hidden_states_cached(
        model,
        tokenizer,
        prompt_frame,
        Path(output_dir) / "HIDDEN_CACHE",
        latent_config,
        adapter_path,
    )
    del model, tokenizer
    b4.unload_model()
    return result


def _to_factor_cubes(
    frame: pd.DataFrame,
    extraction: dict[str, Any],
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.asarray(extraction["hidden"], dtype=np.float16)
    margins = np.asarray(extraction["margins"], dtype=np.float32)
    labels = len(b1.FACTOR_LABELS)
    cube = np.full((n_rows, labels, hidden.shape[1], hidden.shape[2]), np.nan, dtype=np.float16)
    matrix = np.full((n_rows, labels), np.nan, dtype=np.float32)
    for index, row in enumerate(frame.itertuples(index=False)):
        query, label = int(row.query_row_idx), int(row.label_idx)
        if np.isfinite(matrix[query, label]):
            raise AssertionError(f"Duplicate factor prompt row={query}, label={label}")
        matrix[query, label] = margins[index]
        cube[query, label] = hidden[index]
    if not np.isfinite(matrix).all() or not np.isfinite(cube).all():
        raise ValueError("Factor hidden extraction is incomplete")
    return matrix, cube


def _features(cube: np.ndarray, margins: np.ndarray, rows: np.ndarray, layer_position: int) -> np.ndarray:
    rows = np.asarray(rows, dtype=int)
    values = np.asarray(cube[rows, :, int(layer_position), :], dtype=np.float32)
    flattened = values.reshape(-1, values.shape[-1])
    margin = np.asarray(margins[rows], dtype=np.float32).reshape(-1, 1)
    one_hot = np.tile(np.eye(len(b1.FACTOR_LABELS), dtype=np.float32), (len(rows), 1))
    return np.concatenate([flattened, margin, one_hot], axis=1)


def _targets(targets: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return np.asarray(targets[np.asarray(rows, dtype=int)], dtype=np.int8).reshape(-1)


def _sample_weights(targets: np.ndarray, rows: np.ndarray) -> np.ndarray:
    matrix = np.asarray(targets[np.asarray(rows, dtype=int)], dtype=np.int8)
    weights = np.zeros_like(matrix, dtype=np.float64)
    for label in range(matrix.shape[1]):
        positive = matrix[:, label] == 1
        negative = ~positive
        weights[positive, label] = 0.5 / max(int(positive.sum()), 1)
        weights[negative, label] = 0.5 / max(int(negative.sum()), 1)
    weights *= len(matrix) * matrix.shape[1] / max(weights.sum(), 1e-12)
    return weights.reshape(-1).astype(np.float32)


def _make_probe(c_value: float, seed: int) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(c_value),
            solver="lbfgs",
            max_iter=300,
            random_state=int(seed),
        ),
    )


def _fit_probe(model: Any, x: np.ndarray, targets: np.ndarray, rows: np.ndarray) -> Any:
    model.fit(
        x,
        _targets(targets, rows),
        logisticregression__sample_weight=_sample_weights(targets, rows),
    )
    return model


def _score_matrix(model: Any, x: np.ndarray, n_rows: int) -> np.ndarray:
    score = np.asarray(model.decision_function(x), dtype=np.float32)
    return score.reshape(int(n_rows), len(b1.FACTOR_LABELS))


def _macro_ap(target: np.ndarray, probability: np.ndarray) -> float:
    values = [
        average_precision_score(target[:, label], probability[:, label])
        for label in range(target.shape[1])
        if np.unique(target[:, label]).size == 2
    ]
    return float(np.mean(values))


def crossfit_probe_selection(
    cube: np.ndarray,
    margins: np.ndarray,
    layers: Sequence[int],
    targets: np.ndarray,
    user_ids: np.ndarray,
    train_rows: np.ndarray,
    config: FactorLatentConfig,
) -> tuple[int, float, np.ndarray, pd.DataFrame]:
    train_rows = np.asarray(train_rows, dtype=int)
    groups = np.asarray(user_ids).astype(str)[train_rows]
    splits = min(int(config.inner_splits), len(np.unique(groups)))
    splitter = GroupKFold(n_splits=splits)
    records: list[dict[str, Any]] = []
    score_cache: dict[tuple[int, float], np.ndarray] = {}
    for position, layer in enumerate(map(int, layers)):
        x = _features(cube, margins, train_rows, position)
        for c_value in config.c_grid:
            oof = np.full((len(train_rows), len(b1.FACTOR_LABELS)), np.nan, dtype=np.float32)
            for split_id, (inner_train, inner_valid) in enumerate(
                splitter.split(train_rows, groups=groups)
            ):
                row_train = train_rows[inner_train]
                pair_train = np.concatenate([
                    np.arange(index * len(b1.FACTOR_LABELS), (index + 1) * len(b1.FACTOR_LABELS))
                    for index in inner_train
                ])
                pair_valid = np.concatenate([
                    np.arange(index * len(b1.FACTOR_LABELS), (index + 1) * len(b1.FACTOR_LABELS))
                    for index in inner_valid
                ])
                probe = _make_probe(float(c_value), config.seed + split_id)
                _fit_probe(probe, x[pair_train], targets, row_train)
                oof[inner_valid] = _score_matrix(probe, x[pair_valid], len(inner_valid))
            if not np.isfinite(oof).all():
                raise ValueError("Incomplete B16 inner OOF probe")
            probability = b4.sigmoid(oof)
            metric = _macro_ap(targets[train_rows], probability)
            records.append({"layer": layer, "layer_position": position, "c": float(c_value), "inner_macro_ap": metric})
            score_cache[(layer, float(c_value))] = oof
            print(f"[B16 probe] layer={layer} C={c_value:g} inner-mAP={metric:.4f}")
    audit = pd.DataFrame(records).sort_values(
        ["inner_macro_ap", "layer", "c"], ascending=[False, False, True], kind="mergesort"
    )
    winner = audit.iloc[0]
    key = (int(winner.layer), float(winner.c))
    return key[0], key[1], score_cache[key], audit


def run_fold_gate(
    bundle: b1.DataBundle,
    anchor: dict[str, np.ndarray],
    semantic_cache: b4.SemanticCache,
    verifier_config: b4.B4PConfig,
    adapter_path: str | Path,
    output_dir: str | Path,
    config: FactorLatentConfig,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds, targets = anchor["folds"], anchor["targets"]
    train_rows = np.flatnonzero(folds != int(config.fold))
    valid_rows = np.flatnonzero(folds == int(config.fold))
    prompt_frame = build_all_prompts(bundle, folds, semantic_cache, verifier_config, config.fold)
    prompt_frame.drop(columns=["prompt"]).to_csv(output_dir / "B16_PROMPT_AUDIT.csv", index=False)
    extraction = extract_factor_hidden(
        prompt_frame, adapter_path, verifier_config, output_dir, config
    )
    margins, cube = _to_factor_cubes(prompt_frame, extraction, len(bundle.texts))
    q38_delta = float(np.mean(np.abs(b4.sigmoid(margins[valid_rows]) - anchor["q38_probability"][valid_rows])))
    if q38_delta > 0.01:
        raise AssertionError(f"B16 prompts do not reproduce Q38 OOF margins: MAE={q38_delta:.5f}")

    chosen_layer, chosen_c, inner_scores, selection = crossfit_probe_selection(
        cube, margins, extraction["layers"], targets, bundle.user_ids, train_rows, config
    )
    selection.to_csv(output_dir / "B16_PROBE_SELECTION.csv", index=False)
    layer_position = list(map(int, extraction["layers"])).index(int(chosen_layer))
    x_train = _features(cube, margins, train_rows, layer_position)
    x_valid = _features(cube, margins, valid_rows, layer_position)
    probe = _make_probe(chosen_c, config.seed + int(config.fold))
    _fit_probe(probe, x_train, targets, train_rows)
    joblib.dump(probe, output_dir / "B16_PROBE.joblib")
    valid_scores = _score_matrix(probe, x_valid, len(valid_rows))
    inner_probability, valid_probability = b4.sigmoid(inner_scores), b4.sigmoid(valid_scores)

    base_train = anchor["anchor_probability"][train_rows]
    base_valid = anchor["anchor_probability"][valid_rows]
    threshold_config = b4.B4PConfig(
        threshold_kappa_tail=0.0,
        threshold_kappa_mid=2.0,
        threshold_kappa_head=2.0,
        seed=config.seed,
    )
    baseline_thresholds = b4.fit_factor_thresholds(base_train, targets[train_rows], threshold_config)
    baseline_prediction = (base_valid >= baseline_thresholds).astype(np.int8)
    baseline_metrics, baseline_table = b4.factor_metric_bundle(
        targets[valid_rows], base_valid, baseline_prediction
    )
    baseline_table.insert(0, "system", "Q14_25_Q38_75")

    rows: list[dict[str, Any]] = [{"system": "Q14_25_Q38_75", "alpha": 0.0, **baseline_metrics}]
    tables = [baseline_table]
    candidates: dict[float, dict[str, Any]] = {}
    for alpha in config.blend_alphas:
        train_probability = (1.0 - alpha) * base_train + alpha * inner_probability
        heldout_probability = (1.0 - alpha) * base_valid + alpha * valid_probability
        thresholds = b4.fit_factor_thresholds(train_probability, targets[train_rows], threshold_config)
        prediction = (heldout_probability >= thresholds).astype(np.int8)
        metrics, table = b4.factor_metric_bundle(targets[valid_rows], heldout_probability, prediction)
        name = f"B16_LATENT_BLEND_{int(round(alpha * 100)):03d}"
        rows.append({"system": name, "alpha": float(alpha), **metrics})
        table.insert(0, "system", name)
        tables.append(table)
        candidates[float(alpha)] = {
            "probability": heldout_probability,
            "prediction": prediction,
            "thresholds": thresholds,
            "metrics": metrics,
        }
    summary = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    summary.to_csv(output_dir / "B16_FOLD_SUMMARY.csv", index=False)
    pd.concat(tables, ignore_index=True).to_csv(output_dir / "B16_PER_LABEL.csv", index=False)
    winner = summary.iloc[0]
    if winner.system == "Q14_25_Q38_75":
        delta_f1 = delta_ap = delta_tail = 0.0
        passed = False
    else:
        delta_f1 = float(winner.macro_f1 - baseline_metrics["macro_f1"])
        delta_ap = float(winner.macro_ap - baseline_metrics["macro_ap"])
        delta_tail = float(winner.tail_macro_f1 - baseline_metrics["tail_macro_f1"])
        passed = bool(
            delta_f1 >= config.gate_macro_f1
            and delta_ap >= config.gate_macro_ap
            and delta_tail >= config.gate_tail_floor
        )
    np.savez_compressed(
        output_dir / "B16_FOLD_OUTPUTS.npz",
        row_ids=bundle.row_ids,
        folds=folds,
        valid_rows=valid_rows,
        targets=targets,
        extracted_margins=margins,
        inner_probe_scores=inner_scores,
        valid_probe_scores=valid_scores,
        chosen_layer=np.asarray([chosen_layer]),
        chosen_c=np.asarray([chosen_c]),
    )
    decision = {
        "runtime_revision": B16_RUNTIME_REVISION,
        "fold": int(config.fold),
        "passed": passed,
        "decision": "FREEZE_AND_CONFIRM_FOLDS_1_2" if passed else "STOP_B16",
        "winner": str(winner.system),
        "chosen_layer": int(chosen_layer),
        "chosen_c": float(chosen_c),
        "q38_margin_reproduction_mae": q38_delta,
        "baseline": baseline_metrics,
        "winner_metrics": {key: winner[key] for key in summary.columns if key not in {"system"}},
        "deltas": {"macro_f1": delta_f1, "macro_ap": delta_ap, "tail_macro_f1": delta_tail},
        "gate": {
            "macro_f1_min": config.gate_macro_f1,
            "macro_ap_min": config.gate_macro_ap,
            "tail_macro_f1_floor": config.gate_tail_floor,
        },
        "warning": "Fold 0 is a screening gate. A pass requires frozen confirmation on Folds 1/2.",
    }
    json_dump(decision, output_dir / "B16_FOLD_DECISION.json")
    return decision
