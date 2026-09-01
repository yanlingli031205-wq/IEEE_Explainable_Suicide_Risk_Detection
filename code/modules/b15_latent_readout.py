"""B15: latent risk readout for the frozen Qwen3.8-27B Task-1 adapters.

The Qwen verifier was trained to answer four A/B risk-card questions.  B15
tests whether the answer-token LM head is discarding useful information that
is already present in intermediate decoder states.  It does not fine-tune the
27B model: a group-safe linear readout is fitted on frozen hidden states and is
evaluated on the untouched outer fold.

Hidden-state extraction uses forward hooks instead of ``output_hidden_states``
so only the answer-position vector from a few predeclared layers is retained.
Every chunk is immutable and resumable on Drive.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import b1_experiments as b1
import b4p_anchor_verifier as b4
import b4_task1_q38 as task1
import b8_risk_only as b8


B15_RUNTIME_REVISION = "2026-08-30.latent-risk-readout-v1"


@dataclass
class LatentReadoutConfig:
    fold: int = 0
    selected_layers: tuple[int, ...] = (15, 31, 47, 63)
    max_length: int = 1536
    extraction_batch_size: int = 2
    extraction_chunk_size: int = 128
    inner_splits: int = 3
    c_grid: tuple[float, ...] = (0.001, 0.01, 0.1)
    fixed_layer: int | None = None
    fixed_c: float | None = None
    primary_blend_alpha: float = 0.25
    diagnostic_blend_alphas: tuple[float, ...] = (0.50,)
    bootstrap_samples: int = 1000
    seed: int = 42
    gate_weighted_f1_delta: float = 0.008
    gate_macro_f1_delta: float = -0.003
    gate_behavior_recall_delta: float = -0.03
    gate_attempt_recall_delta: float = -0.03


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _dump_json(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _fingerprint(value: Any, length: int = 16) -> str:
    payload = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def locate_task1_fold(root: str | Path, fold: int) -> dict[str, Path]:
    """Resolve the already trained Full64 adapter/spec/evaluation for one fold."""
    root = Path(root)
    fold = int(fold)
    bases = [
        root / f"results/B4_TASK1_Q38_FULL64_FOLD0/Q38_FULL64/fold_{fold}",
        root / f"results/B4_TASK1_Q38_FULL64_OUTER_CONFIRM/Q38_FULL64/fold_{fold}",
    ]
    for base in bases:
        result = {
            "base": base,
            "adapter": base / "adapter/adapter_final",
            "spec": base / "frozen_task1_spec.json",
            "evaluation": base / "EVALUATION/q38_task1_fold_outputs.npz",
        }
        if all(result[key].exists() for key in ("adapter", "spec", "evaluation")):
            return result

    discovered = sorted(root.glob(f"results/**/fold_{fold}/frozen_task1_spec.json"))
    valid: list[dict[str, Path]] = []
    for spec in discovered:
        base = spec.parent
        result = {
            "base": base,
            "adapter": base / "adapter/adapter_final",
            "spec": spec,
            "evaluation": base / "EVALUATION/q38_task1_fold_outputs.npz",
        }
        if result["adapter"].exists() and result["evaluation"].exists():
            valid.append(result)
    if len(valid) == 1:
        return valid[0]
    raise FileNotFoundError(
        f"Could not uniquely locate frozen Task1 Full64 fold {fold}: "
        f"{[str(item['base']) for item in valid]}"
    )


def load_task1_config(spec_path: str | Path) -> task1.Task1Full64Config:
    payload = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    payload = payload.get("config", payload)
    allowed = {field.name for field in fields(task1.Task1Full64Config)}
    payload = {key: value for key, value in payload.items() if key in allowed}
    for key in ("candidate_caps_for_audit", "lora_target_leaves"):
        if key in payload and payload[key] is not None:
            payload[key] = tuple(payload[key])
    return task1.Task1Full64Config(**payload)


def discover_decoder_layers(model: Any) -> tuple[str, dict[int, Any]]:
    """Find the language decoder layer list beneath PEFT/multimodal wrappers."""
    groups: dict[str, dict[int, Any]] = {}
    pattern = re.compile(r"^(.*\.layers)\.(\d+)$")
    for name, module in model.named_modules():
        lowered = name.lower()
        if any(token in lowered for token in ("vision", "visual", "image")):
            continue
        match = pattern.match(name)
        if match:
            groups.setdefault(match.group(1), {})[int(match.group(2))] = module
    if not groups:
        raise RuntimeError("No language decoder layers were found")
    prefix, layers = max(
        groups.items(), key=lambda item: (len(item[1]), max(item[1], default=-1))
    )
    expected = set(range(max(layers) + 1))
    if set(layers) != expected:
        raise RuntimeError(
            f"Decoder layer discovery is not contiguous for {prefix}: "
            f"found={sorted(layers)[:5]}...{sorted(layers)[-5:]}"
        )
    return prefix, layers


def _first_hidden(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_hidden(item)
            except TypeError:
                continue
    if isinstance(value, dict):
        for key in ("last_hidden_state", "hidden_states", "hidden_state"):
            if key in value:
                return _first_hidden(value[key])
    raise TypeError(f"Cannot extract hidden tensor from {type(value)!r}")


@torch.inference_mode()
def _hidden_batch(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    layer_modules: dict[int, Any],
    selected_layers: Sequence[int],
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_id: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = _first_hidden(output)
            if hidden.ndim != 3:
                raise RuntimeError(
                    f"Layer {layer_id} emitted unexpected shape {tuple(hidden.shape)}"
                )
            # Prompts are left padded, so position -1 is always the answer position.
            captured[layer_id] = hidden[:, -1, :].detach().to("cpu", dtype=torch.float16)

        return hook

    for layer_id in selected_layers:
        handles.append(layer_modules[int(layer_id)].register_forward_hook(make_hook(int(layer_id))))
    try:
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        prepared = b4.render_chat_prompts(tokenizer, prompts)
        encoded = tokenizer(
            prepared,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = model(**encoded, use_cache=False)
        a_id, b_id, _, _ = b4.resolve_answer_token_ids(tokenizer)
        margins = (
            outputs.logits[:, -1, a_id] - outputs.logits[:, -1, b_id]
        ).float().cpu().numpy()
    finally:
        for handle in handles:
            handle.remove()
    missing = [layer for layer in selected_layers if int(layer) not in captured]
    if missing:
        raise RuntimeError(f"Forward hooks did not fire for layers {missing}")
    hidden = np.stack([captured[int(layer)].numpy() for layer in selected_layers], axis=1)
    return margins.astype(np.float32), hidden.astype(np.float16)


def extract_hidden_states_cached(
    model: Any,
    tokenizer: Any,
    prompt_frame: pd.DataFrame,
    cache_dir: str | Path,
    config: LatentReadoutConfig,
    adapter_path: str | Path,
) -> dict[str, Any]:
    """Extract answer-position hidden states with immutable chunk recovery."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pair_ids = prompt_frame.pair_id.astype(str).to_numpy()
    prefix, layers = discover_decoder_layers(model)
    selected = tuple(int(layer) for layer in config.selected_layers)
    unavailable = [layer for layer in selected if layer not in layers]
    if unavailable:
        raise IndexError(
            f"Requested decoder layers {unavailable}; discovered 0..{max(layers)} at {prefix}"
        )
    print(f"[B15] language decoder: {prefix}; layers={len(layers)}; selected={selected}")
    fingerprint = _fingerprint(
        {
            "revision": B15_RUNTIME_REVISION,
            "adapter": str(Path(adapter_path).resolve()),
            "pair_ids": pair_ids.tolist(),
            "prompts": [hashlib.sha1(str(item).encode("utf-8")).hexdigest() for item in prompt_frame.prompt],
            "layers": selected,
            "max_length": config.max_length,
        }
    )
    manifest_path = cache_dir / f"hidden_{fingerprint}_manifest.json"
    _dump_json(
        {
            "revision": B15_RUNTIME_REVISION,
            "fingerprint": fingerprint,
            "adapter": str(adapter_path),
            "decoder_prefix": prefix,
            "decoder_layer_count": len(layers),
            "selected_layers": selected,
            "pairs": len(pair_ids),
        },
        manifest_path,
    )

    margins_parts: list[np.ndarray] = []
    hidden_parts: list[np.ndarray] = []
    started = time.perf_counter()
    chunk_size = int(config.extraction_chunk_size)
    for left in range(0, len(prompt_frame), chunk_size):
        right = min(len(prompt_frame), left + chunk_size)
        path = cache_dir / f"hidden_{fingerprint}_{left:06d}_{right:06d}.npz"
        expected = pair_ids[left:right]
        if path.exists():
            saved = np.load(path, allow_pickle=True)
            if saved["pair_ids"].astype(str).tolist() != expected.astype(str).tolist():
                raise AssertionError(f"Hidden cache pair mismatch: {path}")
            margins = np.asarray(saved["margins"], dtype=np.float32)
            hidden = np.asarray(saved["hidden"], dtype=np.float16)
        else:
            batch_margins: list[np.ndarray] = []
            batch_hidden: list[np.ndarray] = []
            cursor = left
            batch_size = int(config.extraction_batch_size)
            while cursor < right:
                stop = min(right, cursor + batch_size)
                try:
                    margins, hidden = _hidden_batch(
                        model,
                        tokenizer,
                        prompt_frame.prompt.iloc[cursor:stop].tolist(),
                        layers,
                        selected,
                        config.max_length,
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if batch_size <= 1:
                        raise
                    batch_size = max(1, batch_size // 2)
                    print(f"[B15] OOM; reducing extraction batch to {batch_size}")
                    continue
                batch_margins.append(margins)
                batch_hidden.append(hidden)
                cursor = stop
            margins = np.concatenate(batch_margins, axis=0)
            hidden = np.concatenate(batch_hidden, axis=0)
            np.savez_compressed(path, pair_ids=expected, margins=margins, hidden=hidden)
        if hidden.shape[0] != right - left or hidden.shape[1] != len(selected):
            raise ValueError(f"Invalid hidden cache shape {hidden.shape}: {path}")
        margins_parts.append(margins)
        hidden_parts.append(hidden)
        elapsed = max(time.perf_counter() - started, 1e-6)
        rate = right / elapsed
        eta = (len(prompt_frame) - right) / max(rate, 1e-6) / 60
        print(f"[B15 hidden] {right}/{len(prompt_frame)} | ETA {eta:.1f} min")
    return {
        "pair_ids": pair_ids,
        "margins": np.concatenate(margins_parts).astype(np.float32),
        "hidden": np.concatenate(hidden_parts).astype(np.float16),
        "layers": np.asarray(selected, dtype=np.int16),
        "fingerprint": fingerprint,
        "decoder_prefix": prefix,
    }


def _to_cubes(
    frame: pd.DataFrame,
    extraction: dict[str, Any],
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.asarray(extraction["hidden"], dtype=np.float16)
    margins = np.asarray(extraction["margins"], dtype=np.float32)
    cube = np.full(
        (int(n_rows), 4, hidden.shape[1], hidden.shape[2]), np.nan, dtype=np.float16
    )
    matrix = np.full((int(n_rows), 4), np.nan, dtype=np.float32)
    for index, record in enumerate(frame.itertuples(index=False)):
        row, risk = int(record.query_row_idx), int(record.risk_id)
        if np.isfinite(matrix[row, risk]):
            raise AssertionError(f"Duplicate prompt row={row}, risk={risk}")
        matrix[row, risk] = margins[index]
        cube[row, risk] = hidden[index]
    if not np.isfinite(matrix).all() or not np.isfinite(cube).all():
        raise ValueError("Latent extraction did not cover every row/risk pair")
    return matrix, cube


def _pair_features(cube: np.ndarray, rows: np.ndarray, layer_position: int) -> np.ndarray:
    values = np.asarray(cube[rows, :, int(layer_position), :], dtype=np.float32)
    flattened = values.reshape(-1, values.shape[-1])
    risk_one_hot = np.tile(np.eye(4, dtype=np.float32), (len(rows), 1))
    return np.concatenate([flattened, risk_one_hot], axis=1)


def _pair_targets(gold: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return (
        np.arange(4, dtype=np.int8)[None, :]
        == np.asarray(gold, dtype=np.int8)[rows, None]
    ).reshape(-1).astype(np.int8)


def _make_probe(c_value: float, seed: int) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(c_value),
            solver="lbfgs",
            max_iter=400,
            random_state=int(seed),
        ),
    )


def _decision_matrix(model: Any, features: np.ndarray, n_rows: int) -> np.ndarray:
    scores = np.asarray(model.decision_function(features), dtype=np.float32)
    return scores.reshape(int(n_rows), 4)


def select_probe_configuration(
    cube: np.ndarray,
    layers: Sequence[int],
    gold: np.ndarray,
    user_ids: np.ndarray,
    train_rows: np.ndarray,
    config: LatentReadoutConfig,
) -> tuple[int, float, pd.DataFrame]:
    if config.fixed_layer is not None and config.fixed_c is not None:
        if int(config.fixed_layer) not in set(map(int, layers)):
            raise ValueError(f"Fixed layer {config.fixed_layer} is not in {list(layers)}")
        return int(config.fixed_layer), float(config.fixed_c), pd.DataFrame(
            [{"layer": int(config.fixed_layer), "c": float(config.fixed_c), "fixed": True}]
        )

    train_rows = np.asarray(train_rows, dtype=int)
    groups = np.asarray(user_ids).astype(str)[train_rows]
    unique_groups = np.unique(groups)
    splits = min(int(config.inner_splits), len(unique_groups))
    if splits < 2:
        raise ValueError("Need at least two user groups for probe selection")
    splitter = GroupKFold(n_splits=splits)
    records: list[dict[str, Any]] = []
    layer_values = list(map(int, layers))
    for layer_position, layer_id in enumerate(layer_values):
        all_features = _pair_features(cube, train_rows, layer_position)
        for c_value in config.c_grid:
            oof_scores = np.full((len(train_rows), 4), np.nan, dtype=np.float32)
            for split_id, (inner_train, inner_valid) in enumerate(
                splitter.split(train_rows, groups=groups)
            ):
                row_train = train_rows[inner_train]
                y_train = _pair_targets(gold, row_train)
                pair_train = np.concatenate(
                    [np.arange(index * 4, index * 4 + 4) for index in inner_train]
                )
                pair_valid = np.concatenate(
                    [np.arange(index * 4, index * 4 + 4) for index in inner_valid]
                )
                probe = _make_probe(float(c_value), config.seed + split_id)
                probe.fit(all_features[pair_train], y_train)
                oof_scores[inner_valid] = _decision_matrix(
                    probe, all_features[pair_valid], len(inner_valid)
                )
            if not np.isfinite(oof_scores).all():
                raise ValueError("Incomplete inner OOF probe scores")
            prediction = np.argmax(oof_scores, axis=1)
            records.append(
                {
                    "layer": layer_id,
                    "layer_position": layer_position,
                    "c": float(c_value),
                    "inner_weighted_f1": float(
                        f1_score(gold[train_rows], prediction, average="weighted", zero_division=0)
                    ),
                    "inner_macro_f1": float(
                        f1_score(gold[train_rows], prediction, average="macro", zero_division=0)
                    ),
                }
            )
            print(
                f"[B15 probe] layer={layer_id} C={c_value:g} "
                f"inner-WF1={records[-1]['inner_weighted_f1']:.4f}"
            )
    audit = pd.DataFrame(records).sort_values(
        ["inner_weighted_f1", "inner_macro_f1", "layer", "c"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    winner = audit.iloc[0]
    return int(winner.layer), float(winner.c), audit


def _softmax(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = matrix - matrix.max(axis=1, keepdims=True)
    values = np.exp(matrix)
    return (values / np.maximum(values.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)


def _row_standardize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    return centered / np.maximum(scale, 1e-5)


def _blend_prediction(
    baseline_margins: np.ndarray,
    probe_scores: np.ndarray,
    alpha: float,
) -> np.ndarray:
    baseline_probability = _softmax(_row_standardize(baseline_margins))
    probe_probability = _softmax(_row_standardize(probe_scores))
    probability = (1.0 - float(alpha)) * baseline_probability + float(alpha) * probe_probability
    return np.argmax(probability, axis=1).astype(np.int8)


def _paired_bootstrap(
    gold: np.ndarray,
    baseline: np.ndarray,
    challenger: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(int(samples), dtype=np.float32)
    for index in range(int(samples)):
        rows = rng.integers(0, len(gold), len(gold))
        values[index] = f1_score(
            gold[rows], challenger[rows], average="weighted", zero_division=0
        ) - f1_score(gold[rows], baseline[rows], average="weighted", zero_division=0)
    return {
        "mean": float(values.mean()),
        "p025": float(np.quantile(values, 0.025)),
        "p50": float(np.quantile(values, 0.50)),
        "p975": float(np.quantile(values, 0.975)),
        "probability_positive": float(np.mean(values > 0)),
    }


def run_latent_readout_fold(
    bundle: b1.DataBundle,
    folds: np.ndarray,
    root: str | Path,
    output_dir: str | Path,
    config: LatentReadoutConfig,
) -> dict[str, Any]:
    """Extract one fold, fit the readout, and perform an untouched-fold gate."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = locate_task1_fold(root, config.fold)
    task1_config = load_task1_config(artifacts["spec"])
    if task1_config.model_name != "Qwen/Qwen3.8-27B":
        raise AssertionError(f"Unexpected frozen model: {task1_config.model_name}")
    if task1_config.lora_last_n_layers is not None:
        raise AssertionError("B15 requires the confirmed Full64 adapter")

    rows = np.arange(len(bundle.texts), dtype=int)
    prompt_frame = task1.build_risk_prompt_frame(
        bundle, rows, task1_config, evidence_by_row=None, stage="b15-document"
    )
    prompt_frame.drop(columns=["prompt"]).to_csv(output_dir / "B15_PROMPT_AUDIT.csv", index=False)

    model, tokenizer = b4.load_quantized_causal_model(
        task1_config.model_name,
        adapter_path=artifacts["adapter"],
        training=False,
        attention_implementation=task1_config.attention_implementation,
        qwen35_fa2_position_guard=task1_config.qwen35_fa2_position_guard,
        require_qwen35_fast_kernels=task1_config.require_qwen35_fast_kernels,
    )
    extraction = extract_hidden_states_cached(
        model,
        tokenizer,
        prompt_frame,
        output_dir / "HIDDEN_CACHE",
        config,
        artifacts["adapter"],
    )
    del model, tokenizer
    b4.unload_model()
    gc.collect()
    document_margins, cube = _to_cubes(prompt_frame, extraction, len(bundle.texts))

    frozen = b8.load_frozen_risk_margins(
        artifacts["evaluation"], bundle, folds, config.fold
    )
    valid_rows = np.flatnonzero(np.asarray(folds, dtype=int) == int(config.fold))
    train_rows = np.flatnonzero(np.asarray(folds, dtype=int) != int(config.fold))
    frozen_document = frozen["B4_DOCUMENT"][valid_rows]
    extracted_document = document_margins[valid_rows]
    reproduction = {
        "prediction_agreement": float(
            np.mean(np.argmax(frozen_document, axis=1) == np.argmax(extracted_document, axis=1))
        ),
        "margin_correlation": float(
            np.corrcoef(frozen_document.reshape(-1), extracted_document.reshape(-1))[0, 1]
        ),
        "margin_rmse": float(np.sqrt(np.mean((frozen_document - extracted_document) ** 2))),
    }
    if reproduction["prediction_agreement"] < 0.995 or reproduction["margin_correlation"] < 0.999:
        raise AssertionError(f"Frozen document scoring was not reproduced: {reproduction}")

    layers = extraction["layers"].astype(int).tolist()
    chosen_layer, chosen_c, selection = select_probe_configuration(
        cube,
        layers,
        bundle.risk_ids,
        bundle.user_ids,
        train_rows,
        config,
    )
    selection.to_csv(output_dir / "B15_PROBE_SELECTION.csv", index=False)
    layer_position = layers.index(int(chosen_layer))
    x_train = _pair_features(cube, train_rows, layer_position)
    y_train = _pair_targets(bundle.risk_ids, train_rows)
    x_valid = _pair_features(cube, valid_rows, layer_position)
    probe = _make_probe(chosen_c, config.seed + config.fold)
    probe.fit(x_train, y_train)
    joblib.dump(probe, output_dir / "B15_PROBE.joblib")
    probe_scores = _decision_matrix(probe, x_valid, len(valid_rows))

    gold = np.asarray(bundle.risk_ids, dtype=int)[valid_rows]
    baseline_margins = frozen["B4_FIXED_BLEND"][valid_rows]
    predictions: dict[str, np.ndarray] = {
        "B4_FIXED_BLEND": np.argmax(baseline_margins, axis=1).astype(np.int8),
        "B15_DOCUMENT_LM_HEAD": np.argmax(extracted_document, axis=1).astype(np.int8),
        "B15_LATENT_PROBE": np.argmax(probe_scores, axis=1).astype(np.int8),
        f"B15_LATENT_BLEND_{int(round(100 * config.primary_blend_alpha)):03d}": _blend_prediction(
            baseline_margins, probe_scores, config.primary_blend_alpha
        ),
    }
    for alpha in config.diagnostic_blend_alphas:
        name = f"B15_DIAGNOSTIC_BLEND_{int(round(100 * alpha)):03d}"
        predictions[name] = _blend_prediction(baseline_margins, probe_scores, alpha)
    metrics = {name: b8.risk_metrics(gold, value) for name, value in predictions.items()}
    primary_name = f"B15_LATENT_BLEND_{int(round(100 * config.primary_blend_alpha)):03d}"
    baseline = metrics["B4_FIXED_BLEND"]
    primary = metrics[primary_name]
    deltas = {
        "weighted_f1": float(primary["weighted_f1"] - baseline["weighted_f1"]),
        "macro_f1": float(primary["macro_f1"] - baseline["macro_f1"]),
        "behavior_recall": float(
            primary["per_class"]["Behavior"]["recall"]
            - baseline["per_class"]["Behavior"]["recall"]
        ),
        "attempt_recall": float(
            primary["per_class"]["Attempt"]["recall"]
            - baseline["per_class"]["Attempt"]["recall"]
        ),
    }
    passed = bool(
        deltas["weighted_f1"] >= config.gate_weighted_f1_delta
        and deltas["macro_f1"] >= config.gate_macro_f1_delta
        and deltas["behavior_recall"] >= config.gate_behavior_recall_delta
        and deltas["attempt_recall"] >= config.gate_attempt_recall_delta
    )
    decision = {
        "runtime_revision": B15_RUNTIME_REVISION,
        "fold": int(config.fold),
        "rows": int(len(valid_rows)),
        "primary": primary_name,
        "baseline": "B4_FIXED_BLEND",
        "chosen_layer": int(chosen_layer),
        "chosen_c": float(chosen_c),
        "metrics": metrics,
        "deltas": deltas,
        "reproduction": reproduction,
        "paired_bootstrap": _paired_bootstrap(
            gold,
            predictions["B4_FIXED_BLEND"],
            predictions[primary_name],
            config.bootstrap_samples,
            config.seed + config.fold,
        ),
        "passed": passed,
        "decision": "FREEZE_READOUT_AND_RUN_FOLDS_1_2" if passed else "STOP_B15",
        "gate": {
            "weighted_f1_delta_min": config.gate_weighted_f1_delta,
            "macro_f1_delta_min": config.gate_macro_f1_delta,
            "behavior_recall_delta_min": config.gate_behavior_recall_delta,
            "attempt_recall_delta_min": config.gate_attempt_recall_delta,
        },
        "config": asdict(config),
        "method_note": (
            "Only Fold 0 may choose layer/C. If it passes, freeze those two values for "
            "Folds 1/2. Diagnostic alpha=0.50 is not eligible for the gate."
        ),
    }
    summary = pd.DataFrame(
        [
            {
                "system": name,
                **{
                    key: value[key]
                    for key in ("weighted_f1", "macro_f1", "accuracy", "mean_absolute_severity_error")
                },
                "indicator_recall": value["per_class"]["Indicator"]["recall"],
                "ideation_recall": value["per_class"]["Ideation"]["recall"],
                "behavior_recall": value["per_class"]["Behavior"]["recall"],
                "attempt_recall": value["per_class"]["Attempt"]["recall"],
            }
            for name, value in metrics.items()
        ]
    ).sort_values("weighted_f1", ascending=False)
    summary.to_csv(output_dir / "B15_FOLD_SUMMARY.csv", index=False)
    pd.DataFrame(
        {
            "row_id": bundle.row_ids[valid_rows],
            "gold_risk": [b1.RISK_LABELS[item] for item in gold],
            **{
                f"prediction__{name}": [b1.RISK_LABELS[item] for item in prediction]
                for name, prediction in predictions.items()
            },
        }
    ).to_csv(output_dir / "B15_FOLD_PREDICTIONS.csv", index=False)
    np.savez_compressed(
        output_dir / "B15_FOLD_OUTPUTS.npz",
        row_ids=bundle.row_ids,
        folds=np.asarray(folds, dtype=np.int8),
        valid_rows=valid_rows,
        gold=gold,
        baseline_margins=baseline_margins,
        document_margins=extracted_document,
        probe_scores=probe_scores,
        **{f"prediction__{name}": value for name, value in predictions.items()},
    )
    _dump_json(decision, output_dir / "B15_FOLD_DECISION.json")
    return decision


def summarize_completed_folds(
    fold_dirs: Iterable[str | Path], output_dir: str | Path
) -> dict[str, Any]:
    """Combine completed folds without changing their frozen predictions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for directory in map(Path, fold_dirs):
        path = directory / "B15_FOLD_DECISION.json"
        if not path.exists():
            continue
        decision = json.loads(path.read_text(encoding="utf-8"))
        decisions.append(decision)
        for system, value in decision["metrics"].items():
            records.append(
                {
                    "fold": int(decision["fold"]),
                    "system": system,
                    "weighted_f1": value["weighted_f1"],
                    "macro_f1": value["macro_f1"],
                    "accuracy": value["accuracy"],
                }
            )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise FileNotFoundError("No completed B15 fold decisions")
    frame.to_csv(output_dir / "B15_COMPLETED_FOLDS.csv", index=False)
    result = {
        "runtime_revision": B15_RUNTIME_REVISION,
        "completed_folds": sorted({int(item["fold"]) for item in decisions}),
        "all_passed": bool(len(decisions) == 3 and all(item["passed"] for item in decisions)),
        "fold_decisions": decisions,
    }
    _dump_json(result, output_dir / "B15_COMPLETED_DECISION.json")
    return result


def score_latent_test_fold(
    test_corpus: b4.TextCorpus,
    root: str | Path,
    output_dir: str | Path,
    fold: int,
    probe_path: str | Path,
    extraction_batch_size: int = 2,
    extraction_chunk_size: int = 128,
) -> np.ndarray:
    """Apply one frozen Task-1 adapter/readout to the leaderboard corpus.

    Returned values are row-normalized four-class probabilities.  Normalizing
    before the three-fold average prevents one probe's arbitrary linear-score
    scale from dominating the ensemble.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "B151_TEST_LATENT_PROBABILITIES.npz"
    if result_path.exists():
        saved = np.load(result_path, allow_pickle=True)
        if saved["row_ids"].astype(str).tolist() != test_corpus.row_ids.astype(str).tolist():
            raise AssertionError(f"B15.1 test row mismatch: {result_path}")
        probability = np.asarray(saved["probabilities"], dtype=np.float32)
        if probability.shape != (len(test_corpus.texts), 4) or not np.isfinite(probability).all():
            raise ValueError(f"Invalid B15.1 test cache: {result_path}")
        print(f"[B15.1 test resume] {result_path}")
        return probability

    artifacts = locate_task1_fold(root, int(fold))
    task1_config = load_task1_config(artifacts["spec"])
    probe_path = Path(probe_path)
    if not probe_path.exists():
        raise FileNotFoundError(f"Missing fitted B15.1 probe: {probe_path}")
    proxy = SimpleNamespace(
        texts=list(test_corpus.texts),
        row_ids=np.asarray(test_corpus.row_ids).astype(str),
    )
    frame = task1.build_risk_prompt_frame(
        proxy,
        np.arange(len(test_corpus.texts), dtype=int),
        task1_config,
        evidence_by_row=None,
        stage=f"b151-test-fold{int(fold)}",
    )
    frame.drop(columns=["prompt"]).to_csv(output_dir / "B151_TEST_PROMPT_AUDIT.csv", index=False)
    extraction_config = LatentReadoutConfig(
        fold=int(fold),
        selected_layers=(63,),
        max_length=int(task1_config.max_length),
        extraction_batch_size=int(extraction_batch_size),
        extraction_chunk_size=int(extraction_chunk_size),
        fixed_layer=63,
        fixed_c=0.01,
        primary_blend_alpha=1.0,
        diagnostic_blend_alphas=(),
    )
    model, tokenizer = b4.load_quantized_causal_model(
        task1_config.model_name,
        adapter_path=artifacts["adapter"],
        training=False,
        attention_implementation=task1_config.attention_implementation,
        qwen35_fa2_position_guard=task1_config.qwen35_fa2_position_guard,
        require_qwen35_fast_kernels=task1_config.require_qwen35_fast_kernels,
    )
    extraction = extract_hidden_states_cached(
        model,
        tokenizer,
        frame,
        output_dir / "HIDDEN_CACHE",
        extraction_config,
        artifacts["adapter"],
    )
    del model, tokenizer
    b4.unload_model()
    _, cube = _to_cubes(frame, extraction, len(test_corpus.texts))
    features = _pair_features(cube, np.arange(len(test_corpus.texts)), 0)
    probe = joblib.load(probe_path)
    scores = _decision_matrix(probe, features, len(test_corpus.texts))
    probability = _softmax(_row_standardize(scores))
    np.savez_compressed(
        result_path,
        row_ids=test_corpus.row_ids.astype(str),
        fold=np.asarray([int(fold)], dtype=np.int8),
        scores=scores.astype(np.float32),
        probabilities=probability.astype(np.float32),
    )
    return probability.astype(np.float32)


_DIRECT_SUICIDE_CUE = re.compile(
    r"(?i)(?:"
    r"suicid\w*|kill(?:ing)?\s+myself|end(?:ing)?\s+my\s+life|take\s+my\s+own\s+life|"
    r"want\s+to\s+die|wanna\s+die|wish\s+i\s+(?:was|were)\s+dead|better\s+off\s+dead|"
    r"no\s+reason\s+to\s+live|not\s+worth\s+living|attempt(?:ed|ing)?\s+suicide|"
    r"overdos\w*|hang(?:ing)?\s+myself|jump(?:ing)?\s+(?:off|from)|"
    r"slit(?:ting)?\s+my\s+wrists?|shoot(?:ing)?\s+myself|self[- ]harm\w*"
    r")"
)


def direct_cue_evidence(text: str, max_chars: int = 220) -> str:
    """Return one concise verbatim clause for an old-Indicator/new-risk row.

    This is deliberately high precision.  Returning an empty string is safer
    than inventing a weak phrase because the current Indicator submission had
    zero evidence recall on such a row and a false phrase would additionally
    hurt precision.
    """
    text = str(text)
    matches = list(_DIRECT_SUICIDE_CUE.finditer(text))
    if not matches:
        return ""
    candidates: list[tuple[int, int, str]] = []
    for match in matches:
        left = max(
            text.rfind("\n", 0, match.start()),
            text.rfind(".", 0, match.start()),
            text.rfind("!", 0, match.start()),
            text.rfind("?", 0, match.start()),
        ) + 1
        endings = [
            index for token in ("\n", ".", "!", "?")
            if (index := text.find(token, match.end())) >= 0
        ]
        right = min(endings) + 1 if endings else len(text)
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if right - left > int(max_chars):
            half = max(20, int(max_chars) // 2)
            left = max(left, match.start() - half)
            right = min(right, left + int(max_chars))
            while left < match.start() and left > 0 and text[left - 1].isalnum():
                left += 1
            while right > match.end() and right < len(text) and text[right].isalnum():
                right -= 1
        phrase = text[left:right].strip()
        if phrase:
            candidates.append((len(phrase), match.start(), phrase))
    return min(candidates, default=(0, 0, ""))[2]
