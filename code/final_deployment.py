#!/usr/bin/env python3
"""Apply Lenormand's frozen conservative B15.1 deployment to B16.1 output.

The rule is data independent: change Indicator to Ideation only when all three
latent probes predict Ideation and the mean Ideation-minus-Indicator
probability margin is at least 0.08. Risk labels from the test set are never
read. Factor and evidence cells remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True, help="B16.1 submission CSV")
    parser.add_argument("--latent", type=Path, required=True, help="B151_TEST_ENSEMBLE_PROBABILITIES.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.08)
    args = parser.parse_args()

    frame = pd.read_csv(args.base, dtype=str, keep_default_na=False)
    required = ["row_id", "risk_level", "evidence", "factors"]
    if frame.columns.tolist() != required:
        raise ValueError(f"Expected exact columns {required}; got {frame.columns.tolist()}")

    saved = np.load(args.latent, allow_pickle=True)
    row_ids = saved["row_ids"].astype(str)
    fold_probability = np.asarray(saved["fold_probabilities"], dtype=np.float32)
    ensemble = np.asarray(saved["ensemble_probability"], dtype=np.float32)
    if row_ids.tolist() != frame["row_id"].astype(str).tolist():
        raise ValueError("B15.1 probability rows are not aligned to the B16.1 CSV")
    if fold_probability.shape != (3, len(frame), 4) or ensemble.shape != (len(frame), 4):
        raise ValueError((fold_probability.shape, ensemble.shape))
    if not np.isfinite(fold_probability).all() or not np.isfinite(ensemble).all():
        raise ValueError("Non-finite latent probabilities")

    fold_prediction = fold_probability.argmax(axis=2)
    unanimous_ideation = np.all(fold_prediction == 1, axis=0)
    margin = ensemble[:, 1] - ensemble[:, 0]
    selected = (
        frame["risk_level"].eq("Indicator").to_numpy()
        & unanimous_ideation
        & (margin >= float(args.margin))
    )

    output = frame.copy()
    output.loc[selected, "risk_level"] = "Ideation"
    if not np.array_equal(output["evidence"].to_numpy(), frame["evidence"].to_numpy()):
        raise AssertionError("Evidence changed")
    if not np.array_equal(output["factors"].to_numpy(), frame["factors"].to_numpy()):
        raise AssertionError("Factors changed")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(
        {
            "output": str(args.output),
            "changed_rows": int(selected.sum()),
            "changed_row_ids": output.loc[selected, "row_id"].tolist(),
            "frozen_margin": float(args.margin),
            "sha256": digest,
        }
    )


if __name__ == "__main__":
    main()

