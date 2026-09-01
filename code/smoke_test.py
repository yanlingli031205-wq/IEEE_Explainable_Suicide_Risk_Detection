#!/usr/bin/env python3
"""Lightweight environment, module, and optional data-schema audit."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


REQUIRED_MODULES = (
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "torch",
    "transformers",
    "accelerate",
    "peft",
    "bitsandbytes",
)


def version(name: str) -> str:
    module = importlib.import_module(name)
    return str(getattr(module, "__version__", "installed"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()

    status = {name: version(name) for name in REQUIRED_MODULES}
    import sklearn
    import torch

    if sklearn.__version__ != "1.7.2":
        raise RuntimeError(f"scikit-learn must be 1.7.2, found {sklearn.__version__}")
    status["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        status["gpu"] = torch.cuda.get_device_name(0)
        status["gpu_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)

    if args.root is not None:
        root = args.root.resolve()
        train = root / "train.xlsx"
        test = root / "leaderboard.xlsx"
        missing = [str(p) for p in (train, test) if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing official files: " + ", ".join(missing))
        import pandas as pd

        train_columns = pd.read_excel(train, nrows=2).columns.tolist()
        test_columns = pd.read_excel(test, nrows=2).columns.tolist()
        for label, columns in (("train", train_columns), ("test", test_columns)):
            if "row_id" not in columns or not any(x in columns for x in ("post", "text", "content")):
                raise ValueError(f"Unexpected {label} schema: {columns}")
        status["root"] = str(root)
        status["train_columns"] = train_columns
        status["test_columns"] = test_columns

        modules = root
        if str(modules) not in sys.path:
            sys.path.insert(0, str(modules))
        for name in (
            "b1_experiments",
            "b4p_anchor_verifier",
            "b4_task1_q38",
            "b4e_candidate_meta",
            "b15_latent_readout",
            "b16_factor_latent_readout",
            "b161_factor_latent_route",
        ):
            importlib.import_module(name)
        status["solution_modules"] = "PASS"

    print(json.dumps(status, indent=2, default=str))


if __name__ == "__main__":
    main()

