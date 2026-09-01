# Lenormand — IEEE BigData Cup 2026

This archive contains the source code and report for Lenormand's submission to the Explainable Suicide Risk Detection Challenge. The final public leaderboard result was:

Official competition website: [Big Data Competition](https://www.bigdatacompetition.cn/)

| Metric | Score |
|---|---:|
| Subtask 1 (risk + evidence) | 0.8052 |
| Subtask 2 (24 factors) | 0.6636 |
| Composite | 0.7627 |
| Public leaderboard rank | 9 |

The submitted system is **Heterogeneous Readout Decomposition (HRD)**. A card-conditioned Qwen3.8-27B verifier produces both answer-token margins and answer-position hidden states. Risk is compiled from four ordinal card records; most factors retain the token-margin decision, while seven development-selected labels use one shared latent probe; evidence is selected only from exact substrings of the post by a calibrated set decoder. The report in `report/` gives the method, validation protocol, ablations, and limitations.

## Archive layout

```text
./
├── README.md
├── requirements.txt
├── code/
│   ├── notebooks/               # Colab entry points, ordered by stage
│   ├── modules/                 # Python implementation
│   ├── compose_submission.py    # joins Task-1 and Factor predictions safely
│   ├── final_deployment.py      # applies the frozen B15.1 risk rule
│   └── smoke_test.py            # environment and input-schema audit
├── data_preprocess/
│   └── preprocess.py            # optional cleaning and five-fold diagnostic split
├── docs/
│   └── ARTIFACT_MAP.md          # inputs/outputs and frozen artifact lineage
├── previous_trial/                   # archived pre-HRD experiments; not used by the final pipeline
└── report/
    ├── Lenormand_IEEE_BigData2026_Report.pdf
    ├── main.tex
    ├── references.bib
    └── figures/
```

The competition data, Hugging Face model files, trained adapters, intermediate hidden-state caches, and test predictions are intentionally excluded. They are either distributed under the organizer's data-use terms, reproducible from the code, or too large for a source-code archive.

## Hardware and tested environment

The full pipeline was developed in Google Colab Pro using an NVIDIA A100 80 GB GPU. The 27B verifier is loaded in 4-bit and its QLoRA adapters are trained separately by task. A fresh three-fold reproduction requires substantial GPU time (approximately 30–50 A100 GPU-hours, depending on cache hits and kernel availability). Every expensive notebook writes fold- or chunk-level checkpoints to Google Drive and can be resumed after a disconnection.

The final environment used Python 3.13, PyTorch 2.11.0+cu128, Transformers 5.15.0, and scikit-learn 1.7.2. The exact meta-decoder requires scikit-learn 1.7.2. The Qwen3.8 runs were tested with `causal-conv1d`, `flash-linear-attention`, and the Hugging Face kernels FlashAttention-2 fallback. Training cells explicitly audit the active kernels before a long run.

## Data placement

Obtain the official files from the challenge organizers and use these names:

```text
IEEE_BigData2026/
├── train.xlsx
├── leaderboard.xlsx
└── (the Python modules from code/modules/)
```

`train.xlsx` must contain the official post, user, risk, evidence, and factor annotations. `leaderboard.xlsx` must contain the official test row IDs and posts. The loaders accept both the organizer's original factor representation and the cleaned `f_<label>` columns used in our diagnostics. Do not rename row IDs or reorder rows.

### Optional preprocessing audit

The final HRD notebooks read the official `train.xlsx` directly and construct their own three user-grouped outer folds. `data_preprocess/preprocess.py` is retained as a standalone data-quality audit for the earlier five-fold experiments; it is **not** a prerequisite for the 14-stage reproduction below.

Run it from the directory where you want `train_clean.csv` to be written:

```bash
cd data_preprocess
python preprocess.py /absolute/path/to/train.xlsx
```

The script normalizes the four risk labels, parses and de-duplicates the 24 factor labels, marks missing evidence without dropping rows, creates a seeded five-fold `StratifiedGroupKFold` split grouped by `anon_user_id`, and asserts that no user crosses folds. Its input expects the organizer columns `row_id`, `anon_user_id`, `post_id`, `post`, `suicide risk`, `evidence for suicide risk level`, and `factors`.

## Fast setup: Google Colab

1. Create `/content/drive/MyDrive/IEEE_BigData2026/` in Google Drive.
2. Place the official `train.xlsx` and `leaderboard.xlsx` there.
3. Copy every file from `code/modules/` into that Drive directory.
4. Add `HF_TOKEN` as a Colab secret. All referenced models are downloaded from Hugging Face at run time.
5. Open the notebooks from `code/notebooks/` and run them in the order listed below. Their default `ROOT` is already `/content/drive/MyDrive/IEEE_BigData2026`.
6. Use an A100 80 GB runtime for every notebook marked GPU. Keep the generated `results/` directory between sessions.

The first installation cell may require a runtime restart. After restarting, rerun the setup and import cells; existing adapters, logits, prompts, semantic caches, and hidden-state chunks are detected automatically.

## Linux setup

Ubuntu 22.04/24.04 with an NVIDIA CUDA GPU is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# Install the CUDA build of PyTorch appropriate for the host first.
# See https://pytorch.org/get-started/locally/
python -m pip install -r requirements.txt

# Optional but strongly recommended for Qwen3.8 throughput:
python -m pip install -U "flash-linear-attention[cuda]"
python -m pip install -U causal-conv1d --no-build-isolation

python code/smoke_test.py --root /absolute/path/to/IEEE_BigData2026
jupyter lab
```

Copy `code/modules/*.py` into the selected `IEEE_BigData2026` directory, or add `code/modules` to `PYTHONPATH`. In each notebook's first configuration cell, replace the Colab `ROOT` with the absolute Linux path. The computation itself is platform-neutral; Colab-only `drive.mount` and `files.download` lines may be skipped.

### Quick verification

The smoke test has two useful modes:

```bash
# Check installed package versions and CUDA visibility only.
python code/smoke_test.py

# Also check train/test schemas and import the copied solution modules.
python code/smoke_test.py --root /absolute/path/to/IEEE_BigData2026
```

The second form expects `train.xlsx`, `leaderboard.xlsx`, and the files copied from `code/modules/` to be directly inside the supplied root. It validates the environment and schema only; it does not train a model or reproduce leaderboard scores.

## Windows setup

The supported Windows route is **Windows 11 + WSL2 Ubuntu + NVIDIA CUDA for WSL**. Native Windows and CPU-only execution were not validated and are not practical for the 27B stages.

1. Install WSL2 and an Ubuntu distribution.
2. Confirm `nvidia-smi` works inside WSL.
3. Follow the Linux commands above inside WSL.
4. Keep the project in the WSL filesystem (for example `/home/user/IEEE_BigData2026`) rather than a mounted NTFS directory during training.
5. Change the notebook `ROOT` to that WSL path.

## Reproduction order

The notebooks are numbered to make data and artifact dependencies explicit:

1. `01_qwen3_14b_factor_oof.ipynb` — trains the shared 14B factor verifier used in the factor anchor.
2. `02_qwen38_factor_oof.ipynb` — trains and scores the three Full64 27B factor adapters.
3. `03_task1_fold0.ipynb` — trains the Fold-0 risk/evidence verifier and ModernBERT span proposer.
4. `04_evidence_meta_fold0.ipynb` — develops the exact-span candidate meta-decoder.
5. `05_task1_outer12.ipynb` — frozen Fold-1/Fold-2 confirmation and evidence OOF artifacts.
6. `05a_evidence_environment_audit.ipynb` — CPU-only sklearn 1.7.2 reproducibility audit of the now-frozen evidence meta-model. It intentionally runs after Stage 5 because it consumes those artifacts.
7. `06_base_test_inference.ipynb` — refits the evidence meta-model on OOF rows and runs the base three-fold test ensemble.
8. `07_task1_probe_diagnostics.ipynb` — metric-aligned evidence and factor deployment diagnostics used before latent routing.
9. `08_risk_latent_development.ipynb` — Fold-0 B15 latent readout development.
10. `09_risk_latent_confirmation.ipynb` — locked Fold-1/Fold-2 B15.1 confirmation.
11. `10_risk_latent_test.ipynb` — three-fold B15.1 test readout.
12. `11_factor_latent_development.ipynb` — Fold-0 B16 factor latent gate.
13. `12_factor_latent_confirmation.ipynb` — locked B16.1 seven-label confirmation.
14. `13_factor_latent_test.ipynb` — B16.1 three-fold test deployment and final factor CSV.

The public submission used the B16.1 factor output and the conservative B15.1 risk deployment. For the final risk boundary, a change was accepted only under exact three-of-three fold agreement and a frozen margin rule; no public test labels were used to fit the probes or thresholds. Apply that frozen rule after stages 10 and 13:

```bash
python code/final_deployment.py \
  --base /path/to/B16.1/Lenormand.csv \
  --latent /path/to/B151_TEST_ENSEMBLE_PROBABILITIES.npz \
  --output /path/to/final/Lenormand.csv
```

With the frozen competition artifacts and official test order, this command reproduces the submitted CSV byte-for-byte (SHA-256 `ebbd892be237abbafbd25e6a6f753f6dcdee1733f644f732e5c10218293c343b`). The CSV itself is not redistributed in this source package.

See `docs/ARTIFACT_MAP.md` for the concrete Drive paths consumed and produced at each stage.

## Building the final CSV

Once Task-1 and factor notebooks have produced aligned CSV files:

```bash
python code/compose_submission.py \
  --test /absolute/path/to/leaderboard.xlsx \
  --task1 /absolute/path/to/task1_predictions.csv \
  --factors /absolute/path/to/factor_predictions.csv \
  --output /absolute/path/to/Lenormand.csv
```

The utility preserves official test order, validates the four risk labels, checks that factors are legal list values, verifies evidence is copied verbatim from the post, and prints the SHA-256 checksum. The expected output columns are exactly:

```text
row_id,risk_level,evidence,factors
```

`compose_submission.py` also blanks evidence for `Indicator` rows. It accepts CSV or Excel input tables, aligns both prediction files by `row_id` rather than their current row order, and rejects duplicate or missing IDs. Run `final_deployment.py` afterward only when the B15.1 probability archive is available and you intend to apply the frozen conservative risk adjustment.

## Report sources

The submitted PDF is available at `report/Lenormand_IEEE_BigData2026_Report.pdf`. Its editable sources are `report/main.tex`, `report/references.bib`, and `report/figures/`. `report/make_figures.py` regenerates the two architecture/protocol figures in both PDF and PNG formats; running it overwrites those four figure files.

## Models downloaded at run time

- `Qwen/Qwen3.8-27B` — Full64 task adapters and latent readouts.
- `Qwen/Qwen3-14B` — shared factor verifier in the anchor ensemble.
- `Qwen/Qwen3-30B-A3B-Instruct-2507` — screened anchor component; retained in the original B4-P experiment but not relied on as the final decision rule.
- `answerdotai/ModernBERT-base` — token-level evidence proposal.
- `intfloat/e5-large-v2` — fold-safe retrieval features.

Users are responsible for complying with the data and model licenses. The system is a research benchmark submission, not a clinical diagnostic or intervention tool.

## Authors

- Zirui Li — `25044237g@connect.polyu.hk`
- Yanling Li — `lynnn.li@connect.polyu.hk`
- Kaolanglang Gao — `2510032049@mails.szu.edu.cn`
