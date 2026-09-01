# Artifact map and run order

All paths below are relative to `ROOT`, which defaults to `/content/drive/MyDrive/IEEE_BigData2026` in the Colab notebooks. Large artifacts are not included in the submission archive.

| Stage | Notebook | Principal output |
|---:|---|---|
| 1 | `01_qwen3_14b_factor_oof.ipynb` | `results/B4P_AVC_FAST3/B4P_CORE_OOF.npz`; three 14B adapters |
| 2 | `02_qwen38_factor_oof.ipynb` | `results/B4_Q38F_FULL64_THREE_FOLD_OOF/Q38_FULL64_OOF.npz`; three 27B factor adapters |
| 3 | `03_task1_fold0.ipynb` | Fold-0 risk/evidence adapter, lexicon, proposer, validation candidates |
| 4 | `04_evidence_meta_fold0.ipynb` | frozen candidate-level evidence meta-decoder configuration |
| 5 | `05_task1_outer12.ipynb` | Fold-1/Fold-2 risk/evidence adapters and OOF candidate audits |
| 5a | `05a_evidence_environment_audit.ipynb` | CPU-only sklearn 1.7.2 reproducibility audit of the frozen evidence meta-model |
| 6 | `06_base_test_inference.ipynb` | base factor ensemble and exact-span Task-1 test predictions |
| 7 | `07_task1_probe_diagnostics.ipynb` | frozen deployment diagnostics and evidence-threshold probes |
| 8 | `08_risk_latent_development.ipynb` | `results/B15_LATENT_RISK_READOUT/fold_0/` |
| 9 | `09_risk_latent_confirmation.ipynb` | `results/B151_LATENT_STANDALONE_CONFIRMATION/fold_{1,2}/` |
| 10 | `10_risk_latent_test.ipynb` | B15.1 three-fold risk probabilities and candidate CSVs |
| 11 | `11_factor_latent_development.ipynb` | `results/B16_FACTOR_LATENT_GATE/fold_0/` |
| 12 | `12_factor_latent_confirmation.ipynb` | `results/B161_FACTOR_LATENT_ROUTE_CONFIRMATION/fold_{1,2}/` |
| 13 | `13_factor_latent_test.ipynb` | B16.1 three-fold routed factor probabilities and CSV |

## Frozen validation design

- Three outer folds are grouped by anonymized user.
- Retrieval memory for an outer validation row contains only outer-training rows and excludes the query user and duplicate text.
- Fold 0 is the development fold for latent probe/layer/route selection.
- Folds 1 and 2 are locked confirmation folds.
- Evidence candidate calibration is fitted on out-of-fold rows; candidate text and all returned evidence are exact post substrings.
- The public leaderboard is reported as an external distribution check, not used as a source of labels.

## Final readouts

- **Risk:** the B15.1 probe compares the four card-conditioned layer-63 records. A conservative deployment accepts only frozen, three-fold-consensus boundary changes.
- **Factors:** the Qwen3.8/Qwen3 token-margin anchor remains the default. Seven labels selected on Fold 0 are routed through one shared B16.1 latent probe after positive confirmation on Folds 1 and 2.
- **Evidence:** ModernBERT proposes token spans; Qwen3.8 scores event candidates; a scikit-learn 1.7.2 candidate meta-model enforces exact-span, event-level, and risk-conditional selection.

The final deployment utility `code/final_deployment.py` applies the preregistered conservative B15.1 rule to the B16.1 CSV: an Indicator-to-Ideation change is permitted only under exact agreement of all three probes and mean probability margin at least 0.08. The utility never reads test labels and leaves evidence and factors byte-identical.

## Reproducibility notes

The notebook defaults are intentionally conservative and assert runtime revision strings before long jobs. Do not bypass a failed assertion by editing a marker. Replace the corresponding module with the matching file in `code/modules/`. If Colab disconnects, reopen the same notebook and rerun from the first cell: adapters, per-fold logits, semantic embeddings, prompts, and hidden chunks are resumed from Drive.
