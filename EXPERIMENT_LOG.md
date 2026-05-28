# Experiment Log

This file tracks submission versions and leaderboard results.

## Targets (competition)

- **Goal (public):** `0.03560` — primary target (Hamming loss, lower is better).
- **Baseline 2 (public):** `0.04168` — intermediate milestone.
- **Our best so far (public):** `0.05219` — gap to goal: `0.01659`; gap to Baseline 2: `0.01051`.

## Current Best (our submissions)

- Public score: `0.05219`
- Commit: `5931056`
- Notebook: `solution.ipynb` (v0.6 classical)
- Accelerator: `GPU P100`
- Notes: TF-IDF + time-based CV and threshold tuning.

## Submission History

| Version | Date | Commit | Public score | Setup | Key changes | Decision |
|---|---|---|---:|---|---|---|
| v0.1 | 2026-05-17 | f7f76c5 | 0.05290 | P100 | First production submit | — |
| v0.2 | 2026-05-17 | 3ef3cae | 0.05384 | P100 | Metric-direction fix | Regressed |
| v0.3 | 2026-05-17 | ca56674 | 0.05282 | P100 | TF-IDF + transformer ablation | — |
| v0.4 | 2026-05-17 | c89aa80 | 0.05282 | P100 | Multi-seed TF-IDF | No gain |
| v0.6 | 2026-05-18 | 5931056 | 0.05219 | P100 | Time-based CV + thresholds | **Current best** |
| v0.7 | 2026-05-18 | 15bcf1d | 0.05574 | P100 | Disentangled sparse | Regressed |
| v0.8 | 2026-05-18 | 00d571a | 0.05621 | P100 | Forward-time CV LR-only | Regressed |
| v0.8.1 | 2026-05-18 | 3a2427d | n/a | P100 | OOF-mask fix | Superseded |
| v1.0 | 2026-05-28 | 6020fb1 | pending | T4/P100 | Neural fine-tune + GO | Superseded |
| v1.0.1 | 2026-05-28 | 77e2f83 | pending | T4 | CUDA `TRAINING_DEVICE` smoke test | Superseded |
| v1.1 | 2026-05-28 | (pending) | pending | T4 | Hybrid TF-IDF + neural, GO stack, 3 seeds | **Submit next** |
| v1.2 | config | — | — | T4 | `MODEL_PROFILE=base`, 3 epochs, final TF-IDF refit | Switch profile on Kaggle |

## v1.1 / v1.2 notebook features

- `RUN_MODE`: `debug` | `full`
- `MODEL_PROFILE`: `tiny` (`rubert-tiny2`) | `base` (`rubert-base-cased`)
- `ENABLE_HYBRID`: TF-IDF (`title_text`) + neural (`title_text_source_date`)
- Head selection: `alpha_blend` vs `GO_stack` on masked OOF
- Kaggle: fail-fast if `full` on CPU — see [`KAGGLE.md`](KAGGLE.md)

## Next steps

1. Kaggle **GPU T4 ×2**, `RUN_MODE=full`, `MODEL_PROFILE=tiny` → record public LB.
2. If OOF masked ≤ ~0.042, submit; else try `MODEL_PROFILE=base`.
3. Update this log after each public score.
