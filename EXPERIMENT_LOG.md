# Experiment Log

This file tracks submission versions and leaderboard results.

## Targets (competition)

- **Goal (public):** `0.03560` — primary target to reach (Hamming loss, lower is better).
- **Baseline 2 (public):** `0.04168` — intermediate milestone.
- **Our best so far (public):** `0.05219` — gap to goal: `0.01659`; gap to Baseline 2: `0.01051`.

## Current Best (our submissions)

- Public score: `0.05219`
- Commit: `5931056`
- Notebook: `solution.ipynb`
- Accelerator: `GPU P100`
- Notes: v0.6 with time-based CV and wider threshold grid.

## Submission History

| Version | Date | Commit | Public score | Setup | Key changes | Decision |
|---|---|---|---:|---|---|---|
| v0.1 | 2026-05-17 | f7f76c5 | 0.05290 | P100, full pipeline | First production submit with ablations and threshold tuning | Baseline to beat |
| v0.2 | 2026-05-17 | 3ef3cae | 0.05384 | P100, full pipeline | Metric-direction fix + path cleanup + score tracking | Regressed, discard |
| v0.3 | 2026-05-17 | ca56674 | 0.05282 | P100, full pipeline | Dual branch (TF-IDF + transformer), text fusion ablation, blend tuning | — |
| v0.4 | 2026-05-17 | c89aa80 | 0.05282 | P100, full pipeline | Multi-seed TF-IDF ensemble + optional transformer blend | No gain vs v0.3 |
| v0.6 | 2026-05-18 | 5931056 | 0.05219 | P100, full pipeline | Time-based CV and threshold tuning | **Current best** |
| v0.7 | 2026-05-18 | 15bcf1d | 0.05574 | P100, full pipeline | Disentangled sparse features + LR/SGD ensemble | Regressed, discard |
| v0.8 | 2026-05-18 | 00d571a | 0.05621 | P100, full pipeline | Forward-time CV + LR-only stabilization | Regressed, discard |
| v0.8.1 | 2026-05-18 | 3a2427d | n/a | P100, full pipeline | OOF-mask fix + stable text rollback | Superseded by v1.0 |
| v1.0 | 2026-05-28 | 6020fb1 | pending | T4/P100 | Neural fine-tune `rubert-tiny2`, forward-time CV, GO LogReg, thresholds | Awaiting LB |
| v1.0.1 | 2026-05-28 | (this push) | pending | T4 recommended | CUDA smoke test + `TRAINING_DEVICE` CPU fallback | Kaggle stability fix |

## Next Candidate (v1.1)

Planned after v1.0 LB:

- hybrid stack: TF-IDF (forward-time CV) + neural OOF blend/stack;
- GO calibration on stacked OOF;
- per-label thresholds on masked OOF;
- aim public ≤ `0.04168`, then ≤ `0.03560`.
