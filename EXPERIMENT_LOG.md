# Experiment Log

This file tracks submission versions and leaderboard results.

## Targets (competition)

- **Goal (public):** `0.03560` — primary target (Hamming loss, lower is better).
- **Baseline 2 (public):** `0.04168` — intermediate milestone.
- **Our best so far (public):** `0.05179` — gap to goal: `0.01619`; gap to Baseline 2: `0.01011`.

## Current Best (our submissions)

- Public score: `0.05179`
- Commit: `e11a2b7` (v1.1 hybrid on Kaggle T4)
- Notebook: `solution.ipynb`
- Accelerator: `GPU T4 ×2`
- Notes: GO_stack on [TF-IDF, neural]; OOF tuned `0.050599`; neural OOF weak (`~0.073`), TF-IDF `~0.055`.

## Submission History

| Version | Date | Commit | Public score | Setup | Key changes | Decision |
|---|---|---|---:|---|---|---|
| v0.6 | 2026-05-18 | 5931056 | 0.05219 | P100 | Time-based CV + thresholds | Superseded |
| v1.0 / v1.0.1 | 2026-05-28 | 77e2f83 | — | T4 | Neural + CUDA fix | No LB / infra |
| v1.1 | 2026-05-28 | e11a2b7 | **0.05179** | T4, full, tiny | Hybrid TF-IDF+neural, GO_stack, 3 seeds | **Current best** |
| v1.2 | 2026-05-28 | (next) | pending | T4 | LR+SGD TF-IDF blend, optional `ENABLE_NEURAL=False` | Submit next |

## v1.1 Kaggle diagnostics (public 0.05179)

| Metric | Value |
|--------|------:|
| OOF GO_stack @0.5 | 0.051677 |
| OOF tuned (masked) | 0.050599 |
| Hybrid alpha (TF-IDF) | 0.80 |
| Neural ensemble OOF @0.5 | 0.073069 |
| TF-IDF ensemble OOF @0.5 | 0.055075 |
| Selected head | GO_stack |
| Thresholds | [0.46, 0.43, 0.60, 0.42, 0.22] |

## Next steps

1. Run **v1.2** (`TFIDF_LR_SGD_BLEND=True`) on Kaggle T4 — expect lower TF-IDF OOF (~0.046 zone).
2. Optional fast path: `ENABLE_NEURAL=False` (~10 min) if GO on TF-IDF alone is enough.
3. If OOF tuned ≤ ~0.048, try `MODEL_PROFILE=base` for goal `0.03560`.
4. Record each public score below.
