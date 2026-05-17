# Experiment Log

This file tracks submission versions and leaderboard results.

## Current Best

- Public score: `0.05290`
- Commit: `f7f76c5`
- Notebook: `solution.ipynb`
- Accelerator: `GPU P100`
- Notes: self-contained notebook, Kaggle path autodetection enabled.

## Submission History

| Version | Date | Commit | Public score | Setup | Key changes | Decision |
|---|---|---|---:|---|---|---|
| v0.1 | 2026-05-17 | f7f76c5 | 0.05290 | P100, full pipeline | First production submit with ablations and threshold tuning | Baseline to beat |
| v0.2 | 2026-05-17 | 3ef3cae | 0.05384 | P100, full pipeline | Metric-direction fix + path cleanup + score tracking | Regressed, discard |

## Next Candidate (v0.3)

Planned upgrade:
- run text-fusion ablation (`title+text`, `title+text+source`, `title+text+source+date`);
- train two branches (TF-IDF and transformer embeddings) and blend OOF probabilities;
- tune blend weight and per-label thresholds on full OOF with fixed seed (`322`);
- optimize for lower metric values (target: beat `0.03774`).
