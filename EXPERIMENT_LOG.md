# Experiment Log

This file tracks submission versions and leaderboard results.

## Current Best

- Public score: `0.05282`
- Commit: `ca56674`
- Notebook: `solution.ipynb`
- Accelerator: `GPU P100`
- Notes: v0.3 dual-branch pipeline with text-fusion ablation and blend tuning.

## Submission History

| Version | Date | Commit | Public score | Setup | Key changes | Decision |
|---|---|---|---:|---|---|---|
| v0.1 | 2026-05-17 | f7f76c5 | 0.05290 | P100, full pipeline | First production submit with ablations and threshold tuning | Baseline to beat |
| v0.2 | 2026-05-17 | 3ef3cae | 0.05384 | P100, full pipeline | Metric-direction fix + path cleanup + score tracking | Regressed, discard |
| v0.3 | 2026-05-17 | ca56674 | 0.05282 | P100, full pipeline | Dual branch (TF-IDF + transformer), text fusion ablation, blend tuning | Current best |

## Next Candidate (v0.4)

Planned upgrade:
- run multi-seed TF-IDF ensemble (`322`, `42`, `2026`) for both OOF and test inference;
- keep transformer branch as optional additive blend over TF-IDF ensemble;
- tune blend weight and per-label thresholds on full OOF with fixed reproducible setup;
- optimize for lower metric values (target: beat `0.03774`).
