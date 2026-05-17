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

## Next Candidate (v0.2)

Planned upgrade:
- expand feature fusion ablations (`title`, `text`, `source`, `publication_date`);
- compare 2 classifiers on fixed embeddings (LogReg vs LinearSVC-compatible route);
- keep threshold tuning and lock same seed (`322`) for comparability.
- optimize for lower metric values (target: beat `0.03774`).
