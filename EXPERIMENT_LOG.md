# Experiment Log

This file tracks submission versions and leaderboard results.

## Current Best

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
| v0.3 | 2026-05-17 | ca56674 | 0.05282 | P100, full pipeline | Dual branch (TF-IDF + transformer), text fusion ablation, blend tuning | Current best |
| v0.4 | 2026-05-17 | c89aa80 | 0.05282 | P100, full pipeline | Multi-seed TF-IDF ensemble + optional transformer blend | No gain vs v0.3 |
| v0.6 | 2026-05-18 | 5931056 | 0.05219 | P100, full pipeline | Time-based CV and threshold tuning | New best |

## Next Candidate (v0.7)

Planned upgrade:
- switch TF-IDF branch to disentangled sparse channels (`title`, `text-word`, `text-char`, `source/month/year`);
- keep time-based CV and dual-model ensemble (`LogReg + SGD(log_loss)`);
- tune TF-IDF internal model blend weight on OOF (`w * LR + (1-w) * SGD`);
- keep transformer branch optional and disabled by default for faster iterations;
- tune per-label thresholds on full OOF with fixed reproducible setup;
- optimize for lower metric values (target: beat `0.03774`).
