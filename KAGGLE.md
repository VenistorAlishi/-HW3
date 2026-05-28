# Kaggle runbook

## Setup

1. Link this repo or upload [`solution.ipynb`](solution.ipynb) from branch `main`.
2. **Settings → Accelerator → GPU T4 ×2** (not P100 if it fails with CUDA kernel errors, not TPU).
3. **Internet: ON** (download `cointegrated/rubert-tiny2` or `DeepPavlov/rubert-base-cased`).
4. Competition data attached as input (`train.csv`, `test.csv`, `sample_submission.csv`).

## Run

1. **Run All** cells.
2. In the config cell output, confirm:
   - `Training device: cuda`
   - `Run mode: full` (or `debug` for a quick smoke test)
3. Output file: `/kaggle/working/sample_submission.csv` → **Submit**.

## Debug vs full

| `RUN_MODE` | Folds | Seeds | Epochs | Use |
|------------|-------|-------|--------|-----|
| `debug` | 3 | [322] | 1 | Quick pipeline check |
| `full` | 5 | [322, 42, 2026] | 2–3 | Leaderboard submit |

If `RUN_MODE=full` on Kaggle and GPU is unavailable, the notebook **raises** instead of running for hours on CPU.

## Model profiles

- `MODEL_PROFILE = "tiny"` — default, `cointegrated/rubert-tiny2`, faster.
- `MODEL_PROFILE = "base"` — `DeepPavlov/rubert-base-cased`, 3 epochs, slower; use when chasing goal **0.03560**.
- `ENABLE_NEURAL = False` — skip transformer training (~10 min saved); TF-IDF + GO only.
- `TFIDF_LR_SGD_BLEND = True` — v1.2: LR + SGD blend per seed (recommended).

**Best public so far:** `0.05179` (v1.1, GO_stack, T4).

## Targets (public Hamming loss, lower is better)

- Baseline 2: **0.04168**
- Goal: **0.03560**

After submit, record the public score in [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).
