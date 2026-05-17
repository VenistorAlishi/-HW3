# 2026 NLP Kaggle Solution

This repository contains a fully reproducible solution notebook for the 2026 NLP multilabel competition.

## Files
- `solution.ipynb` - main notebook with EDA, training, inference, postprocessing, and submission creation.
- `src/` - helper modules imported by the notebook.
- `pyproject.toml` - dependencies.

## Reproducibility
- Global seed is fixed to `322`.
- The notebook reads `train.csv`, `test.csv`, `sample_submission.csv` from the project root.
- Running all cells in `solution.ipynb` produces the final `sample_submission.csv`.

## Run
1. Put competition files in repository root:
   - `train.csv`
   - `test.csv`
   - `sample_submission.csv`
2. Install dependencies:
   - `uv sync`
3. Run notebook:
   - `uv run jupyter lab`

## Notes
- Pretrained model weights are downloaded at runtime and are not stored in this repository.
- No external datasets are used.
