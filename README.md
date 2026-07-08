# Task 17 — Advanced Hyperparameter Tuning

AI/ML Developer · PlaceMux Phase 1 Industry Immersion

## Objective
Efficient hyperparameter optimisation using Bayesian search, pruning, cross-validation,
and early stopping to reach peak validated performance — without wasting compute or
overfitting the search.

## Dataset
[Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) — predicting
whether a customer will cancel their subscription (binary classification).

## Approach
- **Model:** LightGBM (gradient boosted trees)
- **Search:** Optuna, TPE sampler (Bayesian/efficient search)
- **Pruning:** MedianPruner + LightGBMPruningCallback — kills weak trials early
- **Validation:** 5-fold Stratified Cross-Validation, scored on ROC-AUC
- **Early stopping:** 50 rounds, applied within every fold
- **Reproducibility:** fixed seed (42) throughout; every trial logged to SQLite via Optuna storage

## Results

| Stage                          | AUC     |
|---------------------------------|---------|
| Baseline (untuned LightGBM)     | 0.8294  |
| Best CV AUC (search)            | 0.8505  |
| **Final held-out test AUC**     | **0.8401** |

- 100 trials run, **81 pruned early** — most compute spent on promising configs, not wasted on weak ones.
- Small, healthy gap between CV score (0.8505) and test score (0.8401) confirms the tuning generalized rather than overfitting to validation folds.

## Project structure
## How to run
```bash
pip install -r requirements.txt   # optuna, optuna-integration[lightgbm], lightgbm, scikit-learn, pandas, numpy

python baseline.py     # baseline AUC
python tune.py         # runs 100-trial Optuna search
python final_eval.py   # confirms best config on test set
python demo.py         # full end-to-end demo
```