import json
import numpy as np
import optuna
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from optuna_integration import LightGBMPruningCallback

from data_prep import get_data
from search_space import suggest_params

SEED = 42
N_SPLITS = 5

X_trainval, X_test, y_trainval, y_test = get_data()


def objective(trial):
    params = suggest_params(trial)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    scores = []

    for tr_idx, val_idx in skf.split(X_trainval, y_trainval):
        X_tr, X_val = X_trainval[tr_idx], X_trainval[val_idx]
        y_tr, y_val = y_trainval[tr_idx], y_trainval[val_idx]

        train_set = lgb.Dataset(X_tr, label=y_tr)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        pruning_callback = LightGBMPruningCallback(trial, "auc", valid_name="valid_0")

        model = lgb.train(
            params,
            train_set,
            num_boost_round=1000,
            valid_sets=[val_set],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
                pruning_callback,
            ],
        )

        preds = model.predict(X_val, num_iteration=model.best_iteration)
        scores.append(roc_auc_score(y_val, preds))

    return float(np.mean(scores))


if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="lgbm_hpo_churn",
        storage="sqlite:///logs/optuna_study_churn.db",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=100, show_progress_bar=True)

    print("\nBest CV AUC:", study.best_value)
    print("Best params:", study.best_params)

    with open("logs/best_params_churn.json", "w") as f:
        json.dump(study.best_params, f, indent=2)