import optuna
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from data_prep import get_data

study = optuna.load_study(
    study_name="lgbm_hpo_churn",
    storage="sqlite:///logs/optuna_study_churn.db"
)

print(f"Total trials run: {len(study.trials)}")
pruned = sum(1 for t in study.trials if t.state.name == "PRUNED")
print(f"Trials pruned early: {pruned}")
print(f"Best CV AUC found: {study.best_value:.4f}")
print(f"Best trial number: {study.best_trial.number}")
print(f"Best params: {study.best_params}")

X_trainval, X_test, y_trainval, y_test = get_data()
model = lgb.Booster(model_file="models/final_tuned_model_churn.txt")
preds = model.predict(X_test)
print(f"\nFinal tuned model — held-out test AUC: {roc_auc_score(y_test, preds):.4f}")