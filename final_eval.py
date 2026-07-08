import json
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, classification_report
from data_prep import get_data

X_trainval, X_test, y_trainval, y_test = get_data()

with open("logs/best_params_churn.json") as f:
    best_params = json.load(f)

best_params.update({
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "seed": 42,
})

train_set = lgb.Dataset(X_trainval, label=y_trainval)
final_model = lgb.train(best_params, train_set, num_boost_round=500)

test_preds = final_model.predict(X_test)
test_auc = roc_auc_score(y_test, test_preds)

print(f"\nHELD-OUT TEST AUC (tuned model): {test_auc:.4f}")
print("\nClassification report:")
print(classification_report(y_test, (test_preds > 0.5).astype(int)))

final_model.save_model("models/final_tuned_model_churn.txt")
print("\nSaved tuned model to models/final_tuned_model_churn.txt")