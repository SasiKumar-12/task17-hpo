import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from data_prep import get_data

X_trainval, X_test, y_trainval, y_test = get_data()

model = lgb.LGBMClassifier(random_state=42)
model.fit(X_trainval, y_trainval)

preds = model.predict_proba(X_test)[:, 1]
print("Baseline test AUC:", roc_auc_score(y_test, preds))