import pandas as pd
from sklearn.metrics import roc_auc_score
import joblib
from scipy.stats import ks_2samp

xgb_model = joblib.load('../data/xgb_pd_model.pkl')
X_test, y_test = joblib.load('../data/test_data.pkl')

pd_preds = xgb_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, pd_preds)
gini = 2 * auc - 1

pd_bons = pd_preds[y_test == 0]
pd_maus = pd_preds[y_test == 1]
ks_stat, _ = ks_2samp(pd_bons, pd_maus)

print(f"Gini Index: {gini:.4f}")
print(f"KS Statistic: {ks_stat:.4f}")
