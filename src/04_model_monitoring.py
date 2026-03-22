import numpy as np
import joblib

def calculate_psi(expected, actual, bins=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    return np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))

X_test, _ = joblib.load('../data/test_data.pkl')
pd_desenvolvimento = joblib.load('../data/xgb_pd_model.pkl').predict_proba(X_test)[:, 1]

X_operacao = X_test.copy()
X_operacao['taxa_juros'] = X_operacao['taxa_juros'] * 1.5 
pd_operacao = joblib.load('../data/xgb_pd_model.pkl').predict_proba(X_operacao)[:, 1]

psi_score = calculate_psi(pd_desenvolvimento, pd_operacao)
print(f"Population Stability Index (PSI): {psi_score:.4f}")
