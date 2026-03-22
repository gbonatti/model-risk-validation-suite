import pandas as pd
import numpy as np
import joblib

def calculate_ecl():
    X_test, _ = joblib.load('../data/test_data.pkl')
    xgb_model = joblib.load('../data/xgb_pd_model.pkl')
    
    pd_12m = xgb_model.predict_proba(X_test)[:, 1]
    ead = X_test['valor_emprestimo'] * np.random.uniform(0.9, 1.1, len(X_test))
    lgd = np.clip(X_test['LTV'] * 0.6 + np.random.normal(0, 0.1, len(X_test)), 0.1, 1.0)
    
    ecl = pd_12m * lgd * ead
    print(f"Exposição Total (EAD): R$ {ead.sum():,.2f}")
    print(f"Provisão Total Esperada (ECL): R$ {ecl.sum():,.2f}")

if __name__ == "__main__":
    calculate_ecl()
