import pandas as pd
from sklearn.metrics import roc_auc_score
import joblib
from scipy.stats import ks_2samp


def validate_model(model_path, test_data_path, model_name="Modelo"):
    """
    Calcula as métricas de discriminação: AUC, Gini Index e KS Statistic.
    Retorna um dicionário com os resultados para uso em outros módulos.
    """
    model = joblib.load(model_path)
    X_test, y_test = joblib.load(test_data_path)

    # Compatibilidade: LogisticRegression é salva como tupla (model, scaler)
    if isinstance(model, tuple):
        model, scaler = model
        X_test = scaler.transform(X_test)

    pd_preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, pd_preds)
    gini = 2 * auc - 1

    pd_bons = pd_preds[y_test == 0]
    pd_maus = pd_preds[y_test == 1]
    ks_stat, _ = ks_2samp(pd_bons, pd_maus)

    print(f"\n{'─'*40}")
    print(f"  Validação: {model_name}")
    print(f"{'─'*40}")
    print(f"  AUC-ROC:        {auc:.4f}")
    print(f"  Gini Index:     {gini:.4f}  {'✓ Aprovado' if gini >= 0.40 else '✗ Abaixo do benchmark'}")
    print(f"  KS Statistic:   {ks_stat:.4f}  {'✓ Aprovado' if ks_stat >= 0.20 else '✗ Abaixo do benchmark'}")

    return {"auc": auc, "gini": gini, "ks": ks_stat}


if __name__ == "__main__":
    # Valida o XGBoost (modelo candidato)
    resultados_xgb = validate_model(
        '../data/xgb_pd_model.pkl',
        '../data/test_data.pkl',
        model_name="XGBoost (Modelo Candidato)"
    )

    # Valida a Regressão Logística (benchmark regulatório)
    resultados_lr = validate_model(
        '../data/lr_pd_model.pkl',
        '../data/test_data.pkl',
        model_name="Regressão Logística (Benchmark Regulatório)"
    )
