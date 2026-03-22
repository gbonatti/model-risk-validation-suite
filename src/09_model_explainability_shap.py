import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

def validate_model_explainability():
    """
    Validação de Explicabilidade (XAI) para modelos de Machine Learning.
    Utiliza SHAP (SHapley Additive exPlanations) para garantir que as 
    premissas não-lineares do XGBoost respeitam a lógica econômica.
    """
    print("Iniciando Validação de Explicabilidade (SHAP)...")
    
    # Carrega o modelo e os dados de teste
    xgb_model = joblib.load('../data/xgb_pd_model.pkl')
    X_test, _ = joblib.load('../data/test_data.pkl')
    
    # Amostra para otimizar o tempo de cálculo do SHAP
    X_sample = X_test.sample(n=1000, random_state=42)
    
    # Calcula os valores SHAP
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)
    
    # Validação de Premissa Econômica (Monotonicidade)
    # Exemplo: A taxa de juros deve ter correlação positiva com o risco (SHAP value)
    juros_idx = X_sample.columns.get_loc('taxa_juros')
    correlacao_juros_shap = pd.Series(X_sample.iloc[:, juros_idx].values).corr(pd.Series(shap_values[:, juros_idx]))
    
    print("\n--- Auditoria de Premissas Econômicas ---")
    print(f"Correlação Taxa de Juros vs Impacto no Risco (SHAP): {correlacao_juros_shap:.2f}")
    if correlacao_juros_shap > 0.5:
        print("Status: Aprovado. O modelo entende que juros maiores aumentam o risco de default.")
    else:
        print("Status: ALERTA. O modelo pode estar violando a lógica econômica.")

    # Gera o gráfico de impacto global (Feature Importance baseada em SHAP)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    
    os.makedirs('../data/plots', exist_ok=True)
    plt.savefig('../data/plots/shap_summary_plot.png', bbox_inches='tight')
    print("Gráfico de explicabilidade salvo em '../data/plots/shap_summary_plot.png'")

if __name__ == "__main__":
    validate_model_explainability()