import numpy as np
import joblib


def calculate_psi(expected, actual, bins=10):
    """
    Calcula o Population Stability Index (PSI) entre duas distribuições.

    PSI = Σ (Ai - Ei) * ln(Ai / Ei)

    Limiares de decisão:
        PSI < 0.10  → Verde:  população estável
        PSI < 0.25  → Amarelo: alerta, investigar causas
        PSI >= 0.25 → Vermelho: modelo desatualizado, recalibração necessária
    """
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf

    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents   = np.histogram(actual,   bins=breakpoints)[0] / len(actual)

    # Suavização para evitar log(0)
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents   = np.where(actual_percents   == 0, 0.0001, actual_percents)

    return np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))


def get_psi_status(psi_score):
    """Retorna o status regulatório com base no valor do PSI."""
    if psi_score < 0.10:
        return "🟢 VERDE  — População estável. Modelo apto para uso."
    elif psi_score < 0.25:
        return "🟡 AMARELO — Alerta. Investigar causas do drift populacional."
    else:
        return "🔴 VERMELHO — Instabilidade crítica. Recalibração ou retreinamento obrigatório."


if __name__ == "__main__":
    X_test, _ = joblib.load('../data/test_data.pkl')
    xgb_model = joblib.load('../data/xgb_pd_model.pkl')

    # Scores do período de desenvolvimento (referência)
    pd_desenvolvimento = xgb_model.predict_proba(X_test)[:, 1]

    # Simula amostra operacional com drift macroeconômico:
    # Alta de 50% nas taxas de juros (ciclo de aperto monetário)
    X_operacao = X_test.copy()
    X_operacao['taxa_juros'] = X_operacao['taxa_juros'] * 1.5
    pd_operacao = xgb_model.predict_proba(X_operacao)[:, 1]

    psi_score = calculate_psi(pd_desenvolvimento, pd_operacao)
    status    = get_psi_status(psi_score)

    print("─" * 55)
    print("  Monitoramento de Estabilidade do Modelo (PSI)")
    print("─" * 55)
    print(f"  PSI Calculado : {psi_score:.4f}")
    print(f"  Status        : {status}")
    print("─" * 55)
    print("  Nota: drift simulado via aumento de 50% em taxa_juros,")
    print("  representando cenário de aperto monetário.")
