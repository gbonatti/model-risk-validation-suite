import numpy as np
import pandas as pd
from scipy.stats import chi2

def kupiec_pof_test(exceptions, n_observations, confidence_level):
    """
    Teste de Kupiec (Proportion of Failures) para Backtesting de VaR.
    Testa se a taxa de exceções real é estatisticamente igual à esperada.
    """
    p_expected = 1 - confidence_level
    p_observed = exceptions / n_observations
    
    # Log-likelihood ratio test
    # Se p_observed for 0, o log falha, então tratamos esse caso:
    if exceptions == 0:
        lr_pof = -2 * np.log((1 - p_expected)**n_observations)
    else:
        lr_pof = -2 * (
            np.log(((1 - p_expected)**(n_observations - exceptions)) * (p_expected**exceptions)) -
            np.log(((1 - p_observed)**(n_observations - exceptions)) * (p_observed**exceptions))
        )
    
    # P-valor usando distribuição Chi-quadrado com 1 grau de liberdade
    p_value = 1 - chi2.cdf(lr_pof, df=1)
    return lr_pof, p_value

def validate_var_model():
    """Simula o backtesting de um modelo de Value at Risk (VaR) Histórico de 1 dia."""
    np.random.seed(42)
    n_days = 252 # 1 ano de dias úteis
    confidence = 0.99 # VaR a 99%
    
    print("--- Validação de Risco de Mercado: Backtesting de VaR ---")
    
    # Simulando P&L (Lucros e Perdas) diários da mesa de operações e o VaR estimado pelo modelo
    # Em uma distribuição normal, esperamos ~2.5 exceções em 252 dias para 99% de confiança
    pnl_real = np.random.normal(loc=10000, scale=100000, size=n_days)
    var_99_estimado = np.full(n_days, -230000) # Modelo estima perda máxima de 230k
    
    # Conta quantas vezes a perda real foi pior (mais negativa) que o VaR
    excecoes = np.sum(pnl_real < var_99_estimado)
    
    print(f"Dias observados: {n_days}")
    print(f"Nível de Confiança: {confidence * 100}%")
    print(f"Exceções Esperadas: {n_days * (1 - confidence):.2f}")
    print(f"Exceções Observadas: {excecoes}")
    
    # Aplica o Teste de Kupiec
    stat, p_valor = kupiec_pof_test(excecoes, n_days, confidence)
    print(f"\nEstatística de Kupiec (LR): {stat:.4f}")
    print(f"P-Valor: {p_valor:.4f}")
    
    if p_valor > 0.05:
        print("Status: Verde. Modelo de VaR validado. As exceções estão dentro da margem estatística (H0 não rejeitada).")
    else:
        print("Status: Vermelho. Modelo calibrado incorretamente. Risco de subestimação severa.")

if __name__ == "__main__":
    validate_var_model()