import numpy as np
import pandas as pd

def validate_fraud_detection_benford():
    """
    Aplica a Lei de Benford para detectar anomalias em volumes transacionais (PLD/Fraude).
    """
    print("--- Auditoria de PLD: Análise de Primeiro Dígito (Lei de Benford) ---")
    
    # Simulando 1000 transações suspeitas
    transacoes = np.random.lognormal(mean=10, sigma=2, size=1000)
    primeiros_digitos = [int(str(abs(x))[0]) for x in transacoes if x > 0]
    
    contagem_real = pd.Series(primeiros_digitos).value_counts(normalize=True).sort_index()
    benford_ideal = pd.Series([np.log10(1 + 1/d) for d in range(1, 10)], index=range(1, 10))
    
    comparativo = pd.DataFrame({'Real': contagem_real, 'Benford': benford_ideal})
    print(comparativo)
    
    desvio_max = (comparativo['Real'] - comparativo['Benford']).abs().max()
    
    if desvio_max > 0.05:
        print(f"\nALERTA: Desvio significativo ({desvio_max:.2%}). Possível manipulação de dados ou fraude.")
    else:
        print(f"\nOK: Distribuição de transações segue o padrão natural esperado.")

if __name__ == "__main__":
    validate_fraud_detection_benford()