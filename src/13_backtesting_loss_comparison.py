import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_performance_backtesting():
    """
    Conduz a avaliação periódica comparando Perdas Estimadas (ECL) 
    vs Perdas Observadas (Realized Loss).
    """
    print("Iniciando Backtesting de Perdas (Estimado vs Realizado)...")
    
    # Simulando 12 meses de dados históricos
    meses = pd.date_range('2025-01-01', periods=12, freq='MS').strftime('%Y-%m')
    np.random.seed(123)
    
    backtest_df = pd.DataFrame({
        'Mes': meses,
        'Perda_Estimada_ECL': [1.2, 1.3, 1.25, 1.4, 1.5, 1.45, 1.6, 1.7, 1.65, 1.8, 1.9, 2.0], # em R$ Milhões
        'Perda_Realizada': [1.15, 1.35, 1.10, 1.45, 1.55, 1.60, 1.50, 1.85, 1.90, 2.10, 2.05, 2.20]
    })
    
    # Indicador de Desvio (Acompanhamento de falhas/limitações)
    backtest_df['Desvio_Percentual'] = (backtest_df['Perda_Realizada'] - backtest_df['Perda_Estimada_ECL']) / backtest_df['Perda_Estimada_ECL']
    
    # Plotagem
    plt.figure(figsize=(12, 6))
    plt.plot(backtest_df['Mes'], backtest_df['Perda_Estimada_ECL'], label='Perda Estimada (ECL)', marker='o', ls='--')
    plt.plot(backtest_df['Mes'], backtest_df['Perda_Realizada'], label='Perda Realizada (Inadimplência Real)', marker='s', color='red')
    
    plt.fill_between(backtest_df['Mes'], backtest_df['Perda_Estimada_ECL'], backtest_df['Perda_Realizada'], color='red', alpha=0.1)
    
    plt.title('Backtesting de Performance: Perda Estimada vs. Realizada', fontsize=14)
    plt.ylabel('Valor em R$ Milhões')
    plt.legend()
    plt.xticks(rotation=45)
    
    os.makedirs('../data/plots', exist_ok=True)
    plt.savefig('../data/plots/loss_backtesting.png', bbox_inches='tight')
    print("Gráfico de Backtesting salvo em '../data/plots/loss_backtesting.png'")
    
    # Alerta de Governança
    desvio_medio = backtest_df['Desvio_Percentual'].mean()
    if desvio_medio > 0.10:
        print(f"ALERTA DE GOVERNANÇA: Modelo subestimando perdas em {desvio_medio*100:.2f}% na média.")

if __name__ == "__main__":
    run_performance_backtesting()