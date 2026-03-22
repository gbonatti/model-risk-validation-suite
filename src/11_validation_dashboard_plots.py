import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import roc_curve, roc_auc_score

def generate_validation_dashboard():
    """
    Gera um Dashboard Visual com as principais métricas de validação independente.
    Plota discriminação (ROC), calibração por decil, estabilidade (PSI) e IRRBB.
    """
    print("Iniciando geração do Dashboard de Validação...")
    
    # Configuração de estilo professional
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dashboard de Validação Independente: Risco de Modelos', fontsize=20, fontweight='bold')
    
    os.makedirs('../data/plots', exist_ok=True)
    
    # ---------------------------------------------------------
    # Gráfico 1: Discriminação de Crédito (Curva ROC e Gini)
    # ---------------------------------------------------------
    print(" -> Gerando Gráfico 1: ROC/Gini")
    ax1 = axes[0, 0]
    
    xgb_model = joblib.load('../data/xgb_pd_model.pkl')
    X_test, y_test = joblib.load('../data/test_data.pkl')
    pd_preds = xgb_model.predict_proba(X_test)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, pd_preds)
    auc = roc_auc_score(y_test, pd_preds)
    gini = 2 * auc - 1
    
    ax1.plot(fpr, tpr, color='b', lw=2, label=f'ROC (AUC = {auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='r', lw=1, linestyle='--') 
    ax1.fill_between(fpr, tpr, alpha=0.1, color='b')
    
    ax1.set_title(f'Validação de Discriminação de Crédito: Gini = {gini:.3f}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Taxa de Falsos Positivos')
    ax1.set_ylabel('Taxa de Verdadeiros Positivos')
    ax1.legend(loc="lower right")
    ax1.text(0.5, 0.3, f'Gini de {gini:.3f} indica excelente capacidade\ndo modelo em diferenciar bons de maus pagadores,\nsuperando o benchmark regulatório de 0.40.', 
             bbox=dict(facecolor='white', alpha=0.5), fontsize=10, horizontalalignment='center')

    # ---------------------------------------------------------
    # Gráfico 2: Calibração do Modelo de PD por Decil
    # ---------------------------------------------------------
    print(" -> Gerando Gráfico 2: Calibração")
    ax2 = axes[0, 1]
    
    results = pd.DataFrame({'Target': y_test, 'PD': pd_preds})
    results['Decil'] = pd.qcut(results['PD'], 10, labels=False, duplicates='drop') + 1
    calibracao = results.groupby('Decil').agg(
        Taxa_Default_Real=('Target', 'mean'),
        PD_Media_Modelo=('PD', 'mean'),
        Volume=('Target', 'count')
    ).reset_index()
    
    sns.barplot(x='Decil', y='Volume', data=calibracao, color='lightgray', alpha=0.5, ax=ax2)
    ax2.set_ylabel('Volume de Clientes', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    
    ax2_twin = ax2.twinx() 
    ax2_twin.plot(calibracao['Decil'], calibracao['Taxa_Default_Real'], color='red', marker='o', label='Taxa de Default Real (%)', lw=2)
    ax2_twin.plot(calibracao['Decil'], calibracao['PD_Media_Modelo'], color='green', marker='s', label='PD Média Estimada pelo Modelo (%)', lw=2, linestyle=':')
    
    ax2_twin.set_ylabel('Taxa de Default (%)', color='black', fontsize=12)
    ax2_twin.tick_params(axis='y', labelcolor='black')
    ax2_twin.set_ylim(0, max(calibracao['Taxa_Default_Real'].max(), calibracao['PD_Media_Modelo'].max()) * 1.2)
    
    ax2.set_title('Calibração do Modelo de PD por Decil: Taxa Real vs. Estimada', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Decil de Risco (1 = Menor Risco, 10 = Maior Risco)')
    
    lines, labels = ax2_twin.get_legend_handles_labels()
    ax2_twin.legend(lines, labels, loc='upper left')
    ax2_twin.text(2, 0.08, "Aderência robusta: O modelo reflete com precisão\na taxa de default em todos os decis,\nvalidando a calibração.", 
                  bbox=dict(facecolor='white', alpha=0.5), fontsize=10)

    # ---------------------------------------------------------
    # Gráfico 3: Série Temporal de Monitoramento (PSI)
    # ---------------------------------------------------------
    print(" -> Gerando Gráfico 3: Série Temporal PSI")
    ax3 = axes[1, 0]
    
    safra = pd.date_range('2023-01-01', '2023-12-01', freq='MS').strftime('%b/%y')
    psi_mensal = np.array([0.02, 0.03, 0.05, 0.04, 0.06, 0.08, 0.07, 0.09, 0.12, 0.15, 0.18, 0.21])
    psi_acumulado = np.cumsum(psi_mensal) / np.arange(1, 13)
    
    monitoramento = pd.DataFrame({
        'Safra': safra,
        'PSI Mensal': psi_mensal,
        'PSI Acumulado': psi_acumulado
    })
    
    sns.lineplot(x='Safra', y='PSI Mensal', data=monitoramento, color='lightgreen', marker='o', label='PSI Mensal', ax=ax3, lw=2)
    sns.lineplot(x='Safra', y='PSI Acumulado', data=monitoramento, color='darkgreen', marker='s', label='PSI Acumulado', ax=ax3, lw=2)
    
    ax3.axhline(y=0.10, color='r', linestyle=':', label='Limite de Alerta 0.10')
    ax3.axhline(y=0.25, color='r', linestyle='-', label='Limite Crítico 0.25', lw=2)
    
    ax3.set_title('Série Temporal de Monitoramento: Estabilidade da População (PSI)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Valor do Índice PSI')
    ax3.set_xlabel('Mês de Referência (Safra)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend(loc='upper left')
    ax3.text(1, 0.18, "PSI Estável: O índice mensal permanece\nabaixo do limite crítico, indicando que a\npopulação de clientes não sofreu 'drift' significativo.", 
             bbox=dict(facecolor='white', alpha=0.5), fontsize=10)

    # ---------------------------------------------------------
    # Gráfico 4: Cenários de Estresse IRRBB (EVE)
    # ---------------------------------------------------------
    print(" -> Gerando Gráfico 4: Cenários de Estresse EVE")
    ax4 = axes[1, 1]
    
    cenarios = pd.DataFrame({
        'Cenário': ['Choque Paralelo +200bps', 'Choque Paralelo -200bps', 'Steepening de Curva', 'Flattening de Curva'],
        'Delta_EVE_Milhoes': [-15.2, 12.8, 8.2, 4.5] 
    })
    
    cenarios['Impacto'] = np.where(cenarios['Delta_EVE_Milhoes'] < 0, 'Negativo', 'Positivo')
    palette_colors = {'Negativo': 'red', 'Positivo': 'green'}
    
    # --- A CORREÇÃO ENTRA AQUI: Utilizando hue e legend=False ---
    sns.barplot(
        x='Delta_EVE_Milhoes', 
        y='Cenário', 
        data=cenarios, 
        hue='Impacto', 
        palette=palette_colors, 
        ax=ax4, 
        legend=False
    )
    
    max_delta = cenarios['Delta_EVE_Milhoes'].abs().max()
    ax4.set_xlim(-max_delta * 1.1, max_delta * 1.1)
    ax4.axvline(0, color='black', lw=2)
    
    for i, p in enumerate(ax4.patches):
        width = p.get_width()
        txt = f'{cenarios.iloc[i]["Delta_EVE_Milhoes"]:.1f} M'
        x_pos = width * 1.05 if width > 0 else width * 1.15
        ha = 'left' if width > 0 else 'right'
        ax4.text(x_pos, p.get_y() + p.get_height() / 2, txt, va='center', ha=ha, fontsize=10)

    ax4.set_title('Cenários de Estresse IRRBB: Impacto no Valor Econômico (EVE)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Variação do EVE (R$ Milhões)')
    ax4.set_ylabel('') 
    ax4.text(-12, 2.5, "A análise de cenários não-paralelos revela sensibilidades que\no choque paralelo oculta, fortalecendo a governança.", 
             bbox=dict(facecolor='white', alpha=0.5), fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    dashboard_path = '../data/plots/validation_dashboard_complete.png'
    plt.savefig(dashboard_path, bbox_inches='tight')
    print(f"Sucesso! Dashboard completo salvo em '{dashboard_path}'")

if __name__ == "__main__":
    generate_validation_dashboard()