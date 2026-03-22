import pandas as pd
import os

def manage_model_inventory():
    """
    Simula a manutenção de um inventário centralizado de modelos (Model Inventory).
    Controla finalidade, versão, materialidade e status de validação.
    """
    print("Atualizando Inventário Centralizado de Modelos...")
    
    inventory_data = {
        'model_id': ['MOD-CR-001', 'MOD-MK-002', 'MOD-LQ-003', 'MOD-PR-004'],
        'nome_modelo': ['PD_IFRS9_Varejo', 'VaR_Estatistico_Mesa', 'LCR_Stress_Test', 'Black_Scholes_Call'],
        'finalidade': ['Provisão de Crédito', 'Risco de Mercado', 'Risco de Liquidez', 'Precificação Derivativos'],
        'responsavel': ['Risco de Crédito', 'Tesouraria', 'Financeiro', 'Mesa de Operações'],
        'materialidade': ['Alta', 'Alta', 'Média', 'Baixa'], # Classificação baseada em relevância
        'status_validacao': ['Validado', 'Em Revisão', 'Aprovado com Ressalvas', 'Validado'],
        'ultima_validacao': ['2025-12-10', '2026-03-01', '2026-01-15', '2025-11-20'],
        'versao': ['2.1.0', '1.0.4', '3.0.0', '1.0.0']
    }
    
    df_inventory = pd.DataFrame(inventory_data)
    
    # Lógica de Classificação de Materialidade (Requisito da Vaga)
    # Define que modelos de Crédito e Mercado são sempre 'Alta' exposição
    df_inventory['grau_exposicao'] = df_inventory['materialidade'].map({
        'Alta': 'Crítico', 'Média': 'Monitoramento', 'Baixa': 'Informativo'
    })

    os.makedirs('../data', exist_ok=True)
    df_inventory.to_csv('../data/central_model_inventory.csv', index=False)
    
    print("\n--- Inventário de Modelos (Amostra) ---")
    print(df_inventory[['model_id', 'nome_modelo', 'materialidade', 'status_validacao']])
    print("\nInventário salvo em '../data/central_model_inventory.csv'")

if __name__ == "__main__":
    manage_model_inventory()