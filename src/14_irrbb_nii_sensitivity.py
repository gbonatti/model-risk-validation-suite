import pandas as pd
import numpy as np

def validate_nii_sensitivity():
    """
    Valida o impacto de choques de juros na Margem Financeira (NII).
    Analisa o descasamento (gap) entre ativos e passivos sensíveis a juros.
    """
    print("--- Auditoria de IRRBB: Sensibilidade do NII (12 meses) ---")
    
    # Gap de Reprecificação (Ativos - Passivos) por faixa temporal
    gaps = pd.DataFrame({
        'Prazo': ['0-30d', '31-90d', '91-180d', '181-360d'],
        'Gap_Reprecificacao': [150_000_000, -50_000_000, 200_000_000, -100_000_000] # R$
    })
    
    choque = 0.02 # +200 bps
    
    # Cálculo do Impacto no NII: Gap * Choque * (Tempo_Restante / 360)
    gaps['Impacto_NII'] = gaps['Gap_Reprecificacao'] * choque
    
    total_impacto = gaps['Impacto_NII'].sum()
    
    print(gaps)
    print(f"\nImpacto Total Estimado no NII (+200bps): R$ {total_impacto:,.2f}")
    
    if total_impacto < 0:
        print("ALERTA: Instituição está 'passiva' em juros. Alta nas taxas reduz a margem financeira.")
    else:
        print("OK: Instituição se beneficia de altas nas taxas de juros no curto prazo.")

if __name__ == "__main__":
    validate_nii_sensitivity()