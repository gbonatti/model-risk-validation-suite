import pandas as pd

def validate_liquidity_survival():
    """
    Valida o horizonte de sobrevivência da instituição em cenário de estresse severo.
    """
    print("--- Validação de Risco de Liquidez: Horizonte de Sobrevivência ---")
    
    fluxo_estressado = pd.DataFrame({
        'Dia': range(1, 11),
        'Entradas_Esperadas': [10, 8, 5, 2, 1, 1, 0, 0, 0, 0], # Milhões
        'Saídas_Estresse': [15, 12, 10, 8, 7, 5, 5, 5, 5, 5]    # Milhões
    })
    
    caixa_inicial = 45 # Milhões
    fluxo_estressado['Fluxo_Liquido'] = fluxo_estressado['Entradas_Esperadas'] - fluxo_estressado['Saídas_Estresse']
    fluxo_estressado['Posicao_Acumulada'] = caixa_inicial + fluxo_estressado['Fluxo_Liquido'].cumsum()
    
    print(fluxo_estressado)
    
    dias_sobrevivencia = fluxo_estressado[fluxo_estressado['Posicao_Acumulada'] < 0].index.min()
    
    if pd.isna(dias_sobrevivencia):
        print("\nSTATUS: Aprovado. Caixa suporta mais de 10 dias de estresse severo.")
    else:
        print(f"\nCRÍTICO: Caixa esgota no Dia {dias_sobrevivencia + 1}. Abaixo do apetite de risco.")

if __name__ == "__main__":
    validate_liquidity_survival()