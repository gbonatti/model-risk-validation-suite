import pandas as pd
import numpy as np
import os

def generate_credit_data(n_samples=50000, random_state=42):
    """
    Gera dataset sintético para modelagem de risco de crédito (PD).
    """
    np.random.seed(random_state)
    
    # 1. Geração das variáveis do cliente e do empréstimo
    data = {
        'id_cliente': np.arange(1, n_samples + 1),
        'idade': np.random.randint(18, 75, n_samples),
        'renda_mensal': np.random.lognormal(mean=8.5, sigma=0.8, size=n_samples),
        'score_bureau': np.random.randint(300, 850, n_samples),
        'valor_emprestimo': np.random.lognormal(mean=9.0, sigma=0.9, size=n_samples),
        'taxa_juros': np.random.uniform(0.05, 0.25, n_samples),
        'LTV': np.random.uniform(0.3, 1.2, n_samples),
        'safra': np.random.choice(pd.date_range('2022-01-01', '2023-12-01', freq='MS'), n_samples)
    }
    df = pd.DataFrame(data)
    
    # 2. Target gerada antes de inserir o erro (Risco Real)
    z = (
        - df['score_bureau'] * 0.015 
        + df['LTV'] * 3.5 
        - np.log1p(df['renda_mensal']) * 0.5 
        + df['taxa_juros'] * 10 
        + np.random.normal(0, 2, n_samples)
    )
    
    prob_default = 1 / (1 + np.exp(-z))
    df['default_12m'] = (prob_default > np.percentile(prob_default, 85)).astype(int) 
    
    # 3. Inserindo ruído (DQ) — random_state garante reprodutibilidade
    df.loc[df.sample(frac=0.02, random_state=random_state).index, 'renda_mensal'] = np.nan
    df.loc[df.sample(frac=0.01, random_state=random_state + 1).index, 'score_bureau'] = -999
    
    return df

if __name__ == "__main__":
    print("Gerando dataset sintético corrigido...")
    df = generate_credit_data()
    
    # Cria a pasta data um nível acima (na raiz)
    os.makedirs('../data', exist_ok=True)
    df.to_parquet('../data/credit_portfolio.parquet')
    
    print("Dataset salvo com sucesso na pasta '../data/credit_portfolio.parquet'")