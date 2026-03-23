import pandas as pd
import numpy as np
import joblib


def calculate_ecl(random_state=42):
    """
    Calcula a Perda de Crédito Esperada (ECL) conforme IFRS 9 / CPC 48.

    Equação fundamental:
        ECL = PD × LGD × EAD

    Parâmetros
    ----------
    random_state : int
        Semente aleatória para garantir reprodutibilidade dos fatores
        estocásticos de EAD e LGD.
    """
    np.random.seed(random_state)  # Garante reprodutibilidade

    X_test, _ = joblib.load('../data/test_data.pkl')
    xgb_model = joblib.load('../data/xgb_pd_model.pkl')

    # ── Probability of Default (PD) ─────────────────────────────────────────
    # Saída do modelo XGBoost para horizonte de 12 meses (Estágio 1 - IFRS 9)
    pd_12m = xgb_model.predict_proba(X_test)[:, 1]

    # ── Exposure at Default (EAD) ────────────────────────────────────────────
    # Fator de conversão entre 90%-110% do saldo contratual
    # Simula saques adicionais em linhas de crédito rotativas (CCF)
    ead = X_test['valor_emprestimo'] * np.random.uniform(0.9, 1.1, len(X_test))

    # ── Loss Given Default (LGD) ─────────────────────────────────────────────
    # LGD = LTV * 0.6 (taxa histórica de perda de ~60%, recuperação de ~40%)
    # + ruído idiossincrático por cliente
    lgd = np.clip(
        X_test['LTV'] * 0.6 + np.random.normal(0, 0.1, len(X_test)),
        0.10,  # LGD mínimo: 10%
        1.00   # LGD máximo: 100%
    )

    # ── ECL por cliente ───────────────────────────────────────────────────────
    ecl = pd_12m * lgd * ead

    print("─" * 55)
    print("  Cálculo de ECL — IFRS 9 (Estágio 1, Horizonte 12m)")
    print("─" * 55)
    print(f"  Clientes analisados    : {len(X_test):,}")
    print(f"  Exposição Total (EAD)  : R$ {ead.sum():>15,.2f}")
    print(f"  LGD Médio da Carteira  : {lgd.mean():.2%}")
    print(f"  PD Médio da Carteira   : {pd_12m.mean():.2%}")
    print(f"  Provisão Total (ECL)   : R$ {ecl.sum():>15,.2f}")
    print(f"  Índice de Cobertura    : {ecl.sum() / ead.sum():.2%}")
    print("─" * 55)

    return {"ead_total": ead.sum(), "ecl_total": ecl.sum(), "pd_medio": pd_12m.mean(), "lgd_medio": lgd.mean()}


if __name__ == "__main__":
    calculate_ecl()
