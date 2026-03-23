import numpy as np
from scipy.stats import norm


def black_scholes_call(S, K, T, r, sigma):
    """
    Precifica uma opção de compra europeia pelo modelo de Black-Scholes (1973).

    Fórmula:
        C = S * N(d1) - K * e^(-rT) * N(d2)
        d1 = [ln(S/K) + (r + σ²/2) * T] / (σ * √T)
        d2 = d1 - σ * √T

    Parâmetros
    ----------
    S     : float — Preço atual do ativo subjacente
    K     : float — Preço de exercício (strike)
    T     : float — Tempo até o vencimento (em anos)
    r     : float — Taxa de juros livre de risco (contínua)
    sigma : float — Volatilidade anualizada do ativo

    Retorna
    -------
    (price, delta) : tuple
        price — Preço justo da call
        delta — Sensibilidade ao preço do ativo (∂C/∂S = N(d1))
    """
    # Guarda para volatilidade nula: opção vira payoff determinístico
    if sigma <= 1e-8:
        intrinsic = max(0.0, S - K * np.exp(-r * T))
        delta = 1.0 if S > K * np.exp(-r * T) else 0.0
        return intrinsic, delta

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)

    return price, delta


if __name__ == "__main__":
    # Exemplo: call OTM — S=100, K=105, T=1 ano, r=5%, σ=20%
    price, delta = black_scholes_call(S=100.0, K=105.0, T=1.0, r=0.05, sigma=0.20)

    print("─" * 45)
    print("  Precificação Black-Scholes — Call Europeia")
    print("─" * 45)
    print(f"  Preço Spot (S)   : R$ 100,00")
    print(f"  Strike (K)       : R$ 105,00")
    print(f"  Vencimento (T)   : 1 ano")
    print(f"  Taxa Livre (r)   : 5,00%")
    print(f"  Volatilidade (σ) : 20,00%")
    print("─" * 45)
    print(f"  Preço da Call    : R$ {price:.3f}")
    print(f"  Delta (∂C/∂S)    : {delta:.3f}")
    print("─" * 45)
