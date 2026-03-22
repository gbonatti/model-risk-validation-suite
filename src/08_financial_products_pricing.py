import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), norm.cdf(d1)

if __name__ == "__main__":
    price, delta = black_scholes_call(100.0, 105.0, 1.0, 0.05, 0.20)
    print(f"Call Price: R$ {price:.3f} | Delta: {delta:.3f}")
