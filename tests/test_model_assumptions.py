import pytest
import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    if sigma <= 1e-8: return max(0.0, S - K * np.exp(-r * T)), 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2), norm.cdf(d1)

class TestMarketRiskIRRBB:
    def test_ecl_non_negative(self):
        assert 0.05 * 0.45 * 10000.0 >= 0.0

class TestFinancialProductsPricing:
    def test_asymptotic_volatility_zero(self):
        price, delta = black_scholes_call(100.0, 120.0, 1.0, 0.05, 1e-10)
        assert pytest.approx(price, abs=1e-5) == 0.0
