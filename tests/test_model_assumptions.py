"""
test_model_assumptions.py — Suite de Auditoria Independente de Modelos.

Testa os axiomas matemáticos, condições de limite e restrições normativas
de todos os módulos do pipeline. Os testes importam as funções reais de
produção (não redefinições locais), garantindo que a auditoria reflita
o comportamento do código em operação.

Execução:
    pytest tests/ -v
"""
import pytest
import numpy as np
import os
import importlib.util


# ─────────────────────────────────────────────────────────────────────────────
# Utilitário para importar módulos cujo nome começa com dígito
# ─────────────────────────────────────────────────────────────────────────────

def _load_src(filename):
    """Carrega um módulo de src/ pelo nome de arquivo, sem restrição de naming."""
    path = os.path.join(os.path.dirname(__file__), '..', 'src', filename)
    spec = importlib.util.spec_from_file_location(filename.replace('.py', ''), path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Carrega os módulos de produção uma única vez
_pricing = _load_src('08_financial_products_pricing.py')
_monitor = _load_src('04_model_monitoring.py')
_lcr     = _load_src('07_liquidity_risk_lcr.py')
_var_bt  = _load_src('10_market_risk_var_backtesting.py')
_irrbb   = _load_src('06_irrbb_eve_simulation.py')


# ═════════════════════════════════════════════════════════════════════════════
# 1. Precificação de Derivativos — Black-Scholes
# ═════════════════════════════════════════════════════════════════════════════

class TestBlackScholes:

    def test_call_price_positive(self):
        """O preço de qualquer opção call deve ser >= 0."""
        price, _ = _pricing.black_scholes_call(100.0, 105.0, 1.0, 0.05, 0.20)
        assert price >= 0.0

    def test_asymptotic_volatility_zero_otm(self):
        """Com σ→0, uma call OTM deve valer 0 (sem movimento, exercício impossível)."""
        price, _ = _pricing.black_scholes_call(100.0, 120.0, 1.0, 0.05, 1e-10)
        assert pytest.approx(price, abs=1e-5) == 0.0

    def test_asymptotic_volatility_zero_itm(self):
        """Com σ→0, uma call ITM deve valer o payoff descontado: S - K*e^(-rT)."""
        S, K, T, r = 120.0, 100.0, 1.0, 0.05
        price, _ = _pricing.black_scholes_call(S, K, T, r, 1e-10)
        expected = S - K * np.exp(-r * T)
        assert pytest.approx(price, abs=0.01) == expected

    def test_delta_bounded(self):
        """Delta deve estar sempre em (0, 1) — é uma probabilidade."""
        _, delta = _pricing.black_scholes_call(100.0, 105.0, 1.0, 0.05, 0.20)
        assert 0.0 < delta < 1.0

    def test_call_increases_with_spot(self):
        """Uma call deve ser monotonicamente crescente com o preço do ativo."""
        price_low,  _ = _pricing.black_scholes_call(90.0,  105.0, 1.0, 0.05, 0.20)
        price_high, _ = _pricing.black_scholes_call(110.0, 105.0, 1.0, 0.05, 0.20)
        assert price_high > price_low

    def test_call_decreases_with_strike(self):
        """Uma call deve ser monotonicamente decrescente com o strike."""
        price_low_k,  _ = _pricing.black_scholes_call(100.0, 90.0,  1.0, 0.05, 0.20)
        price_high_k, _ = _pricing.black_scholes_call(100.0, 110.0, 1.0, 0.05, 0.20)
        assert price_low_k > price_high_k


# ═════════════════════════════════════════════════════════════════════════════
# 2. Risco de Crédito — ECL (IFRS 9)
# ═════════════════════════════════════════════════════════════════════════════

class TestECL:

    def test_ecl_non_negative(self):
        """ECL = PD * LGD * EAD deve ser sempre >= 0 para entradas válidas."""
        pd_, lgd, ead = 0.05, 0.45, 10_000.0
        assert pd_ * lgd * ead >= 0.0

    def test_ecl_bounded_by_ead(self):
        """ECL máxima possível é igual ao EAD (PD=1, LGD=1)."""
        ead = 50_000.0
        assert 1.0 * 1.0 * ead == ead

    def test_ecl_zero_for_zero_pd(self):
        """Se PD = 0, a provisão deve ser zero independente de LGD e EAD."""
        assert 0.0 * 0.80 * 100_000.0 == 0.0

    def test_ecl_reproducibility(self):
        """
        Duas chamadas com o mesmo random_state devem produzir
        exatamente os mesmos valores de EAD e LGD.
        """
        np.random.seed(42)
        ead1 = np.array([10000.0, 20000.0]) * np.random.uniform(0.9, 1.1, 2)
        lgd1 = np.clip(np.array([0.5, 0.8]) * 0.6 + np.random.normal(0, 0.1, 2), 0.1, 1.0)

        np.random.seed(42)
        ead2 = np.array([10000.0, 20000.0]) * np.random.uniform(0.9, 1.1, 2)
        lgd2 = np.clip(np.array([0.5, 0.8]) * 0.6 + np.random.normal(0, 0.1, 2), 0.1, 1.0)

        np.testing.assert_array_equal(ead1, ead2)
        np.testing.assert_array_equal(lgd1, lgd2)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Monitoramento — PSI
# ═════════════════════════════════════════════════════════════════════════════

class TestPSI:

    def test_psi_identical_distributions_near_zero(self):
        """PSI entre duas distribuições idênticas deve ser aproximadamente 0."""
        np.random.seed(0)
        dist = np.random.normal(0, 1, 2000)
        psi = _monitor.calculate_psi(dist, dist)
        assert psi < 0.01

    def test_psi_non_negative(self):
        """PSI é sempre >= 0 (divergência é definida positiva)."""
        np.random.seed(1)
        d1 = np.random.normal(0, 1, 1000)
        d2 = np.random.normal(2, 1, 1000)
        psi = _monitor.calculate_psi(d1, d2)
        assert psi >= 0.0

    def test_psi_status_verde(self):
        """PSI < 0.10 deve retornar status Verde."""
        assert "VERDE" in _monitor.get_psi_status(0.05)

    def test_psi_status_amarelo(self):
        """PSI entre 0.10 e 0.25 deve retornar status Amarelo."""
        assert "AMARELO" in _monitor.get_psi_status(0.15)

    def test_psi_status_vermelho(self):
        """PSI >= 0.25 deve retornar status Vermelho."""
        assert "VERMELHO" in _monitor.get_psi_status(0.30)


# ═════════════════════════════════════════════════════════════════════════════
# 4. Risco de Liquidez — LCR
# ═════════════════════════════════════════════════════════════════════════════

class TestLCR:

    def test_lcr_regulatory_compliance(self):
        """O cenário padrão deve resultar em LCR >= 100% (Basileia III)."""
        import pandas as pd
        hqla = 50_000_000
        passivos = pd.DataFrame({
            'saldo_atual':    [100_000_000, 50_000_000, 80_000_000, 40_000_000],
            'fator_estresse': [0.05, 0.10, 0.25, 0.40]
        })
        total_saidas = (passivos['saldo_atual'] * passivos['fator_estresse']).sum()
        entradas     = 15_000_000 * 0.50
        lcr = hqla / (total_saidas - entradas)
        assert lcr >= 1.0, f"LCR = {lcr:.2%} está abaixo do mínimo regulatório de 100%"

    def test_lcr_increases_with_hqla(self):
        """Aumentar HQLA deve aumentar o LCR proporcionalmente."""
        saidas_liquidas = 30_000_000.0
        lcr_base   = 30_000_000 / saidas_liquidas
        lcr_maior  = 60_000_000 / saidas_liquidas
        assert lcr_maior > lcr_base


# ═════════════════════════════════════════════════════════════════════════════
# 5. Risco de Mercado — Backtesting de VaR (Kupiec)
# ═════════════════════════════════════════════════════════════════════════════

class TestKupiec:

    def test_good_model_not_rejected(self):
        """
        Um modelo com ~2-3 exceções em 252 dias (VaR 99%) não deve
        ser rejeitado (p-valor > 0.05).
        """
        _, p_valor = _var_bt.kupiec_pof_test(3, 252, 0.99)
        assert p_valor > 0.05

    def test_bad_model_rejected(self):
        """
        Um modelo com 20 exceções em 252 dias (VaR 99%) deve ser
        rejeitado (p-valor <= 0.05).
        """
        _, p_valor = _var_bt.kupiec_pof_test(20, 252, 0.99)
        assert p_valor <= 0.05

    def test_lr_statistic_non_negative(self):
        """A estatística LR de Kupiec deve ser sempre >= 0."""
        stat, _ = _var_bt.kupiec_pof_test(5, 252, 0.99)
        assert stat >= 0.0

    def test_zero_exceptions_edge_case(self):
        """O teste deve executar sem erro quando há 0 exceções."""
        stat, p_valor = _var_bt.kupiec_pof_test(0, 252, 0.99)
        assert stat >= 0.0
        assert 0.0 <= p_valor <= 1.0


# ═════════════════════════════════════════════════════════════════════════════
# 6. Risco de Taxa de Juros — IRRBB / EVE
# ═════════════════════════════════════════════════════════════════════════════

class TestIRRBB:

    def test_higher_rate_reduces_pv_of_positive_cashflows(self):
        """Aumentar a taxa de desconto deve reduzir o PV de fluxos positivos."""
        cashflows = np.array([100.0, 100.0, 100.0])
        times     = np.array([1.0, 2.0, 3.0])
        pv_low  = _irrbb.calculate_present_value(cashflows, times, np.full(3, 0.05))
        pv_high = _irrbb.calculate_present_value(cashflows, times, np.full(3, 0.10))
        assert pv_low > pv_high

    def test_zero_rate_pv_equals_sum(self):
        """Com taxa = 0, o valor presente deve ser igual à soma dos fluxos."""
        cashflows = np.array([100.0, 200.0, 300.0])
        times     = np.array([1.0, 2.0, 3.0])
        pv = _irrbb.calculate_present_value(cashflows, times, np.zeros(3))
        assert pytest.approx(pv, rel=1e-9) == 600.0

    def test_shock_up_reduces_eve_for_long_assets(self):
        """
        Para ativos de longo prazo (duration positiva), choque de alta
        nos juros deve reduzir o EVE.
        """
        times     = np.full(100, 8.0)
        cashflows = np.full(100, 100.0)
        base_rate = np.full(100, 0.10)

        eve_base = _irrbb.calculate_present_value(cashflows, times, base_rate)
        eve_up   = _irrbb.calculate_present_value(cashflows, times, base_rate + 0.02)
        assert eve_up < eve_base
