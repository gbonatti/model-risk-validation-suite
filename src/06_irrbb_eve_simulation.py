import numpy as np

def calculate_present_value(cashflows, times, rate_curve):
    return np.sum(cashflows / ((1 + rate_curve) ** times))

def simulate_irrbb_shocks():
    np.random.seed(42)
    times = np.random.uniform(1, 10, 1000)
    cashflows = np.where(times > 5, np.random.uniform(500, 1000, 1000), np.random.uniform(-800, -200, 1000))
    base_rate = np.full(1000, 0.10)
    
    eve_base = calculate_present_value(cashflows, times, base_rate)
    eve_up = calculate_present_value(cashflows, times, base_rate + 0.02)
    print(f"EVE Base: R$ {eve_base:,.2f} | EVE +200bps: R$ {eve_up:,.2f}")

if __name__ == "__main__":
    simulate_irrbb_shocks()
