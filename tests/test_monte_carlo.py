import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from monte_carlo import (
    monte_carlo_call,
    monte_carlo_call_antithetic,
    monte_carlo_call_control_variate,
    monte_carlo_put,
    monte_carlo_put_antithetic,
    monte_carlo_put_control_variate,
)
from black_scholes import black_scholes_call, black_scholes_put


# ---- Convergence to Black-Scholes ----

def test_mc_call_converges_to_black_scholes():
    bs_price = black_scholes_call(S=100, K=100, r=0.05, sigma=0.2, T=1)
    mc_price = monte_carlo_call(S=100, K=100, r=0.05, sigma=0.2, T=1,
                                 n_simulations=1000000, seed=42)

    assert abs(bs_price - mc_price) < 0.05


def test_mc_put_converges_to_black_scholes():
    bs_price = black_scholes_put(S=100, K=100, r=0.05, sigma=0.2, T=1)
    mc_price = monte_carlo_put(S=100, K=100, r=0.05, sigma=0.2, T=1,
                                n_simulations=1000000, seed=42)

    assert abs(bs_price - mc_price) < 0.05


def test_mc_call_antithetic_converges_to_black_scholes():
    bs_price = black_scholes_call(S=100, K=100, r=0.05, sigma=0.2, T=1)
    mc_price = monte_carlo_call_antithetic(S=100, K=100, r=0.05, sigma=0.2, T=1,
                                            n_simulations=1000000, seed=42)

    assert abs(bs_price - mc_price) < 0.05


def test_mc_put_antithetic_converges_to_black_scholes():
    bs_price = black_scholes_put(S=100, K=100, r=0.05, sigma=0.2, T=1)
    mc_price = monte_carlo_put_antithetic(S=100, K=100, r=0.05, sigma=0.2, T=1,
                                           n_simulations=1000000, seed=42)

    assert abs(bs_price - mc_price) < 0.05


def test_mc_call_control_variate_converges_to_black_scholes():
    bs_price = black_scholes_call(S=100, K=100, r=0.05, sigma=0.2, T=1)
    mc_price = monte_carlo_call_control_variate(S=100, K=100, r=0.05, sigma=0.2, T=1,
                                                 n_simulations=1000000, seed=42)

    assert abs(bs_price - mc_price) < 0.05


def test_mc_put_control_variate_converges_to_black_scholes():
    bs_price = black_scholes_put(S=100, K=100, r=0.05, sigma=0.2, T=1)
    mc_price = monte_carlo_put_control_variate(S=100, K=100, r=0.05, sigma=0.2, T=1,
                                                n_simulations=1000000, seed=42)

    assert abs(bs_price - mc_price) < 0.05


# ---- Reproducibility (same seed => same result) ----

def test_seed_gives_reproducible_result():
    price1 = monte_carlo_call(S=100, K=100, r=0.05, sigma=0.2, T=1,
                               n_simulations=10000, seed=123)
    price2 = monte_carlo_call(S=100, K=100, r=0.05, sigma=0.2, T=1,
                               n_simulations=10000, seed=123)

    assert price1 == price2


# ---- Variance reduction actually reduces spread ----

def test_antithetic_reduces_spread_vs_plain():
    """Across repeated runs at equal n_simulations, antithetic variance
    reduction should produce a tighter spread than plain Monte Carlo."""
    np.random.seed(1)  # seeds the *test's* randomness for reproducible spread comparison
    plain_runs = [monte_carlo_call(S=100, K=100, r=0.05, sigma=0.2, T=1, n_simulations=5000)
                  for _ in range(20)]

    np.random.seed(1)
    antithetic_runs = [monte_carlo_call_antithetic(S=100, K=100, r=0.05, sigma=0.2, T=1, n_simulations=5000)
                        for _ in range(20)]

    plain_spread = np.std(plain_runs)
    antithetic_spread = np.std(antithetic_runs)

    assert antithetic_spread < plain_spread


def test_control_variate_reduces_spread_vs_plain():
    """Same idea as above, but for the control variate technique."""
    np.random.seed(2)
    plain_runs = [monte_carlo_call(S=100, K=100, r=0.05, sigma=0.2, T=1, n_simulations=5000)
                  for _ in range(20)]

    np.random.seed(2)
    cv_runs = [monte_carlo_call_control_variate(S=100, K=100, r=0.05, sigma=0.2, T=1, n_simulations=5000)
               for _ in range(20)]

    plain_spread = np.std(plain_runs)
    cv_spread = np.std(cv_runs)

    assert cv_spread < plain_spread