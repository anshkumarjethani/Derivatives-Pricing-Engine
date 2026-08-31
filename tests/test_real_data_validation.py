import numpy as np
import sys
import os
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import yfinance as yf

from market_data import get_spot_price, filter_liquid_options
from black_scholes import black_scholes_call, black_scholes_put, delta_call, delta_call_numerical
from binomial import binomial_price
from monte_carlo import monte_carlo_call, monte_carlo_put, monte_carlo_call_antithetic, monte_carlo_call_control_variate
from market_data import get_spot_price, get_risk_free_rate, get_liquid_near_the_money_contract, get_historical_volatility


def test_three_methods_agree_on_real_market_data():
    """
    Fetch a real, live ^SPX option chain, pick a liquid near-the-money
    call, and confirm Black-Scholes, binomial, and all three Monte Carlo
    variants (plain, antithetic, control variate) price it closely to
    one another. This is the primary cross-method agreement proof —
    every pricing approach in the project, checked against real,
    unseen market data rather than fixed test numbers.
    """
    S, K, sigma, T = get_liquid_near_the_money_contract("^SPX", option_side='calls')

    r = get_risk_free_rate()
    q = 0.0034  # ^SPX's approximate current dividend yield

    bs_price = black_scholes_call(S=S, K=K, r=r, sigma=sigma, T=T, q=q)
    binom_price = binomial_price(S=S, K=K, r=r, sigma=sigma, T=T, N=500, q=q,
                                  option_type='call', american=False)
    mc_price = monte_carlo_call(S=S, K=K, r=r, sigma=sigma, T=T, q=q,
                                 n_simulations=100000, seed=42)
    mc_anti_price = monte_carlo_call_antithetic(S=S, K=K, r=r, sigma=sigma, T=T, q=q,
                                                 n_simulations=100000, seed=42)
    mc_cv_price = monte_carlo_call_control_variate(S=S, K=K, r=r, sigma=sigma, T=T, q=q,
                                                    n_simulations=100000, seed=42)

    prices = [bs_price, binom_price, mc_price, mc_anti_price, mc_cv_price]
    spread = max(prices) - min(prices)

    relative_spread = spread / np.mean(prices)
    assert relative_spread < 0.02, f"Methods disagree too much on real data (relative): {relative_spread:.4f}, prices: {prices}"


def test_three_methods_agree_on_real_market_data_puts():
    """
    Same cross-method agreement check, for puts — catches any
    put-specific bug (e.g. a sign error) a calls-only test could miss.
    Core 3 methods only (Black-Scholes, binomial, plain Monte Carlo) —
    the antithetic/control-variate techniques are already validated for
    correctness on puts via the Stage 3 synthetic-data test suite; this
    test's job is confirming no put-specific issue on real inputs, which
    3 methods proves as well as 5 would.
    """
    S, K, sigma, T = get_liquid_near_the_money_contract("^SPX", option_side='puts')

    r = get_risk_free_rate()
    q = 0.0034

    bs_price = black_scholes_put(S=S, K=K, r=r, sigma=sigma, T=T, q=q)
    binom_price = binomial_price(S=S, K=K, r=r, sigma=sigma, T=T, N=500, q=q,
                                  option_type='put', american=False)
    mc_price = monte_carlo_put(S=S, K=K, r=r, sigma=sigma, T=T, q=q,
                                n_simulations=100000, seed=42)

    prices = [bs_price, binom_price, mc_price]
    spread = max(prices) - min(prices)

    relative_spread = spread / np.mean(prices)
    assert relative_spread < 0.02, f"Methods disagree too much on real data (relative): {relative_spread:.4f}, prices: {prices}"
    
def test_greeks_agree_on_real_market_data():
    """
    Confirms analytic and numerical (bump-and-revalue) Delta still agree
    closely on real, non-round market inputs. There's no 'real market
    Delta' to compare against directly (option chains don't quote
    Greeks), so this checks internal consistency: do the two
    independently-derived calculation methods still agree on messy,
    real data, not just clean round test numbers.
    """
    S, K, sigma, T = get_liquid_near_the_money_contract("^SPX", option_side='calls')

    r = get_risk_free_rate()
    q = 0.0034

    analytic = delta_call(S=S, K=K, r=r, sigma=sigma, T=T, q=q)
    numerical = delta_call_numerical(S=S, K=K, r=r, sigma=sigma, T=T, q=q)

    assert abs(analytic - numerical) < 0.001, \
        f"Analytic and numerical Delta disagree on real data: {analytic} vs {numerical}"

def test_historical_volatility_returns_sane_value():
    """
    Confirms get_historical_volatility() returns a plausible annualized
    volatility (as a decimal, not a percentage, and within a realistic
    range for a real, liquid stock). This doesn't check historical vol
    against implied vol for closeness — they're expected to genuinely
    differ, since one is backward-looking (realized) and the other is
    forward-looking (market-implied). This test only confirms the
    calculation itself is sound.
    """
    hist_vol = get_historical_volatility("^SPX")

    assert 0 < hist_vol < 2.0, \
        f"Historical volatility out of plausible range: {hist_vol}"
        