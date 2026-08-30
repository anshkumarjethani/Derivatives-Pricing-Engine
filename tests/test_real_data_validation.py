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

def _get_liquid_near_the_money_contract(ticker_symbol, option_side='calls', min_days=20):
    """
    Shared helper: fetches a real option chain, filters for liquidity,
    and returns the contract nearest the current spot price, along with
    spot price and time to expiry. Used by all real-data tests below to
    avoid repeating the same fetch/filter/select logic in each one.
    """
    ticker = yf.Ticker(ticker_symbol)
    S = get_spot_price(ticker_symbol)

    expiries = ticker.options
    assert len(expiries) > 0, "No option expiries returned — data may be unavailable"

    chosen_expiry = None
    for expiry in expiries:
        expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
        days_out = (expiry_date - datetime.now()).days
        if days_out >= min_days:
            chosen_expiry = expiry
            break

    assert chosen_expiry is not None, f"No expiry found with at least {min_days} days to expiration"

    chain = ticker.option_chain(chosen_expiry)
    raw_contracts = chain.calls if option_side == 'calls' else chain.puts
    liquid_contracts = filter_liquid_options(raw_contracts, min_volume=10, min_open_interest=100)

    assert len(liquid_contracts) > 0, f"No sufficiently liquid {option_side} found for this expiry"

    liquid_contracts = liquid_contracts.copy()
    liquid_contracts['distance_from_spot'] = abs(liquid_contracts['strike'] - S)
    nearest = liquid_contracts.sort_values('distance_from_spot').iloc[0]

    K = float(nearest['strike'])
    sigma = float(nearest['impliedVolatility'])

    expiry_date = datetime.strptime(chosen_expiry, '%Y-%m-%d')
    T = (expiry_date - datetime.now()).days / 365

    return S, K, sigma, T


def test_three_methods_agree_on_real_market_data():
    """
    Fetch a real, live AAPL option chain, pick a liquid near-the-money
    call, and confirm Black-Scholes, binomial, and all three Monte Carlo
    variants (plain, antithetic, control variate) price it closely to
    one another. This is the primary cross-method agreement proof —
    every pricing approach in the project, checked against real,
    unseen market data rather than fixed test numbers.
    """
    S, K, sigma, T = _get_liquid_near_the_money_contract("AAPL", option_side='calls')

    r = 0.05
    q = 0.0034  # AAPL's approximate current dividend yield

    bs_price = black_scholes_call(S=S, K=K, r=r, sigma=sigma, T=T, q=q)
    binom_price = binomial_price(S=S, K=K, r=r, sigma=sigma, T=T, N=500, q=q,
                                  option_type='call', american=True)
    mc_price = monte_carlo_call(S=S, K=K, r=r, sigma=sigma, T=T, q=q,
                                 n_simulations=100000, seed=42)
    mc_anti_price = monte_carlo_call_antithetic(S=S, K=K, r=r, sigma=sigma, T=T, q=q,
                                                 n_simulations=100000, seed=42)
    mc_cv_price = monte_carlo_call_control_variate(S=S, K=K, r=r, sigma=sigma, T=T, q=q,
                                                    n_simulations=100000, seed=42)

    prices = [bs_price, binom_price, mc_price, mc_anti_price, mc_cv_price]
    spread = max(prices) - min(prices)

    assert spread < 0.10, f"Methods disagree too much on real data: {prices}"


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
    S, K, sigma, T = _get_liquid_near_the_money_contract("AAPL", option_side='puts')

    r = 0.05
    q = 0.0034

    bs_price = black_scholes_put(S=S, K=K, r=r, sigma=sigma, T=T, q=q)
    binom_price = binomial_price(S=S, K=K, r=r, sigma=sigma, T=T, N=500, q=q,
                                  option_type='put', american=True)
    mc_price = monte_carlo_put(S=S, K=K, r=r, sigma=sigma, T=T, q=q,
                                n_simulations=100000, seed=42)

    prices = [bs_price, binom_price, mc_price]
    spread = max(prices) - min(prices)

    assert spread < 0.10, f"Put methods disagree too much on real data: {prices}"


def test_greeks_agree_on_real_market_data():
    """
    Confirms analytic and numerical (bump-and-revalue) Delta still agree
    closely on real, non-round market inputs. There's no 'real market
    Delta' to compare against directly (option chains don't quote
    Greeks), so this checks internal consistency: do the two
    independently-derived calculation methods still agree on messy,
    real data, not just clean round test numbers.
    """
    S, K, sigma, T = _get_liquid_near_the_money_contract("AAPL", option_side='calls')

    r = 0.05
    q = 0.0034

    analytic = delta_call(S=S, K=K, r=r, sigma=sigma, T=T, q=q)
    numerical = delta_call_numerical(S=S, K=K, r=r, sigma=sigma, T=T, q=q)

    assert abs(analytic - numerical) < 0.001, \
        f"Analytic and numerical Delta disagree on real data: {analytic} vs {numerical}"