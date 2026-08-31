import pandas as pd
import yfinance as yf
from datetime import datetime

from market_data import get_spot_price, filter_liquid_options, get_risk_free_rate
from black_scholes import (
    black_scholes_call, black_scholes_put,
    delta_call, delta_put, gamma, vega, theta_call, theta_put, rho_call, rho_put,
)


def build_greeks_dashboard(ticker_symbol, option_side='calls', min_days=20, q=0.0):
    """
    Build a table of theoretical price and all Greeks for every liquid
    contract in a real, live option chain.

    Parameters:
    ticker_symbol : e.g. '^SPX'
    option_side   : 'calls' or 'puts'
    min_days      : minimum days to expiry to include (skips near-zero
                    time-to-expiry contracts, which can behave oddly
                    in the binomial model)
    q             : dividend yield to use (annualized decimal)

    Returns:
    dashboard : DataFrame with strike, market bid/ask, model price, and
                all 5 Greeks for each liquid contract
    """
    ticker = yf.Ticker(ticker_symbol)
    S = get_spot_price(ticker_symbol)
    r = get_risk_free_rate()

    expiries = ticker.options
    if len(expiries) == 0:
        raise ValueError(f"No option expiries found for '{ticker_symbol}'.")

    chosen_expiry = None
    for expiry in expiries:
        days_out = (datetime.strptime(expiry, '%Y-%m-%d') - datetime.now()).days
        if days_out >= min_days:
            chosen_expiry = expiry
            break

    if chosen_expiry is None:
        raise ValueError(f"No expiry found with at least {min_days} days to expiration.")

    T = (datetime.strptime(chosen_expiry, '%Y-%m-%d') - datetime.now()).days / 365

    chain = ticker.option_chain(chosen_expiry)
    raw_contracts = chain.calls if option_side == 'calls' else chain.puts
    liquid_contracts = filter_liquid_options(raw_contracts, min_volume=10, min_open_interest=100)

    if len(liquid_contracts) == 0:
        raise ValueError(f"No sufficiently liquid {option_side} found for '{ticker_symbol}'.")

    rows = []
    for _, contract in liquid_contracts.iterrows():
        K = float(contract['strike'])
        sigma = float(contract['impliedVolatility'])

        if sigma <= 0:
            continue

        if option_side == 'calls':
            price = black_scholes_call(S=S, K=K, r=r, sigma=sigma, T=T, q=q)
            delta = delta_call(S=S, K=K, r=r, sigma=sigma, T=T, q=q)
            theta = theta_call(S=S, K=K, r=r, sigma=sigma, T=T, q=q)
            rho = rho_call(S=S, K=K, r=r, sigma=sigma, T=T, q=q)
        else:
            price = black_scholes_put(S=S, K=K, r=r, sigma=sigma, T=T, q=q)
            delta = delta_put(S=S, K=K, r=r, sigma=sigma, T=T, q=q)
            theta = theta_put(S=S, K=K, r=r, sigma=sigma, T=T, q=q)
            rho = rho_put(S=S, K=K, r=r, sigma=sigma, T=T, q=q)

        gamma_val = gamma(S=S, K=K, r=r, sigma=sigma, T=T, q=q)
        vega_val = vega(S=S, K=K, r=r, sigma=sigma, T=T, q=q)

        rows.append({
            'strike': K,
            'market_bid': float(contract['bid']),
            'market_ask': float(contract['ask']),
            'model_price': round(price, 2),
            'implied_vol': round(sigma, 4),
            'delta': round(delta, 4),
            'gamma': round(gamma_val, 6),
            'vega': round(vega_val, 4),
            'theta_daily': round(theta / 365, 4),
            'rho': round(rho, 4),
        })

    dashboard = pd.DataFrame(rows)
    dashboard = dashboard.sort_values('strike').reset_index(drop=True) 

    return S, dashboard
       
