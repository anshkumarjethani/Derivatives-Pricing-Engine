import yfinance as yf
import numpy as np
import pandas as pd
import scipy.stats as norm
from datetime import datetime

def get_spot_price(ticker):
    """
    Fetch the most recent closing price for a given ticker using
    Yahoo Finance historical data (more reliable than .info for
    getting a current price).

    Parameters:
    ticker : Yahoo Finance ticker symbol, e.g. 'RELIANCE.NS', '^NSEI'

    Returns:
    price : most recent closing price (float)
    """
    stock = yf.Ticker(ticker)
    history = stock.history(period="1d")

    if history.empty:
        raise ValueError(f"No price data found for ticker '{ticker}'. Check the symbol is correct.")

    price = history['Close'].iloc[-1]

    return float(price)

def filter_liquid_options(options_df, min_volume=1, min_open_interest=1):
    """
    Filter an option chain DataFrame (from yfinance) to keep only rows
    with genuine trading activity, removing stale/untraded quotes that
    can show implausible prices or implied volatility.

    Parameters:
    options_df        : DataFrame from chain.calls or chain.puts
    min_volume        : minimum today's volume required to keep a row
    min_open_interest : minimum open interest required to keep a row

    Returns:
    filtered_df : DataFrame containing only liquid rows
    """
    has_volume = options_df['volume'].fillna(0)>= min_volume
    has_open_interet = options_df['openInterest'].fillna(0)>= min_open_interest

    filtered_df = options_df[has_volume & has_open_interet]
    return filtered_df

def get_risk_free_rate():
    """
    Fetch the current risk-free rate using the 13-week US Treasury bill
    yield (ticker ^IRX), appropriate for short-dated options (weeks to
    a few months to expiry).

    Note: ^IRX is quoted by Yahoo Finance as a percentage (e.g. 5.25
    meaning 5.25%), so this converts it to decimal form (0.0525) to
    match the convention used throughout this project's pricing functions.

    Returns:
    rate : risk-free rate as a decimal (e.g. 0.0525 for 5.25%)
    """
    treasury = yf.Ticker("^IRX")
    history = treasury.history(period="1d")

    if history.empty:
        raise ValueError("No treasury yield data found ^IRX")

    rate_percent = history['Close'].iloc[-1]

    return float(rate_percent) / 100

def get_liquid_near_the_money_contract(ticker_symbol, option_side='calls', min_days=20):
    """
    Fetches a real option chain, filters for liquidity, and returns the
    contract nearest the current spot price, along with spot price and
    time to expiry.

    Parameters:
    ticker_symbol : e.g. 'AAPL'
    option_side   : 'calls' or 'puts'
    min_days      : minimum days to expiry required (avoids picking
                     contracts with almost no time value left)

    Returns:
    S, K, sigma, T : spot price, strike, implied volatility, time to expiry (years)
    """
    ticker = yf.Ticker(ticker_symbol)
    S = get_spot_price(ticker_symbol)

    expiries = ticker.options
    if len(expiries) == 0:
        raise ValueError(f"No option expiries found for '{ticker_symbol}'.")

    chosen_expiry = None
    for expiry in expiries:
        expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
        days_out = (expiry_date - datetime.now()).days
        if days_out >= min_days:
            chosen_expiry = expiry
            break

    if chosen_expiry is None:
        raise ValueError(f"No expiry found with at least {min_days} days to expiration.")

    chain = ticker.option_chain(chosen_expiry)
    raw_contracts = chain.calls if option_side == 'calls' else chain.puts
    liquid_contracts = filter_liquid_options(raw_contracts, min_volume=10, min_open_interest=100)

    if len(liquid_contracts) == 0:
        raise ValueError(f"No sufficiently liquid {option_side} found for '{ticker_symbol}' at expiry {chosen_expiry}.")

    liquid_contracts = liquid_contracts.copy()
    liquid_contracts['distance_from_spot'] = abs(liquid_contracts['strike'] - S)
    nearest = liquid_contracts.sort_values('distance_from_spot').iloc[0]

    K = float(nearest['strike'])
    sigma = float(nearest['impliedVolatility'])
    T = (datetime.strptime(chosen_expiry, '%Y-%m-%d') - datetime.now()).days / 365

    return S, K, sigma, T

def get_historical_volatility(ticker_symbol, lookback_days= 60):
    """
    Compute annualized historical (realized) volatility from a stock's
    recent daily price history — an independent volatility estimate,
    not derived from any option's price, unlike implied volatility.

    Parameters:
    ticker_symbol : e.g. 'AAPL'
    lookback_days : number of recent trading days to use

    Returns:
    volatility : annualized historical volatility (decimal, e.g. 0.25 for 25%)
    """
    ticker = yf.Ticker(ticker_symbol)
    history = ticker.history(period=f"{lookback_days}d")

    if history.empty or len(history) < 2:
        raise ValueError(f"Not enough price history found for '{ticker_symbol}'.")

    closes = history['Close']
    log_returns = np.log(closes / closes.shift(1)).dropna()

    daily_std = log_returns.std()
    annualised_volatality = daily_std * np.sqrt(252)

    return float(annualised_volatality)



                  



 
     
     