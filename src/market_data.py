import yfinance as yf
import numpy as np
import pandas as pd
import scipy.stats as norm

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




 
     
     