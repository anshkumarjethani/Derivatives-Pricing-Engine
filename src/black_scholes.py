import pandas as pd
import numpy as np 
from scipy.stats import norm

def black_scholes_call(S, K, r, sigma, T, q=0):
    """
    Price a European call option using the Black-Scholes-Merton formula,
    including continuous dividend yield.

    Parameters:
    S     : current stock price
    K     : strike price
    r     : risk-free interest rate (annualized, decimal e.g. 0.05)
    sigma : volatility (annualized, decimal e.g. 0.2)
    T     : time to expiry, in years
    q     : continuous dividend yield (annualized, decimal e.g. 0.02). Defaults to 0.

    Returns:
    call_price : theoretical price of the call option
    """
    d1 = (np.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return call_price

def black_scholes_put(S, K, r, sigma, T, q=0):
    """
    Price a European put option using the Black-Scholes-Merton formula,
    including continuous dividend yield.

    Parameters:
    S     : current stock price
    K     : strike price
    r     : risk-free interest rate (annualized, decimal e.g. 0.05)
    sigma : volatility (annualized, decimal e.g. 0.2)
    T     : time to expiry, in years
    q     : continuous dividend yield (annualized, decimal e.g. 0.02). Defaults to 0.

    Returns:
    put_price : theoretical price of the put option
    """
    d1 = (np.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    return put_price

def delta_call(S, K, r, sigma, T, q=0):
    """
    Delta of a European call option — sensitivity of price to a
    small change in the stock price S.

    Returns a value between 0 and 1.
    """
    d1 = (np.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    return np.exp(-q * T) * norm.cdf(d1)


def delta_put(S, K, r, sigma, T, q=0):
    """
    Delta of a European put option — sensitivity of price to a
    small change in the stock price S.

    Returns a value between -1 and 0.
    """
    d1 = (np.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    return -np.exp(-q * T) * norm.cdf(-d1)

def gamma(S, K, r, sigma, T, q=0):
    """
    Gamma of a European option (same for calls and puts) — sensitivity
    of Delta to a small change in the stock price S.
    """
    d1 = (np.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    return (np.exp(-q * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))

def vega(S, K, r, sigma, T, q=0):
    """
    Vega of a European option (same for calls and puts) — sensitivity
    of price to a small change in volatility (sigma).

    Note: this returns the change in price per 1.00 (100%) change in
    sigma. By convention, traders usually quote Vega per 1% change —
    divide the result by 100 if you want that convention.
    """
    d1 = (np.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

def theta_call(S, K, r, sigma, T, q=0):
    """
    Theta of a European call option — sensitivity of price to the
    passage of time. Returned as the annualized change in price
    (typically negative). Divide by 365 for an approximate daily decay.
    """
    d1 = (np.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    term1 = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    term3 = q * S * np.exp(-q * T) * norm.cdf(d1)

    return term1 + term2 + term3


def theta_put(S, K, r, sigma, T, q=0):
    """
    Theta of a European put option — sensitivity of price to the
    passage of time. Returned as the annualized change in price
    (typically negative, but less consistently than calls). Divide
    by 365 for an approximate daily decay.
    """
    d1 = (np.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    term1 = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
    term3 = -q * S * np.exp(-q * T) * norm.cdf(-d1)

    return term1 + term2 + term3

def rho_call(S, K, r, sigma, T, q=0):
    """
    Rho of a European call option — sensitivity of price to a small
    change in the risk-free rate r. Returned per 1.00 (100%) change
    in r; divide by 100 for the per-1%-move convention.
    """
    d1 = (np.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * T * np.exp(-r * T) * norm.cdf(d2)


def rho_put(S, K, r, sigma, T, q=0):
    """
    Rho of a European put option — sensitivity of price to a small
    change in the risk-free rate r. Returned per 1.00 (100%) change
    in r; divide by 100 for the per-1%-move convention.
    """
    d1 = (np.log(S / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2)