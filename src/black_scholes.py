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

def delta_call_numerical(S, K, r, sigma, T, q=0, h=0.01):
    """
    Delta of a European call, computed via central finite difference
    (bump-and-revalue) instead of the analytic formula. Should closely
    match delta_call() — this is used to validate the analytic result.

    h : the bump size for S. Small enough to approximate a derivative,
        large enough to avoid floating-point precision issues.
    """
    price_up = black_scholes_call(S + h, K, r, sigma, T, q)
    price_down = black_scholes_call(S - h, K, r, sigma, T, q)

    return (price_up - price_down) / (2 * h)

def delta_put_numerical(S, K, r, sigma, T, q=0, h=0.01):
    """
    Delta of a European put, computed via central finite difference
    (bump-and-revalue) instead of the analytic formula. Should closely
    match delta_put() — this is used to validate the analytic result.

    h : the bump size for S. Small enough to approximate a derivative,
        large enough to avoid floating-point precision issues.
    """
    price_up = black_scholes_put(S + h, K, r, sigma, T, q)
    price_down = black_scholes_put(S - h, K, r, sigma, T, q)

    return (price_up - price_down) / (2 * h)

def gamma_numerical(S, K, r, sigma, T, q=0, h=0.01):
    """
    Gamma of a European option, computed via the central finite-difference
    second-derivative formula (bump-and-revalue) instead of the analytic
    formula. Should closely match gamma() — used to validate the analytic result.

    h : the bump size for S.
    """
    price_up = black_scholes_call(S + h, K, r, sigma, T, q)
    price_now = black_scholes_call(S, K, r, sigma, T, q)
    price_down = black_scholes_call(S - h, K, r, sigma, T, q)

    return (price_up - 2 * price_now + price_down) / h**2

def vega_numerical(S, K, r, sigma, T, q=0, h=0.01):
    """
    Vega of a European option, computed via central finite difference
    (bump-and-revalue) instead of the analytic formula. Should closely
    match vega() — used to validate the analytic result.

    Note: same as the analytic vega(), this returns the change in price
    per 1.00 (100%) change in sigma. Divide by 100 for the per-1%-move
    convention.

    h : the bump size for sigma.
    """
    price_up = black_scholes_call(S, K, r, sigma + h, T, q)
    price_down = black_scholes_call(S, K, r, sigma - h, T, q)

    return (price_up - price_down) / (2 * h)

def theta_call_numerical(S, K, r, sigma, T, q=0, h=0.001):
    """
    Theta of a European call option, computed via central finite
    difference (bump-and-revalue) instead of the analytic formula.
    Should closely match theta_call() — used to validate the analytic
    result.

    Note: matches theta_call()'s convention — returned as the annualized
    change in price per year of time passing (typically negative).

    h : the bump size for T. Kept smaller than the default used for S
        or sigma, since T is often close to 1 or smaller — a large h
        would represent too large a relative change to give an accurate
        approximation, especially for short-dated options.
    """
    price_up = black_scholes_call(S, K, r, sigma, T + h, q)
    price_down = black_scholes_call(S, K, r, sigma, T - h, q)

    return (price_down - price_up) / (2 * h)

def theta_put_numerical(S, K, r, sigma, T, q=0, h=0.001):
    """
    Theta of a European put option, computed via central finite
    difference (bump-and-revalue) instead of the analytic formula.
    Should closely match theta_call() — used to validate the analytic
    result.

    Note: matches theta_put()'s convention — returned as the annualized
    change in price per year of time passing (typically negative).

    h : the bump size for T. Kept smaller than the default used for S
        or sigma, since T is often close to 1 or smaller — a large h
        would represent too large a relative change to give an accurate
        approximation, especially for short-dated options.
    """
    price_up = black_scholes_put(S, K, r, sigma, T + h, q)
    price_down = black_scholes_put(S, K, r, sigma, T - h, q)

    return (price_down - price_up) / (2 * h)

def rho_call_numerical(S, K, r, sigma, T, q=0, h=0.0001):
    """
    Rho of a European call option, computed via central finite
    difference (bump-and-revalue) instead of the analytic formula.
    Should closely match rho_call() — used to validate the analytic
    result.

    Note: same as the analytic rho_call(), this returns the change in
    price per 1.00 (100%) change in r. Divide by 100 for the
    per-1%-move convention.

    h : the bump size for r. Kept small since realistic interest rates
        are themselves small decimals (e.g. 0.05 for 5%).
    """
    price_up = black_scholes_call(S, K, r + h, sigma, T, q)
    price_down = black_scholes_call(S, K, r - h, sigma, T, q)

    return (price_up - price_down) / (2 * h)

def rho_put_numerical(S, K, r, sigma, T, q=0, h=0.0001):
    """
    Rho of a European put option, computed via central finite
    difference (bump-and-revalue) instead of the analytic formula.
    Should closely match rho_call() — used to validate the analytic
    result.

    Note: same as the analytic rho_put(), this returns the change in
    price per 1.00 (100%) change in r. Divide by 100 for the
    per-1%-move convention.

    h : the bump size for r. Kept small since realistic interest rates
        are themselves small decimals (e.g. 0.05 for 5%).
    """
    price_up = black_scholes_put(S, K, r + h, sigma, T, q)
    price_down = black_scholes_put(S, K, r - h, sigma, T, q)

    return (price_up - price_down) / (2 * h)
    
    