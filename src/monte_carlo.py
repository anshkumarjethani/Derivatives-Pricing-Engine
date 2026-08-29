import numpy as np
import pandas as pd
import scipy.stats as norm

def monte_carlo_call(S, K, r, sigma, T, q=0, n_simulations=100000, seed=None):
    """
    Price a European call option using Monte Carlo simulation.

    Parameters:
    S              : current stock price
    K              : strike price
    r              : risk-free interest rate (annualized, decimal)
    sigma          : volatility (annualized, decimal)
    T              : time to expiry, in years
    q              : continuous dividend yield (annualized, decimal). Defaults to 0.
    n_simulations  : number of random price paths to simulate
    seed           : random seed for reproducibility (optional)

    Returns:
    price : estimated option price
    """
    if seed is not None:
        np.random.seed(seed)

    Z = np.random.standard_normal(n_simulations)

    S_T = S * np.exp((r - q - sigma**2 / 2) * T + sigma * np.sqrt(T) * Z)

    payoffs = np.maximum(S_T - K, 0)

    price = np.exp(-r * T) * np.mean(payoffs)

    return price

def monte_carlo_call_antithetic(S, K, r, sigma, T, q=0, n_simulations=100000, seed=None):
    """
    Price a European call option using Monte Carlo simulation with
    antithetic variates for variance reduction.

    For every random draw Z, its mirror -Z is also used — pairing each
    high-outlier path with a corresponding low-outlier path, which
    reduces sampling noise without requiring more independent random draws.

    Parameters: same as monte_carlo_call.
    n_simulations : total number of simulated paths (half will be
                    independently drawn, half will be their negatives).

    Returns:
    price : estimated option price
    """
    if seed is not None:
        np.random.seed(seed)

    half_n = n_simulations // 2

    Z_half = np.random.standard_normal(half_n)
    Z = np.concatenate([Z_half, -Z_half])

    S_T = S * np.exp((r - q - sigma**2 / 2) * T + sigma * np.sqrt(T) * Z)

    payoffs = np.maximum(S_T - K, 0)

    price = np.exp(-r * T) * np.mean(payoffs)

    return price

def monte_carlo_call_control_variate(S, K, r, sigma, T, q=0, n_simulations=100000, seed=None):
    """
    Price a European call option using Monte Carlo simulation with a
    control variate (the simulated final stock price, whose true
    expectation is known exactly) for variance reduction.

    Parameters: same as monte_carlo_call.

    Returns:
    price : estimated option price
    """
    if seed is not None:
        np.random.seed(seed)

    Z = np.random.standard_normal(n_simulations)

    S_T = S * np.exp((r - q - sigma**2 / 2) * T + sigma * np.sqrt(T) * Z)

    payoffs = np.maximum(S_T - K, 0)

    true_stock_expectation = S * np.exp((r - q) * T)

    covariance = np.cov(payoffs, S_T)[0, 1]
    variance_S_T = np.var(S_T)
    b = covariance / variance_S_T

    adjusted_payoffs = payoffs - b * (S_T - true_stock_expectation)

    price = np.exp(-r * T) * np.mean(adjusted_payoffs)

    return price

def monte_carlo_put(S, K, r, sigma, T, q=0, n_simulations=100000, seed=None):
    """
    Price a European put option using Monte Carlo simulation.

    Parameters: same as monte_carlo_call.

    Returns:
    price : estimated option price
    """
    if seed is not None:
        np.random.seed(seed)

    Z = np.random.standard_normal(n_simulations)

    S_T = S * np.exp((r - q - sigma**2 / 2) * T + sigma * np.sqrt(T) * Z)

    payoffs = np.maximum(K - S_T, 0)

    price = np.exp(-r * T) * np.mean(payoffs)

    return price


def monte_carlo_put_antithetic(S, K, r, sigma, T, q=0, n_simulations=100000, seed=None):
    """
    Price a European put option using Monte Carlo simulation with
    antithetic variates for variance reduction.

    Parameters: same as monte_carlo_call_antithetic.

    Returns:
    price : estimated option price
    """
    if seed is not None:
        np.random.seed(seed)

    half_n = n_simulations // 2

    Z_half = np.random.standard_normal(half_n)
    Z = np.concatenate([Z_half, -Z_half])

    S_T = S * np.exp((r - q - sigma**2 / 2) * T + sigma * np.sqrt(T) * Z)

    payoffs = np.maximum(K - S_T, 0)

    price = np.exp(-r * T) * np.mean(payoffs)

    return price


def monte_carlo_put_control_variate(S, K, r, sigma, T, q=0, n_simulations=100000, seed=None):
    """
    Price a European put option using Monte Carlo simulation with a
    control variate (the simulated final stock price) for variance
    reduction.

    Parameters: same as monte_carlo_call_control_variate.

    Returns:
    price : estimated option price
    """
    if seed is not None:
        np.random.seed(seed)

    Z = np.random.standard_normal(n_simulations)

    S_T = S * np.exp((r - q - sigma**2 / 2) * T + sigma * np.sqrt(T) * Z)

    payoffs = np.maximum(K - S_T, 0)

    true_stock_expectation = S * np.exp((r - q) * T)

    covariance = np.cov(payoffs, S_T)[0, 1]
    variance_S_T = np.var(S_T)
    b = covariance / variance_S_T

    adjusted_payoffs = payoffs - b * (S_T - true_stock_expectation)

    price = np.exp(-r * T) * np.mean(adjusted_payoffs)

    return price
