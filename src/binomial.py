import pandas as pd
import numpy as np 
from scipy.stats import norm

def binomial_price(S, K, r, sigma, T, N, q=0, option_type='call', american=False):
    """
    Price a European or American option using the Cox-Ross-Rubinstein
    binomial tree model.

    Parameters:
    S            : current stock price
    K            : strike price
    r            : risk-free interest rate (annualized, decimal)
    sigma        : volatility (annualized, decimal)
    T            : time to expiry, in years
    N            : number of time steps in the tree
    q            : continuous dividend yield (annualized, decimal). Defaults to 0.
    option_type  : 'call' or 'put'
    american     : True for American (early exercise allowed), False for European

    Returns:
    price : theoretical option price
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    # Step 1: stock prices at the final step (expiry)
    stock_prices = np.zeros(N + 1)
    for i in range(N + 1):
        stock_prices[i] = S * (u ** (N - i)) * (d ** i)

    # Step 2: option payoff at expiry, for each final stock price
    if option_type == 'call':
        option_values = np.maximum(stock_prices - K, 0)
    elif option_type == 'put':
        option_values = np.maximum(K - stock_prices, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Step 3: work backward through the tree, one step at a time
    for step in range(N - 1, -1, -1):
        stock_prices = S * (u ** np.arange(step, -1, -1)) * (d ** np.arange(0, step + 1))

        continuation_value = discount * (p * option_values[:step + 1] + (1 - p) * option_values[1:step + 2])

        if american:
            if option_type == 'call':
                exercise_value = np.maximum(stock_prices - K, 0)
            else:
                exercise_value = np.maximum(K - stock_prices, 0)
            option_values = np.maximum(continuation_value, exercise_value)
        else:
            option_values = continuation_value

    return option_values[0]