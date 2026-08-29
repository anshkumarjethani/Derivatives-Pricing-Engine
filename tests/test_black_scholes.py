import pandas as pd
import numpy as np 
from scipy.stats import norm
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from black_scholes import (
    black_scholes_call,
    black_scholes_put,
    delta_call,
    delta_call_numerical,
    delta_put,
    delta_put_numerical,
    gamma,
    gamma_numerical,
    vega,
    vega_numerical,
    theta_call,
    theta_call_numerical,
    theta_put,
    theta_put_numerical,
    rho_call,
    rho_call_numerical,
    rho_put,
    rho_put_numerical
)
# ---- Pricing ----

def test_call_price_known_value():
    price = black_scholes_call(S=100, K=100, r=0.05, sigma=0.2, T=1)
    assert abs(price - 10.45) < 0.01


def test_put_price_known_value():
    price = black_scholes_put(S=100, K=100, r=0.05, sigma=0.2, T=1)
    assert abs(price - 5.57) < 0.01


def test_put_call_parity():
    S, K, r, sigma, T = 100, 100, 0.05, 0.2, 1
    call = black_scholes_call(S, K, r, sigma, T)
    put = black_scholes_put(S, K, r, sigma, T)

    left_side = call - put
    right_side = S - K * np.exp(-r * T)

    assert abs(left_side - right_side) < 1e-8


# ---- Delta ----

def test_delta_call_matches_numerical():
    analytic = delta_call(S=100, K=100, r=0.05, sigma=0.2, T=1)
    numerical = delta_call_numerical(S=100, K=100, r=0.05, sigma=0.2, T=1)
    assert abs(analytic - numerical) < 0.001


def test_delta_put_matches_numerical():
    analytic = delta_put(S=100, K=100, r=0.05, sigma=0.2, T=1)
    numerical = delta_put_numerical(S=100, K=100, r=0.05, sigma=0.2, T=1)
    assert abs(analytic - numerical) < 0.001


def test_delta_parity():
    """delta_call - delta_put should equal e^(-qT); with q=0, that's 1."""
    d_call = delta_call(S=100, K=100, r=0.05, sigma=0.2, T=1)
    d_put = delta_put(S=100, K=100, r=0.05, sigma=0.2, T=1)
    assert abs((d_call - d_put) - 1.0) < 1e-8


# ---- Gamma (shared by calls and puts) ----

def test_gamma_matches_numerical():
    analytic = gamma(S=100, K=100, r=0.05, sigma=0.2, T=1)
    numerical = gamma_numerical(S=100, K=100, r=0.05, sigma=0.2, T=1)
    assert abs(analytic - numerical) < 0.001


# ---- Vega (shared by calls and puts) ----

def test_vega_matches_numerical():
    analytic = vega(S=100, K=100, r=0.05, sigma=0.2, T=1)
    numerical = vega_numerical(S=100, K=100, r=0.05, sigma=0.2, T=1)
    assert abs(analytic - numerical) < 0.01


# ---- Theta ----

def test_theta_call_matches_numerical():
    analytic = theta_call(S=100, K=100, r=0.05, sigma=0.2, T=1)
    numerical = theta_call_numerical(S=100, K=100, r=0.05, sigma=0.2, T=1)
    assert abs(analytic - numerical) < 0.01


def test_theta_put_matches_numerical():
    analytic = theta_put(S=100, K=100, r=0.05, sigma=0.2, T=1)
    numerical = theta_put_numerical(S=100, K=100, r=0.05, sigma=0.2, T=1)
    assert abs(analytic - numerical) < 0.01


def test_theta_parity():
    """theta_call - theta_put should equal q*S*e^(-qT) - r*K*e^(-rT)"""
    S, K, r, sigma, T, q = 100, 100, 0.05, 0.2, 1, 0

    t_call = theta_call(S, K, r, sigma, T)
    t_put = theta_put(S, K, r, sigma, T)

    left_side = t_call - t_put
    right_side = q * S * np.exp(-q * T) - r * K * np.exp(-r * T)

    assert abs(left_side - right_side) < 1e-8   


# ---- Rho ----

def test_rho_call_matches_numerical():
    analytic = rho_call(S=100, K=100, r=0.05, sigma=0.2, T=1)
    numerical = rho_call_numerical(S=100, K=100, r=0.05, sigma=0.2, T=1)
    assert abs(analytic - numerical) < 0.01


def test_rho_put_matches_numerical():
    analytic = rho_put(S=100, K=100, r=0.05, sigma=0.2, T=1)
    numerical = rho_put_numerical(S=100, K=100, r=0.05, sigma=0.2, T=1)
    assert abs(analytic - numerical) < 0.01


def test_rho_parity():
    """rho_call - rho_put should equal K*T*e^(-rT)"""
    S, K, r, sigma, T = 100, 100, 0.05, 0.2, 1

    r_call = rho_call(S, K, r, sigma, T)
    r_put = rho_put(S, K, r, sigma, T)

    left_side = r_call - r_put
    right_side = K * T * np.exp(-r * T)

    assert abs(left_side - right_side) < 1e-8