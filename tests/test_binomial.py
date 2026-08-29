import numpy as np
import pandas as pd
import scipy.stats as norm 

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from binomial import binomial_price
from black_scholes import black_scholes_call, black_scholes_put

def test_european_call_converges_to_black_scholes():
    """As N grows, European binomial call should approach the Black-Scholes price."""
    bs_price = black_scholes_call(S=100, K=100, r=0.05, sigma=0.2, T=1)
    binomial_500 = binomial_price(S=100, K=100, r=0.05, sigma=0.2, T=1, N=500,
                                   q=0, option_type='call', american=False)

    assert abs(bs_price - binomial_500) < 0.05

def test_european_put_converges_to_black_scholes():
    """As N grows, European binomial put should approach the Black-Scholes price."""
    bs_price = black_scholes_put(S=100, K=100, r=0.05, sigma=0.2, T=1)
    binomial_500 = binomial_price(S=100, K=100, r=0.05, sigma=0.2, T=1, N=500,
                                   q=0, option_type='put', american=False)

    assert abs(bs_price - binomial_500) < 0.05

def test_convergence_improves_with_more_steps():
    """A finer tree (more steps) should be closer to Black-Scholes than a coarse one."""
    bs_price = black_scholes_call(S=100, K=100, r=0.05, sigma=0.2, T=1)

    coarse = binomial_price(S=100, K=100, r=0.05, sigma=0.2, T=1, N=10,
                             q=0, option_type='call', american=False)
    fine = binomial_price(S=100, K=100, r=0.05, sigma=0.2, T=1, N=500,
                           q=0, option_type='call', american=False)

    error_coarse = abs(bs_price - coarse)
    error_fine = abs(bs_price - fine)

    assert error_fine < error_coarse

def test_american_put_at_least_as_valuable_as_european():
    """American exercise right should never make an option less valuable."""
    american = binomial_price(S=80, K=100, r=0.05, sigma=0.2, T=1, N=500,
                               q=0, option_type='put', american=True)
    european = binomial_price(S=80, K=100, r=0.05, sigma=0.2, T=1, N=500,
                               q=0, option_type='put', american=False)

    assert american >= european

def test_deep_itm_american_put_equals_intrinsic_value():
    """A deep ITM American put with early exercise should price at (or very
    near) its immediate exercise value, K - S."""
    price = binomial_price(S=80, K=100, r=0.05, sigma=0.2, T=1, N=500,
                            q=0, option_type='put', american=True)

    assert abs(price - 20.0) < 0.5                

def test_american_call_no_dividend_equals_european():
    """With no dividends, early exercise of a call is never optimal, so
    American and European call prices should be (almost) identical."""
    american = binomial_price(S=100, K=100, r=0.05, sigma=0.2, T=1, N=500,
                               q=0, option_type='call', american=True)
    european = binomial_price(S=100, K=100, r=0.05, sigma=0.2, T=1, N=500,
                               q=0, option_type='call', american=False)

    assert abs(american - european) < 0.01


def test_invalid_option_type_raises_error():
    """Passing an invalid option_type should raise a ValueError, not
    silently return a wrong number."""
    try:
        binomial_price(S=100, K=100, r=0.05, sigma=0.2, T=1, N=10,
                        q=0, option_type='banana', american=False)
        assert False, "Expected a ValueError but none was raised"
    except ValueError:
        pass
    