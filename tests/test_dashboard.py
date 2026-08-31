import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dashboard import build_greeks_dashboard


def test_dashboard_produces_sane_greeks_pattern():
    """
    Confirms the dashboard, built from live SPX data, reproduces the
    expected theoretical patterns: Delta should decrease monotonically
    as strike increases (for calls), and every row should have a
    positive model price.
    """
    S, results = build_greeks_dashboard("^SPX", option_side='calls', q=0.0105)

    assert len(results) > 0, "Dashboard returned no rows"
    assert (results['model_price'] > 0).all(), "Found a non-positive model price"

    sorted_by_strike = results.sort_values('strike')
    deltas = sorted_by_strike['delta'].values

    is_decreasing = all(deltas[i] >= deltas[i + 1] for i in range(len(deltas) - 1))
    assert is_decreasing, "Delta should decrease as strike increases for calls"