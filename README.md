# Derivatives Pricing Engine

A quantitative options pricing engine implementing three independent pricing methodologies — **Black-Scholes-Merton (analytic)**, **Cox-Ross-Rubinstein binomial trees**, and **Monte Carlo simulation** — that cross-validate against each other, alongside a full analytic and numerical Greeks suite, all validated against both synthetic test cases and live market data.

This project is built the way a validation/risk quant would approach pricing infrastructure: not just implementing a formula, but proving it's correct through convergence testing, parity checks, independent cross-method validation, and honest reconciliation against real, tradeable market prices.

---

## Why this project exists

Every options desk needs a fast, trustworthy pricer to quote two-sided markets and hedge inventory in real time. A single implementation of Black-Scholes is a common portfolio project — reconciling **three independent pricing methods**, each with its own convergence and validation checks, and then checking all three against **live market data**, is the kind of infrastructure discipline that a real desk (and a model validation function) actually requires.

---

## Project scope

| Stage | Component | Status |
|-------|-----------|--------|
| 1 | Black-Scholes-Merton closed-form pricer (calls & puts, with continuous dividend yield) | ✅ Complete |
| 1 | Analytic Greeks — Delta, Gamma, Vega, Theta, Rho | ✅ Complete |
| 1 | Finite-difference (bump-and-revalue) Greeks, cross-checked against analytic values, including all parity relationships (put-call, Delta, Theta, Rho) | ✅ Complete |
| 2 | Cox-Ross-Rubinstein binomial tree (American early-exercise support) | ✅ Complete |
| 3 | Monte Carlo simulator, plain + antithetic + control variates, calls and puts | ✅ Complete |
| 4 | Real market data integration (live spot prices, option chains, liquidity filtering) | ✅ Complete |
| 4 | Cross-method validation against live market data | ✅ Complete |
| 5 | Real-time Greeks dashboard | ⏳ Planned |

**33 automated tests passing** across pricing accuracy, Greeks, parity relationships, convergence behavior, variance-reduction techniques, and live real-market cross-validation.

---

## Methodology

### 1. Black-Scholes-Merton (analytic)
Closed-form European option pricing extended for continuous dividend yield (Merton, 1973):

```
C = S·e^(−qT)·N(d1) − K·e^(−rT)·N(d2)
P = K·e^(−rT)·N(−d2) − S·e^(−qT)·N(−d1)

d1 = [ln(S/K) + (r − q + σ²/2)·T] / (σ·√T)
d2 = d1 − σ·√T
```

Validated against **put-call parity** (`C − P = S·e^(−qT) − K·e^(−rT)`), which must hold exactly for any correct implementation, independent of the specific pricing formula.

### 2. Binomial tree (Cox-Ross-Rubinstein)
Discrete-time lattice model supporting American-style early exercise, which the closed-form Black-Scholes model cannot price. Validated two ways:
- **Convergence**: European binomial price → Black-Scholes price as steps → ∞
- **Economic correctness**: American price ≥ European price always; a deep in-the-money American put correctly prices at intrinsic value when immediate exercise dominates holding

### 3. Monte Carlo simulation
Simulates thousands of risk-neutral stock price paths to estimate option value. Implemented three ways:
- **Plain** simulation
- **Antithetic variates** — pairing each random draw with its negation, reducing sampling noise at no extra computational cost
- **Control variates** — using the simulated stock price (whose true expectation is known exactly) to correct the option price estimate, based on their covariance

Variance reduction is empirically confirmed, not just asserted: repeated-run spread is measurably tighter for both antithetic and control-variate methods versus plain Monte Carlo at equal simulation count.

### Greeks — two independent methods
Every Greek (Delta, Gamma, Vega, Theta, Rho) is computed two ways:
- **Analytically** — exact closed-form derivatives of the pricing formula
- **Numerically** — finite-difference bump-and-revalue (nudge one input, reprice, measure the change)

The two are cross-validated to agree closely, and every put-call parity relationship (price, Delta, Theta, Rho) is independently derived and tested.

### 4. Real market data validation
Live spot prices, option chains, and risk-free rates (13-week Treasury yield, `^IRX`) are pulled via `yfinance`. Raw chains are filtered for liquidity (minimum volume and open interest) to exclude stale, untraded quotes — which otherwise show implausible implied volatility from illiquid pricing. A near-the-money, liquid contract is priced with all three methods using the real spot price, real time-to-expiry, the underlying's actual dividend yield, and a real live risk-free rate — no placeholder values.

**Finding 1 — cross-method agreement**: all three pricing methods agree tightly with each other on real market data (within a few cents), confirming the cross-method validation holds beyond synthetic test cases.

**Finding 2 — market-price reconciliation**: with a real Treasury-derived risk-free rate in place of an earlier round-number placeholder, model price landed within ~2 cents of the real market ask on a representative AAPL contract — closing a gap that had been roughly 70 cents wide under the placeholder rate. This confirmed the risk-free rate assumption, not the dividend yield, was the dominant driver of the earlier discrepancy.

**Finding 3 — implied vs. historical volatility**: using the market's own implied volatility to price an option and comparing to the market price is partially circular, since implied vol is itself derived from market prices. As an independent check, annualized historical (realized) volatility was computed directly from recent price history (log returns, no options data involved) and used as an alternative input. On the same AAPL contract, historical volatility (32.7%) was meaningfully higher than market-implied volatility (26.1%), and pricing with historical vol produced a theoretical price roughly ₹2.24 above the market price — versus ~2 cents using implied vol. This divergence is an honest, expected finding: implied volatility reflects the market's forward-looking expectation, while historical volatility reflects only what has already happened; a real gap between them is a genuine market signal (related to the volatility risk premium), not a modeling error.

---

## Tech stack

- **Python** — numpy, scipy, pandas, matplotlib, pytest, yfinance
- **Git / GitHub** — version control, staged commit history by development stage
- **VS Code** — Jupyter notebooks for exploration, `.py` modules for production code

---

## Project structure

```
derivatives-pricing-engine/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/             # exploratory analysis, formula testing, data exploration
├── src/
│   ├── black_scholes.py   # closed-form pricer + analytic & numerical Greeks
│   ├── binomial.py        # CRR binomial tree, American/European
│   ├── monte_carlo.py     # MC pricer — plain, antithetic, control variate
│   └── market_data.py     # live spot price & option chain fetching, liquidity filtering
├── tests/
│   ├── test_black_scholes.py
│   ├── test_binomial.py
│   ├── test_monte_carlo.py
│   └── test_real_data_validation.py
├── docs/
├── requirements.txt
└── README.md
```

---

## Inputs & outputs

**Inputs:** option chain (strike, expiry, bid/ask), spot price, dividend schedule, risk-free/discount curve, volatility (market-implied or historical)

**Outputs:** cross-validated theoretical prices, full Greeks (analytic + numerical), convergence analysis across pricing methods, real-market validation report

---

## Getting started

```bash
git clone https://github.com/<your-username>/derivatives-pricing-engine.git
cd derivatives-pricing-engine
pip install -r requirements.txt
pytest tests/ -v
```

```python
from src.black_scholes import black_scholes_call, delta_call, gamma, vega, theta_call, rho_call

price = black_scholes_call(S=100, K=100, r=0.05, sigma=0.2, T=1, q=0.0)
```

---

## Roadmap

Remaining work: a real-time Greeks dashboard consuming live option chain data (planned as a clean tabular output, not an interactive web app, to stay consistent with the rest of the project's scope). The two earlier-flagged validation gaps — a placeholder risk-free rate, and reliance solely on market-implied volatility — have both been resolved: the engine now uses a real live Treasury-derived rate and includes an independent historical-volatility check alongside implied volatility (see Methodology, section 4).