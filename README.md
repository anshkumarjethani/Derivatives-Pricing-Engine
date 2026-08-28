# Derivatives Pricing Engine

A quantitative options pricing engine implementing three independent pricing methodologies — **Black-Scholes-Merton (analytic)**, **Cox-Ross-Rubinstein binomial trees**, and **Monte Carlo simulation** — that cross-validate against each other, alongside a full analytic and numerical Greeks suite.

This project is built the way a validation/risk quant would approach pricing infrastructure: not just implementing a formula, but proving it's correct through convergence testing, parity checks, and independent cross-method validation.

---

## Why this project exists

Every options desk needs a fast, trustworthy pricer to quote two-sided markets and hedge inventory in real time. A single implementation of Black-Scholes is a common portfolio project — reconciling **three independent pricing methods**, each with its own convergence and validation checks, is the kind of infrastructure discipline that a real desk (and a model validation function) actually requires.

---

## Project scope

| Stage | Component | Status |
|-------|-----------|--------|
| 1 | Black-Scholes-Merton closed-form pricer (calls & puts, with continuous dividend yield) | ✅ Complete |
| 1 | Analytic Greeks — Delta, Gamma, Vega, Theta, Rho | ✅ Complete |
| 1 | Finite-difference (bump-and-revalue) Greeks, cross-checked against analytic values | 🔄 In progress |
| 2 | Cox-Ross-Rubinstein binomial tree (American early-exercise support) | ⏳ Planned |
| 3 | Monte Carlo simulator with antithetic and control variates | ⏳ Planned |
| 4 | Cross-method convergence analysis & validation report | ⏳ Planned |

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
Discrete-time lattice model supporting American-style early exercise, which the closed-form Black-Scholes model cannot price. Validated by checking convergence to the Black-Scholes price as the number of time steps → ∞.

### 3. Monte Carlo simulation
Simulates thousands of risk-neutral stock price paths to estimate option value, using **antithetic variates** and **control variates** for variance reduction. Validated by checking convergence to the Black-Scholes price as the number of simulated paths → ∞.

### Greeks — two independent methods
Every Greek (Delta, Gamma, Vega, Theta, Rho) is computed two ways:
- **Analytically** — exact closed-form derivatives of the pricing formula
- **Numerically** — finite-difference bump-and-revalue (nudge one input, reprice, measure the change)

The two must agree closely. This numerical approach is also what makes Greeks computable for the binomial and Monte Carlo methods later, where no closed-form derivative exists.

---

## Tech stack

- **Python** — numpy, scipy (statistical functions), pandas, matplotlib, pytest
- **Git / GitHub** — version control, staged commit history by development stage
- **VS Code** — Jupyter notebooks for exploration, `.py` modules for production code

---

## Project structure

```
derivatives-pricing-engine/
├── data/
│   ├── raw/              # unprocessed market data
│   └── processed/        # cleaned data ready for use
├── notebooks/            # exploratory analysis, formula testing, validation checks
├── src/                  # production pricing modules
│   └── black_scholes.py  # closed-form pricer + analytic Greeks
├── tests/                # pytest unit tests — known-value checks, parity checks
├── docs/                 # supporting documentation
├── requirements.txt
└── README.md
```

---

## Inputs & outputs

**Inputs:** option chain (strike, expiry, bid/ask), spot price, dividend schedule, risk-free/discount curve, volatility

**Outputs:** cross-validated theoretical prices, full Greeks (analytic + numerical), convergence analysis across pricing methods

---

## Getting started

```bash
git clone https://github.com/<your-username>/derivatives-pricing-engine.git
cd derivatives-pricing-engine
pip install -r requirements.txt
```

```python
from src.black_scholes import black_scholes_call, delta_call, gamma, vega, theta_call, rho_call

price = black_scholes_call(S=100, K=100, r=0.05, sigma=0.2, T=1, q=0.0)
```

---

## Roadmap

This is an actively developed, staged project — later stages (binomial tree, Monte Carlo, full cross-validation suite) will build on the Black-Scholes foundation above rather than replace it. See the Project Scope table for current status.

