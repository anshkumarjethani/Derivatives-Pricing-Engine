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
| 5 | Real-time Greeks dashboard (live table: price + all 5 Greeks per contract) | ✅ Complete |

**35 automated tests passing** across pricing accuracy, Greeks, parity relationships, convergence behavior, variance-reduction techniques, and live real-market cross-validation.

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
Live spot prices, option chains, and risk-free rates (13-week Treasury yield, `^IRX`) are pulled via `yfinance`. Raw chains are filtered for liquidity — requiring real volume or open interest, a valid nonzero implied volatility, and a genuine nonzero bid — to exclude stale or broken quotes, which otherwise produce implausible prices and Greeks downstream. Contract selection also enforces a moneyness constraint and retries across successive expiries, since individual expiries can have thin or unreliable data even when others are healthy.

The validation underlying is the **S&P 500 Index (`^SPX`)** — real index options, priced as European (matching their actual exchange convention), after initial testing on individual equity options (AAPL) surfaced several real data-quality issues not present in SPX's much deeper, more reliable market.

**Finding 1 — cross-method agreement**: all three pricing methods agree tightly with each other on real market data (within a fraction of a percent), confirming the cross-method validation holds beyond synthetic test cases.

**Finding 2 — market-price reconciliation**: with a real Treasury-derived risk-free rate in place of an earlier round-number placeholder, model prices land within a small margin of real market bid/ask on representative contracts.

**Finding 3 — implied vs. historical volatility**: using the market's own implied volatility to price an option and comparing to the market price is partially circular, since implied vol is itself derived from market prices. As an independent check, annualized historical (realized) volatility was computed directly from recent price history (log returns, no options data involved). The two estimates diverged meaningfully on the tested contract, an honest and expected finding — implied volatility reflects the market's forward-looking expectation, historical volatility reflects only what has already happened, and a real gap between them is a genuine market signal, not a modeling error.

### 5. Real-time Greeks dashboard
`build_greeks_dashboard()` pulls a live, liquid option chain for a given underlying and expiry, and computes theoretical price plus all five Greeks (Delta, Gamma, Vega, Theta, Rho) for every qualifying contract, alongside the real market bid/ask for direct comparison. Output is a clean, sorted table — deliberately built as a script/notebook-producible table rather than an interactive web app, consistent with the project's Python-only scope. Results are saved to `data/processed/` as CSV snapshots.

On live SPX data, the dashboard reproduces the expected theoretical patterns exactly: Delta decreases monotonically from deep in-the-money toward out-of-the-money, and Gamma peaks near the at-the-money strikes — confirming the full pricing and Greeks stack holds up correctly on genuine, live market data, not just the synthetic test cases used during development.

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

## Project status

All planned stages are complete: three cross-validated pricing methods, full analytic and numerical Greeks with parity checks, live market data integration with robust liquidity/data-quality filtering, real-rate and independent-volatility validation, and a live Greeks dashboard — all backed by 35 passing automated tests spanning synthetic and real-data cases.

Possible future extensions (not required for the current scope): a broker-connected live data feed (e.g. for NSE-specific instruments, via Kite Connect or Groww's API) as a separate integration; path-dependent option support via the existing Monte Carlo engine; an interactive web dashboard as an alternative front-end to the current tabular output.
