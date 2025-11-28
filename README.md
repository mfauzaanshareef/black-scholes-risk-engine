# Black Scholes Risk & Pricing Engine
A Python-based quantitative finance toolkit designed to price European options, calculate Implied Volatility (IV), and compute Black-Scholes Greeks. The project aims to simulate portfolio risk, validate prices via Monte Carlo methods, and analyze Delta-hedging strategies.
## 📂 Project Structure

* **`bs_pricer.py`**: Core Black-Scholes-Merton pricing logic for Calls and Puts (handling continuous dividend yields). Implements `d1` and `d2` calculations using standard libraries.
* **`iv_solver.py`**: A robust Implied Volatility solver. It uses a hybrid approach (Bisection method refined by Newton-Raphson) to back out volatility from market prices without external optimization libraries.
* **`greeks.py`**: Analytical formulas for option sensitivities: Delta ($\Delta$), Gamma ($\Gamma$), Vega ($\nu$), Theta ($\Theta$), and Rho ($\rho$).
* **`generate_options_chain.py`**: A script that fetches real-time spot prices (via `yfinance`), generates synthetic option chains with specific moneyness grids, and calculates theoretical prices/IVs.
* **`stocks.py`**: A configuration file containing the ticker universe grouped by GICS sectors.
* **`utils.py`**: Helper functions for date handling (year fractions, business days) and numerical bounds.

## 🚀 Configuration & Data Generation

The `generate_options_chain.py` script is pre-configured to generate a synthetic testing dataset. It fetches the underlying stock prices for a defined universe of tickers and constructs mock option chains based on the following parameters, which can be adjusted by the user:

- **Simulation Start Date**: The valuation date is set to **March 13, 2025** (`VAL_DATE`). This future date is chosen to ensure a 6-month historical window is available for simulation purposes.
- **Moneyness Grid**: The script generates strikes ranging from **0.80 to 1.20** (80% to 120%) of the spot price to simulate deep ITM and OTM options.
- **Maturities**: Only two specific expiration windows are generated: **3 months** (91 days) and **6 months** (182 days).
- **Risk-Free Rate**: The yields are hardcoded to recent US Treasury Bill rates (approx. **3.86%** for 3M and **3.72%** for 6M).
- **Tickers**: A predefined list of tickers (e.g., AAPL, MSFT, NVDA) is included in the `RAW_TICKERS` list.

*Note: All of these parameters (dates, yields, grid size, and tickers) are defined at the top of `generate_options_chain.py` and can be modified to fit different market conditions or testing scenarios.*
## 🛠️ Installation & Requirements

The core pricing logic requires no external dependencies. However, to run the data generation scripts (`generate_options_chain.py`), you will need the following:

```bash
pip install pandas numpy yfinance
```

## 💻 Usage
1. Generating Option Chains
To generate a CSV file containing synthetic option chains (Calls and Puts) for the defined stock universe:

```bash
python generate_options_chain.py
```

*Output: A CSV file (e.g., `synthetic_options_2025-03-13.csv`) containing strikes, expirations, spot prices, IVs, and calculated Option Prices.*

2. Using the Pricer as a Library

You can use the modules directly in your own scripts.

Example 1: Call Option & Delta. This snippet prices a European Call and calculates its Delta (sensitivity to the underlying price).

```python
from bs_pricer import call_price
from greeks import delta_call

S = 100.0   # Spot
K = 100.0   # Strike
T = 1.0     # Time to maturity (years)
r = 0.05    # Risk-free rate
q = 0.01    # Dividend yield
sigma = 0.2 # Volatility (20%)

price = call_price(S, K, T, r, q, sigma)
delta = delta_call(S, K, T, r, q, sigma)

print(f"Call Price: ${price:.2f}")
print(f"Call Delta: {delta:.4f}")
```

Example 2: Put Option, Delta & Vega. This snippet prices a European Put and calculates two Greeks:

- Delta: How much the put price changes for a $1 change in the stock price.

- Vega: How much the put price changes for a 1% change in volatility (note: returns value per 100% vol, so typically scaled).

```python
from bs_pricer import put_price
from greeks import delta_put, vega

# Using the same inputs as above
price_put = put_price(S, K, T, r, q, sigma)
d_put = delta_put(S, K, T, r, q, sigma)
v_put = vega(S, K, T, r, q, sigma)

print(f"Put Price: ${price_put:.2f}")
print(f"Put Delta: {d_put:.4f}")
print(f"Put Vega: {v_put:.4f}")
```

Explanation of Examples

- Inputs: Both examples use standard Black-Scholes inputs: Spot ($S$), Strike ($K$), Time in years ($T$), Risk-free rate ($r$), Dividend yield ($q$), and Volatility ($\sigma$). 
- Functions: call_price and put_price return the theoretical premium in dollars. delta_call/put and vega return the raw Greek values derived from the analytical formulas in greeks.py.
- After generating your own synthetic option chain using `generate_options_chain.py`, you can apply these functions directly to the data.


## 📝 Future Work / TODO

The following features are currently planned but **not yet implemented**:

- [ ]  **Implement Delta Hedging**: Create a simulation engine to keep the portfolio delta-neutral at a daily granularity by buying/selling the underlying stock.
- [ ]  **Visualization**: Plot Implied Volatility smiles and term structures based on the generated data.
- [ ]  **P&L Analysis**: Compute and compare returns for hedged vs. unhedged portfolios.
- [ ]  **Monte Carlo Validation**: Carry out Monte Carlo simulations to cross-validate the analytical Black-Scholes prices.

