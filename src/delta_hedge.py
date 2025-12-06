import pandas as pd
import numpy as np
import datetime as dt
import math

# Import local modules
from bs_pricer import call_price, put_price
from greeks import delta_call, delta_put
from utils import yearfrac

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
DATA_FILE = "synthetic_options_2025-03-13.csv"
TRADING_DAYS_PER_YEAR = 252
SIMULATION_STEPS_PER_DAY = 1  # Rebalance once daily


# ------------------------------------------------------------------------------
# Helper: Geometric Brownian Motion Generator
# ------------------------------------------------------------------------------
def generate_price_path(S0, mu, sigma, days, seed=None):
    """
    Generates a synthetic price path using Geometric Brownian Motion.
    In a real scenario, replace this with historical data fetching.
    """
    if seed is not None:
        np.random.seed(seed)

    dt_step = 1.0 / TRADING_DAYS_PER_YEAR
    prices = [S0]
    current_price = S0

    for _ in range(days):
        # GBM: S_t = S_{t-1} * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        drift = (mu - 0.5 * sigma ** 2) * dt_step
        shock = sigma * math.sqrt(dt_step) * np.random.normal()
        current_price *= math.exp(drift + shock)
        prices.append(current_price)

    return prices


# ------------------------------------------------------------------------------
# Delta Hedging Engine
# ------------------------------------------------------------------------------
class DeltaHedgingStrategy:
    def __init__(self, ticker, portfolio_df, start_date):
        self.ticker = ticker
        self.portfolio = portfolio_df.copy()
        self.start_date = start_date

        # Portfolio State
        self.cash = 0.0
        self.stock_shares = 0.0
        self.daily_log = []

        # Market Parameters (extracted from the first row of portfolio)
        self.S0 = self.portfolio['underlying_price'].iloc[0]
        self.r = self.portfolio['r_cont'].iloc[0]
        self.q = self.portfolio['q'].iloc[0]

        # Determine Simulation Horizon (until last expiration)
        self.portfolio['expiration_date'] = pd.to_datetime(self.portfolio['expiration']).dt.date
        last_expiry = self.portfolio['expiration_date'].max()
        self.total_days = (last_expiry - self.start_date).days

        # Initial Setup: Pay for options (Negative Cash)
        # We assume we BUY the portfolio (Long positions)
        initial_cost = (self.portfolio['option_price'] * self.portfolio['contractSize']).sum()
        self.cash -= initial_cost
        print(f"\n[{ticker}] Initial Portfolio Cost: ${initial_cost:,.2f}")

    def run_simulation(self):
        # 1. Generate "Market" Data (Price Path)
        # We use the average IV of the portfolio as the 'realized' volatility for the stock
        realized_vol = self.portfolio['iv'].mean()
        price_path = generate_price_path(self.S0, self.r, realized_vol, self.total_days, seed=42)

        current_date = self.start_date

        print(f"[{self.ticker}] Starting Simulation for {self.total_days} days...")
        print(
            f"{'Date':<12} | {'Spot':<8} | {'Opt Value':<10} | {'Pf Delta':<10} | {'Hedge Shares':<12} | {'Cash':<12}")
        print("-" * 85)

        for t in range(self.total_days + 1):
            # A. Get Market Data for today
            S_t = price_path[t]

            # B. Manage Expirations & Options Value
            portfolio_value = 0.0
            net_delta_contracts = 0.0

            # Check each option in the portfolio
            active_options_mask = self.portfolio['expiration_date'] > current_date

            # If date is an expiration date for some options, settle them
            expiring_mask = self.portfolio['expiration_date'] == current_date
            if expiring_mask.any():
                self._settle_expiring_options(self.portfolio[expiring_mask], S_t)
                # Remove expired from active set for delta calc

            # C. Calculate Greeks for Active Options
            active_indices = self.portfolio.index[active_options_mask]

            for idx in active_indices:
                row = self.portfolio.loc[idx]
                K = row['strike']
                T_curr = yearfrac(current_date, row['expiration_date'])
                sigma = row['iv']
                otype = row['option_type']
                size = row['contractSize']

                # Pricing
                if otype == 'C':
                    price = call_price(S_t, K, T_curr, self.r, self.q, sigma)
                    delta = delta_call(S_t, K, T_curr, self.r, self.q, sigma)
                else:
                    price = put_price(S_t, K, T_curr, self.r, self.q, sigma)
                    delta = delta_put(S_t, K, T_curr, self.r, self.q, sigma)

                # Aggregate Value and Delta
                portfolio_value += price * size
                net_delta_contracts += delta * size

            # D. Execute Hedge (Delta Neutral)
            # We want Total Delta = Option Delta + Stock Delta = 0
            # Stock Delta = 1.0 per share.
            # Thus: Target Stock Shares = -Option Delta
            target_shares = -net_delta_contracts
            trade_shares = target_shares - self.stock_shares

            if abs(trade_shares) > 0.001:  # Avoid microscopic trades
                cost = trade_shares * S_t
                self.cash -= cost
                self.stock_shares += trade_shares

            # E. Accrue Interest on Cash (Daily)
            dt_days = 1.0 / 365.0
            interest = self.cash * (math.exp(self.r * dt_days) - 1.0)
            self.cash += interest

            # Logging
            total_equity = self.cash + (self.stock_shares * S_t) + portfolio_value
            self.daily_log.append({
                'Date': current_date,
                'Spot': S_t,
                'OptionValue': portfolio_value,
                'StockValue': self.stock_shares * S_t,
                'Cash': self.cash,
                'TotalEquity': total_equity
            })

            if t % 30 == 0 or t == self.total_days:  # Print monthly
                print(
                    f"{current_date} | {S_t:<8.2f} | {portfolio_value:<10.2f} | {net_delta_contracts:<10.2f} | {self.stock_shares:<12.2f} | {self.cash:<12.2f}")

            current_date += dt.timedelta(days=1)

    def _settle_expiring_options(self, expiring_df, S_t):
        for _, row in expiring_df.iterrows():
            K = row['strike']
            otype = row['option_type']
            size = row['contractSize']

            payoff = 0.0
            if otype == 'C':
                payoff = max(S_t - K, 0.0)
            else:
                payoff = max(K - S_t, 0.0)

            total_payoff = payoff * size
            self.cash += total_payoff
            # Note: We don't print every payoff to keep output clean,
            # but you can add a print statement here.

    def report_results(self):
        final_log = self.daily_log[-1]

        # Unhedged P&L Logic:
        # Initial Cost vs (Final Payoffs + Final Option Values)
        # However, simulating "Unhedged" accurately requires tracking a separate account
        # that bought options and did nothing else.
        # For simplicity, we compare Hedged Equity to what?
        # A pure delta-neutral strategy seeks to capture Volatility premium and eliminate directional risk.
        # Profit = Total Equity (which should be ideally > 0 if realized vol < implied vol, or near 0).

        print("\n" + "=" * 30)
        print(f"RESULTS FOR {self.ticker}")
        print("=" * 30)
        print(f"Final Spot Price:    ${final_log['Spot']:.2f}")
        print(f"Final Cash Balance:  ${final_log['Cash']:.2f}")
        print(f"Stock Position Value:${final_log['StockValue']:.2f}")
        print(f"Remaining Opt Value: ${final_log['OptionValue']:.2f}")
        print("-" * 30)
        print(f"TOTAL HEDGED P&L:    ${final_log['TotalEquity']:.2f}")
        print("=" * 30 + "\n")


# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
def main():
    # 1. Load Data
    print("Loading portfolio from", DATA_FILE)
    df = pd.read_csv(DATA_FILE)

    # 2. Setup
    start_date = pd.to_datetime(df['date'].iloc[0]).date()

    # 3. Select a specific ticker to simulate (e.g., AAPL)
    # The universe has many, pick one for demonstration.
    target_ticker = "AAPL"
    ticker_df = df[df['ticker'] == target_ticker]

    if ticker_df.empty:
        print(f"Ticker {target_ticker} not found in file.")
        return

    # 4. Run Strategy
    strategy = DeltaHedgingStrategy(target_ticker, ticker_df, start_date)
    strategy.run_simulation()
    strategy.report_results()


if __name__ == "__main__":
    main()