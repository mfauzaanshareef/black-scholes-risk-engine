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


# ------------------------------------------------------------------------------
# Helper: Geometric Brownian Motion Generator
# ------------------------------------------------------------------------------
def generate_price_path(S0, mu, sigma, days, seed=None):
    if seed is not None:
        np.random.seed(seed)

    dt_step = 1.0 / TRADING_DAYS_PER_YEAR
    prices = [S0]
    current_price = S0

    for _ in range(days):
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

        # Portfolio State (Hedged)
        self.cash = 0.0
        self.stock_shares = 0.0

        # Portfolio State (Unhedged Tracking)
        self.accumulated_payoffs = 0.0

        # Logging
        self.daily_log = []

        # Market Parameters
        self.S0 = self.portfolio['underlying_price'].iloc[0]
        self.r = self.portfolio['r_cont'].iloc[0]
        self.q = self.portfolio['q'].iloc[0]

        # Determine Simulation Horizon
        self.portfolio['expiration_date'] = pd.to_datetime(self.portfolio['expiration']).dt.date
        last_expiry = self.portfolio['expiration_date'].max()
        self.total_days = (last_expiry - self.start_date).days

        # Initial Costs
        self.initial_cost = (self.portfolio['option_price'] * self.portfolio['contractSize']).sum()

        # Fund the Hedged Portfolio (Debt)
        self.cash -= self.initial_cost

        print(f"\n[{ticker}] Initial Portfolio Cost: ${self.initial_cost:,.2f}")

    def run_simulation(self):
        # Use average IV as realized volatility for the simulation
        realized_vol = self.portfolio['iv'].mean()
        price_path = generate_price_path(self.S0, self.r, realized_vol, self.total_days, seed=42)

        current_date = self.start_date

        print(f"[{self.ticker}] Starting Simulation for {self.total_days} days...")
        print(
            f"{'Date':<12} | {'Spot':<8} | {'Opt Value':<10} | {'Pf Delta':<10} | {'Hedge Shares':<12} | {'Hedged Eq':<12}")
        print("-" * 85)

        for t in range(self.total_days + 1):
            S_t = price_path[t]

            portfolio_value = 0.0
            net_delta_contracts = 0.0

            # --- 1. Manage Expirations ---
            active_options_mask = self.portfolio['expiration_date'] > current_date
            expiring_mask = self.portfolio['expiration_date'] == current_date

            if expiring_mask.any():
                self._settle_expiring_options(self.portfolio[expiring_mask], S_t)

            # --- 2. Calculate Value & Greeks for Active Options ---
            active_indices = self.portfolio.index[active_options_mask]

            for idx in active_indices:
                row = self.portfolio.loc[idx]
                K = row['strike']
                T_curr = yearfrac(current_date, row['expiration_date'])
                sigma = row['iv']
                otype = row['option_type']
                size = row['contractSize']

                if otype == 'C':
                    price = call_price(S_t, K, T_curr, self.r, self.q, sigma)
                    delta = delta_call(S_t, K, T_curr, self.r, self.q, sigma)
                else:
                    price = put_price(S_t, K, T_curr, self.r, self.q, sigma)
                    delta = delta_put(S_t, K, T_curr, self.r, self.q, sigma)

                portfolio_value += price * size
                net_delta_contracts += delta * size

            # --- 3. Execute Hedge (Delta Neutral) ---
            target_shares = -net_delta_contracts
            trade_shares = target_shares - self.stock_shares

            if abs(trade_shares) > 0.001:
                cost = trade_shares * S_t
                self.cash -= cost
                self.stock_shares += trade_shares

            # --- 4. Accrue Interest on Cash ---
            dt_days = 1.0 / 365.0
            interest = self.cash * (math.exp(self.r * dt_days) - 1.0)
            self.cash += interest

            # --- 5. Logging ---
            # Hedged Equity = Cash + Stock Value + Option Value
            hedged_equity = self.cash + (self.stock_shares * S_t) + portfolio_value

            self.daily_log.append({
                'Date': current_date,
                'Spot': S_t,
                'OptionValue': portfolio_value,
                'HedgedEquity': hedged_equity
            })

            if t % 30 == 0 or t == self.total_days:
                print(
                    f"{current_date} | {S_t:<8.2f} | {portfolio_value:<10.2f} | {net_delta_contracts:<10.2f} | {self.stock_shares:<12.2f} | {hedged_equity:<12.2f}")

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

            # Add to Hedged Cash
            self.cash += total_payoff

            # Track for Unhedged P&L
            self.accumulated_payoffs += total_payoff

    def report_results(self):
        final_log = self.daily_log[-1]

        # Unhedged P&L = (Final Option Value + Accumulated Payoffs) - Initial Cost
        final_opt_val = final_log['OptionValue']
        unhedged_pnl = (final_opt_val + self.accumulated_payoffs) - self.initial_cost

        # Hedged P&L = Final Equity (since we started with 0 equity by borrowing cash)
        hedged_pnl = final_log['HedgedEquity']

        print("\n" + "=" * 30)
        print(f"RESULTS FOR {self.ticker}")
        print("=" * 30)
        print(f"Initial Portfolio Cost: ${self.initial_cost:,.2f}")
        print(f"Final Option Value:     ${final_opt_val:,.2f}")
        print(f"Accumulated Payoffs:    ${self.accumulated_payoffs:,.2f}")
        print("-" * 30)
        print(f"UNHEDGED P&L:           ${unhedged_pnl:,.2f}")
        print(f"HEDGED P&L:             ${hedged_pnl:,.2f}")
        print("=" * 30 + "\n")


# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
def main():
    print("Loading portfolio from", DATA_FILE)
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found. Please run generate_options_chain.py first.")
        return

    start_date = pd.to_datetime(df['date'].iloc[0]).date()

    target_ticker = "AAPL"
    ticker_df = df[df['ticker'] == target_ticker]

    if ticker_df.empty:
        print(f"Ticker {target_ticker} not found.")
        return

    strategy = DeltaHedgingStrategy(target_ticker, ticker_df, start_date)
    strategy.run_simulation()
    strategy.report_results()


if __name__ == "__main__":
    main()