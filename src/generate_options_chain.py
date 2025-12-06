# requirements: pandas, numpy, yfinance
# pip install pandas numpy yfinance

import math, random, datetime as dt
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------
# Configuration
# ---------------------------

VAL_DATE = dt.date(2025, 3, 13)  # Simulation Start Date
CONTRACT_SIZE = 100
# Update these two yields if needed to your exact "current" snapshot
YIELD_3M_SIMPLE = 0.0386  # ~3.86% recent 3M T-bill yield
YIELD_6M_SIMPLE = 0.0372  # ~3.72% recent 6M T-bill yield
R3M_CONT = math.log(1.0 + YIELD_3M_SIMPLE)
R6M_CONT = math.log(1.0 + YIELD_6M_SIMPLE)

# Target maturity lengths in calendar days
DAYS_3M = 91
DAYS_6M = 182

# Strikes as a % of spot (moneyness)
M_GRID = [0.80, 0.90, 1.00, 1.10, 1.20]

# Random seed for reproducibility
random.seed(42)

# Tickers
RAW_TICKERS = ["AAPL", "MSFT", "NVDA", "ORCL", "CRM", "JPM", "BAC", "WFC", "C", "GS", "JNJ", "MRK", "ABBV", "PFE",
               "UNH",
               "AMZN", "TSLA", "HD", "MCD", "NKE", "WMT", "COST", "PG", "KO", "PEP", "XOM", "CVX", "COP", "SLB", "EOG",
               "CAT", "HON", "BA", "GE", "UPS", "LIN", "APD", "SHW", "DOW", "NUE", "GOOGL", "META", "DIS", "NFLX", "T",
               "NEE", "DUK", "SO", "EXC", "AEP", "AMT", "PLD", "EQIX", "SPG", "O"]


# ---------------------------
# Black–Scholes helpers
# ---------------------------

def N(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def d1_d2(S, K, T, r, q, sigma):
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return d1, d1 - sigma * math.sqrt(T)


def call_price(S, K, T, r, q, sigma):
    if T <= 0:
        return max(S - K, 0.0)
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    return S * math.exp(-q * T) * N(d1) - K * math.exp(-r * T) * N(d2)


def put_price(S, K, T, r, q, sigma):
    if T <= 0:
        return max(K - S, 0.0)
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    return K * math.exp(-r * T) * N(-d2) - S * math.exp(-q * T) * N(-d1)


def vega_per_vol(S, K, T, r, q, sigma):
    d1, _ = d1_d2(S, K, T, r, q, sigma)
    return S * math.exp(-q * T) * math.sqrt(T) * (math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi))


def implied_vol_from_price(price, S, K, T, r, q, opt_type, tol=1e-8, maxit=100):
    if price <= 0:
        return float("nan"), False
    lo, hi = 1e-4, 3.0
    sigma = 0.25
    for _ in range(maxit):
        model = call_price(S, K, T, r, q, sigma) if opt_type == "C" else put_price(S, K, T, r, q, sigma)
        diff = model - price
        if abs(diff) < tol:
            return sigma, True
        v = vega_per_vol(S, K, T, r, q, sigma)
        if v > 1e-10:
            sigma_try = sigma - diff / v
            if lo < sigma_try < hi:
                sigma = sigma_try
        model_lo = call_price(S, K, T, r, q, lo) if opt_type == "C" else put_price(S, K, T, r, q, lo)
        if (model_lo - price) * diff <= 0:
            hi = sigma
        else:
            lo = sigma
        sigma = 0.5 * (lo + hi)
    return sigma, False


def yearfrac(start: dt.date, end: dt.date) -> float:
    return (end - start).days / 365.0


# ---------------------------
# Data helpers: prices, current chains, parity q
# ---------------------------

def get_close_on_date(ticker: str, the_date: dt.date) -> Optional[float]:
    # 1. Try fetching around the specific date
    start = the_date - dt.timedelta(days=5)
    end = the_date + dt.timedelta(days=5)

    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    except Exception:
        df = pd.DataFrame()

    # 2. If empty (e.g., date is in future), fallback to recent history from TODAY
    if df.empty:
        end_now = dt.date.today() + dt.timedelta(days=1)
        start_now = end_now - dt.timedelta(days=7)
        try:
            df = yf.download(ticker, start=start_now, end=end_now, progress=False, auto_adjust=False)
        except Exception:
            return None

    if df.empty:
        return None

    df = df.reset_index()
    # Normalize Date column to date objects if needed
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.date

    # 3. Try to find the exact requested date
    if "Date" in df.columns:
        row = df.loc[df["Date"] == the_date]
        if not row.empty:
            return float(row["Adj Close"].iloc[0])

    # 4. Fallback: return the *latest available* price (handles future date simulation)
    if "Adj Close" in df.columns and len(df) > 0:
        return float(df.iloc[-1]["Adj Close"])

    return None


def _mid_quote(bid, ask, last):
    if pd.notna(bid) and pd.notna(ask) and ask > 0:
        return 0.5 * (bid + ask)
    return last if pd.notna(last) and last > 0 else np.nan


def estimate_q_from_parity_current(ticker: str) -> Optional[float]:
    tk = yf.Ticker(ticker)
    try:
        if not tk.options:
            return None
    except Exception:
        return None

    today = dt.date.today()
    target_days = 30
    exps = []
    try:
        for e in tk.options:
            d = dt.datetime.strptime(e, "%Y-%m-%d").date()
            exps.append((e, abs((d - today).days - target_days)))
    except Exception:
        return None

    exps.sort(key=lambda t: t[1])
    if not exps: return None

    exp = exps[0][0]
    try:
        ch = tk.option_chain(exp)
        S = tk.fast_info.get("lastPrice") or tk.info.get("regularMarketPrice")
    except Exception:
        return None

    if not S: return None
    S = float(S)

    calls = ch.calls.copy()
    puts = ch.puts.copy()
    calls["mid"] = calls.apply(lambda r: _mid_quote(r.get("bid"), r.get("ask"), r.get("lastPrice")), axis=1)
    puts["mid"] = puts.apply(lambda r: _mid_quote(r.get("bid"), r.get("ask"), r.get("lastPrice")), axis=1)

    if calls.empty or puts.empty: return None

    kc = calls.iloc[(calls["strike"] - S).abs().argsort()[:1]]
    kp = puts.iloc[(puts["strike"] - S).abs().argsort()[:1]]
    if kc.empty or kp.empty: return None

    K = float(kc["strike"].values[0])
    C = float(kc["mid"].values[0])
    P = float(kp["mid"].values[0])
    T = yearfrac(today, dt.datetime.strptime(exp, "%Y-%m-%d").date())

    r_cont = R3M_CONT if T <= 0.5 else R6M_CONT
    denom = (C - P + K * math.exp(-r_cont * T))
    if denom <= 0 or T <= 0: return None

    q = (1.0 / T) * math.log(S / denom)
    return float(np.clip(q, -0.10, 0.10))


def baseline_iv_now(ticker: str) -> Optional[float]:
    tk = yf.Ticker(ticker)
    try:
        if not tk.options: return None
    except Exception:
        return None

    today = dt.date.today()
    target_days = 30
    exps = []
    try:
        for e in tk.options:
            d = dt.datetime.strptime(e, "%Y-%m-%d").date()
            exps.append((e, abs((d - today).days - target_days)))
    except Exception:
        return None

    exps.sort(key=lambda t: t[1])
    if not exps: return None
    exp = exps[0][0]

    try:
        ch = tk.option_chain(exp)
        S = tk.fast_info.get("lastPrice") or tk.info.get("regularMarketPrice")
    except Exception:
        return None

    if not S: return None
    S = float(S)

    calls = ch.calls.copy()
    puts = ch.puts.copy()
    calls["mid"] = calls.apply(lambda r: _mid_quote(r.get("bid"), r.get("ask"), r.get("lastPrice")), axis=1)
    puts["mid"] = puts.apply(lambda r: _mid_quote(r.get("bid"), r.get("ask"), r.get("lastPrice")), axis=1)

    T = yearfrac(today, dt.datetime.strptime(exp, "%Y-%m-%d").date())
    r_cont = R3M_CONT if T <= 0.5 else R6M_CONT

    ivs = []
    for df, typ in [(calls, "C"), (puts, "P")]:
        if df.empty: continue
        row = df.iloc[(df["strike"] - S).abs().argsort()[:1]]
        K = float(row["strike"].values[0])
        mid = float(row["mid"].values[0])
        if not (mid and mid > 0): continue
        iv, ok = implied_vol_from_price(mid, S, K, T, r_cont, 0.0, typ)
        if ok and 0.03 <= iv <= 1.0:
            ivs.append(iv)

    if not ivs: return None
    return float(np.median(ivs))


def indicated_dividend_yield(ticker: str) -> Optional[float]:
    try:
        tk = yf.Ticker(ticker)
        y = tk.info.get("dividendYield", None)
        if y is None: return None
        return float(np.clip(y, 0.0, 0.10))
    except Exception:
        return None


# ---------------------------
# Build synthetic chain
# ---------------------------

def choose_expirations(val_date: dt.date) -> Tuple[dt.date, dt.date]:
    exp3 = val_date + dt.timedelta(days=DAYS_3M)
    exp6 = val_date + dt.timedelta(days=DAYS_6M)
    return exp3, exp6


def ensure_both_maturities(assignments: List[str]) -> List[str]:
    if "3M" not in assignments: assignments[0] = "3M"
    if "6M" not in assignments: assignments[-1] = "6M"
    return assignments


def iv_with_skew(iv_atm: float, K_over_S: float, term_adj: float, skew_coeff: float = -0.25) -> float:
    return float(np.clip(iv_atm * term_adj * math.exp(skew_coeff * math.log(K_over_S)), 0.05, 1.00))


def build_for_ticker(ticker: str) -> List[Dict]:
    # 1) Spot Price
    S = get_close_on_date(ticker, VAL_DATE)
    if S is None:
        print(f"[warn] missing spot for {ticker} (using fallback may have failed)")
        return []

    # 2) Expirations
    exp3, exp6 = choose_expirations(VAL_DATE)
    T3, T6 = yearfrac(VAL_DATE, exp3), yearfrac(VAL_DATE, exp6)
    r3, r6 = YIELD_3M_SIMPLE, YIELD_6M_SIMPLE
    r3c, r6c = R3M_CONT, R6M_CONT

    # 3) Baseline IV
    iv0 = baseline_iv_now(ticker)
    if iv0 is None: iv0 = 0.22
    iv3_atm = iv0 * 1.00
    iv6_atm = iv0 * 0.95

    # 4) Dividend Yield
    q = estimate_q_from_parity_current(ticker)
    if q is None: q = indicated_dividend_yield(ticker)
    if q is None: q = 0.005
    q = float(np.clip(q, 0.0, 0.10))

    # 5) Moneyness Assignment
    assign = [random.choice(["3M", "6M"]) for _ in M_GRID]
    assign = ensure_both_maturities(assign)

    rows = []
    for m, tag in zip(M_GRID, assign):
        K = round(S * m, 2)
        if tag == "3M":
            T, r, r_cont, iv_atm = T3, r3, r3c, iv3_atm
        else:
            T, r, r_cont, iv_atm = T6, r6, r6c, iv6_atm

        iv = iv_with_skew(iv_atm, K_over_S=m, term_adj=1.0)
        c = call_price(S, K, T, r_cont, q, iv)
        p = put_price(S, K, T, r_cont, q, iv)

        rows.append({
            "date": VAL_DATE.isoformat(),
            "expiration": (exp3 if tag == "3M" else exp6).isoformat(),
            "ticker": ticker,
            "underlying_price": round(S, 4),
            "option_type": "C",
            "strike": K,
            "T_years": round(T, 6),
            "q": round(q, 6),
            "iv": round(iv, 6),
            "option_price": round(c, 4),
            "moneyness_K_over_S": round(m, 4),
            "contractSize": CONTRACT_SIZE,
            "r": r,
            "r_cont": round(r_cont, 8),
        })
        rows.append({
            "date": VAL_DATE.isoformat(),
            "expiration": (exp3 if tag == "3M" else exp6).isoformat(),
            "ticker": ticker,
            "underlying_price": round(S, 4),
            "option_type": "P",
            "strike": K,
            "T_years": round(T, 6),
            "q": round(q, 6),
            "iv": round(iv, 6),
            "option_price": round(p, 4),
            "moneyness_K_over_S": round(m, 4),
            "contractSize": CONTRACT_SIZE,
            "r": r,
            "r_cont": round(r_cont, 8),
        })

    return rows


# ---------------------------
# Run and Save
# ---------------------------
if __name__ == "__main__":
    all_rows = []
    print(f"Generating synthetic chains for {len(RAW_TICKERS)} tickers...")

    for t in RAW_TICKERS:
        try:
            res = build_for_ticker(t)
            all_rows.extend(res)
        except Exception as e:
            print(f"[error] {t}: {e}")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values(["ticker", "expiration", "option_type", "strike"]).reset_index(drop=True)

    # UPDATED: Write to current directory with simple filename
    out_path = "synthetic_options_2025-03-13.csv"
    df.to_csv(out_path, index=False)
    print(f"Success! Wrote {len(df)} rows to {out_path}")