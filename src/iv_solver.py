"""
Implied volatility solver for calls and puts using robust bisection with optional Newton steps.
No SciPy required.
"""
from __future__ import annotations
import math
from typing import Tuple, Literal
from bs_pricer import call_price, put_price
from greeks import vega as _vega

def _bounds_call(S,K,T,r,q):
    # No-arbitrage bounds (per share)
    lower = max(S*math.exp(-q*T) - K*math.exp(-r*T), 0.0)
    upper = S*math.exp(-q*T)
    return lower, upper

def _bounds_put(S,K,T,r,q):
    lower = max(K*math.exp(-r*T) - S*math.exp(-q*T), 0.0)
    upper = K*math.exp(-r*T)
    return lower, upper

def implied_vol(price_mkt: float, S: float, K: float, T: float, r: float, q: float,
                opt_type: Literal["C","P"], tol: float = 1e-8, max_iter: int = 100) -> Tuple[float, bool]:
    """
    Solve for sigma such that BS_price(S,K,T,r,q,sigma) = price_mkt.
    Returns (sigma, converged_flag).
    """
    if T <= 0 or S <= 0 or K <= 0:
        return float("nan"), False

    # Check bounds
    if opt_type == "C":
        lo_price, hi_price = _bounds_call(S,K,T,r,q)
        f = lambda s: call_price(S,K,T,r,q,s) - price_mkt
    else:
        lo_price, hi_price = _bounds_put(S,K,T,r,q)
        f = lambda s: put_price(S,K,T,r,q,s) - price_mkt

    if price_mkt < lo_price - 1e-12 or price_mkt > hi_price + 1e-12:
        return float("nan"), False

    # Bisection bracket
    lo, hi = 1e-6, 5.0
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        # Expand hi a bit if needed
        hi = 10.0
        fhi = f(hi)
        if flo * fhi > 0:
            return float("nan"), False

    # Optional Newton refinement inside bisection loop
    sigma = 0.25 * (lo + hi)
    for _ in range(max_iter):
        # Newton step if vega isn't tiny
        v = _vega(S,K,T,r,q,sigma)
        pv = f(sigma)
        if abs(pv) < tol:
            return sigma, True
        if v > 1e-10:
            sigma_new = sigma - pv / v
            if 1e-6 < sigma_new < 10.0:
                sigma = sigma_new
        # Keep bisection guarantees
        if f(lo) * f(sigma) <= 0:
            hi = sigma
        else:
            lo = sigma
        sigma = 0.5 * (lo + hi)

    return sigma, False
