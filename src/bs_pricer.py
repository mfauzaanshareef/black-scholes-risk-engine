"""
Black–Scholes pricing (with continuous dividend yield) and d1/d2 helpers.
No external dependencies beyond the Python stdlib and math.
"""
# from __future__ import annotation
import math
from dataclasses import dataclass

SQRT2PI = math.sqrt(2.0 * math.pi)

def phi(x: float) -> float:
    "Standard normal pdf."
    return math.exp(-0.5 * x * x) / SQRT2PI

def N(x: float) -> float:
    "Standard normal cdf via error function."
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

@dataclass
class BSInputs:
    S: float       # spot
    K: float       # strike
    T: float       # time to maturity in years
    r: float       # risk-free (continuous)
    q: float       # dividend yield (continuous)
    sigma: float   # volatility (annualized)

def d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        raise ValueError("Invalid inputs for d1/d2")
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2

def call_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    Black–Scholes price for a European call with continuous dividend yield q.
    """
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        # With zero vol, option is its discounted intrinsic under BS limit
        return max(S - K * math.exp(-r * T), 0.0)
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    return S * math.exp(-q * T) * N(d1) - K * math.exp(-r * T) * N(d2)

def put_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    Black–Scholes price for a European put with continuous dividend yield q.
    """
    if T <= 0:
        return max(K - S, 0.0)
    if sigma <= 0:
        return max(K * math.exp(-r * T) - S, 0.0)
    d1, d2 = d1_d2(S, K, T, r, q, sigma)
    return K * math.exp(-r * T) * N(-d2) - S * math.exp(-q * T) * N(-d1)
