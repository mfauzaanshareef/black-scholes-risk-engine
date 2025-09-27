"""
Black–Scholes Greeks for calls and puts with continuous dividend yield q.
Returns Greeks on a PER-SHARE basis. Multiply by contract size (e.g., 100) and quantity to scale.
"""
from __future__ import annotations
import math
from typing import Tuple
from bs_pricer import phi, N, d1_d2

def delta_call(S,K,T,r,q,sigma) -> float:
    d1, _ = d1_d2(S,K,T,r,q,sigma)
    return math.exp(-q*T) * N(d1)

def delta_put(S,K,T,r,q,sigma) -> float:
    d1, _ = d1_d2(S,K,T,r,q,sigma)
    return math.exp(-q*T) * (N(d1) - 1.0)

def gamma(S,K,T,r,q,sigma) -> float:
    d1, _ = d1_d2(S,K,T,r,q,sigma)
    return math.exp(-q*T) * phi(d1) / (S * sigma * math.sqrt(T))

def vega(S,K,T,r,q,sigma) -> float:
    d1, _ = d1_d2(S,K,T,r,q,sigma)
    return S * math.exp(-q*T) * math.sqrt(T) * phi(d1)  # per 1.00 vol (100%)

def theta_call(S,K,T,r,q,sigma) -> float:
    d1, d2 = d1_d2(S,K,T,r,q,sigma)
    term1 = - (S*math.exp(-q*T)*phi(d1)*sigma) / (2.0*math.sqrt(T))
    term2 = - r*K*math.exp(-r*T)*N(d2)
    term3 = + q*S*math.exp(-q*T)*N(d1)
    return term1 + term2 + term3  # per year

def theta_put(S,K,T,r,q,sigma) -> float:
    d1, d2 = d1_d2(S,K,T,r,q,sigma)
    term1 = - (S*math.exp(-q*T)*phi(d1)*sigma) / (2.0*math.sqrt(T))
    term2 = + r*K*math.exp(-r*T)*N(-d2)
    term3 = - q*S*math.exp(-q*T)*N(-d1)
    return term1 + term2 + term3  # per year

def rho_call(S,K,T,r,q,sigma) -> float:
    _, d2 = d1_d2(S,K,T,r,q,sigma)
    return K*T*math.exp(-r*T)*N(d2)

def rho_put(S,K,T,r,q,sigma) -> float:
    _, d2 = d1_d2(S,K,T,r,q,sigma)
    return -K*T*math.exp(-r*T)*N(-d2)
