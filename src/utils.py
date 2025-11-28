"""
Utility functions for date handling and robust numerics.
"""

from __future__ import annotations
import datetime as _dt
from typing import Optional

def last_business_friday(ref_date: Optional[_dt.date] = None) -> _dt.date:
    """
    Return the most recent Friday (business week) on or before ref_date.
    If ref_date is None, uses today's date.
    """
    if ref_date is None:
        ref_date = _dt.date.today()
    # 0=Mon ... 4=Fri ... 6=Sun
    offset = (ref_date.weekday() - 4) % 7
    return ref_date - _dt.timedelta(days=offset)

def yearfrac(start: _dt.date, end: _dt.date, basis: str = "ACT/365") -> float:
    """
    Compute year fraction between two dates.
    Supported: ACT/365 (default), ACT/252 (trading days approx).
    """
    delta = (end - start).days
    if basis.upper() == "ACT/252":
        return delta / 252.0
    return delta / 365.0

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
