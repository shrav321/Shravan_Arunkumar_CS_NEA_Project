# analysis.py

import math
from datetime import date, datetime

from market import compute_historical_volatility


def Build_Model_Inputs(ctx):
    """
    Build numerical inputs required for Monte Carlo GBM simulation.
    """
    # Checks whether the context passed in is valid 
    if not ctx.get("valid", False):
        raise ValueError("Invalid contract context")

    ticker = ctx["ticker"]
    expiry = ctx["expiry"]
    closes = ctx.get("closes")

    if closes is None or len(closes) < 2:
        raise ValueError("Insufficient price history to build model inputs")

    # Volatility from historical prices
    sigma = compute_historical_volatility(ticker)

    # Estimate drift using mean of log returns
    log_returns = []
    for i in range(1, len(closes)):
        p0 = float(closes[i - 1])
        p1 = float(closes[i])
        if p0 <= 0 or p1 <= 0:
            continue
        log_returns.append(math.log(p1 / p0))

    if len(log_returns) == 0:
        mu_daily = 0.0
    else:
        mu_daily = sum(log_returns) / len(log_returns)

    mu = mu_daily * 252

    r = 0.05

    # Time to expiry in years
    today = date.today()
    exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
    days_to_expiry = max(1, (exp_date - today).days)
    T_years = days_to_expiry / 252.0

    steps_per_year = 252
    steps = max(1, int(T_years * steps_per_year))

    return {
        "mu": mu,
        "sigma": sigma,
        "r": r,
        "T_years": T_years,
        "steps": steps,
    }
