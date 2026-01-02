# analysis.py

import math
from datetime import date, datetime
from market import compute_historical_volatility
import random
from typing import Any, Dict, List


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

def _randn_box_muller() -> float:
   
    #Generate one standard normal random variable using Box-Muller.
    u1 = random.random()
    u2 = random.random()
    # Guard against log(0)
    if u1 <= 0.0:
        u1 = 1e-12
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def Run_Monte_Carlo(ctx: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulate GBM price paths to expiry and return discounted payoffs plus
    a small subset of paths for plotting.


    inputs must include:
    - mu
    - sigma
    - r
    - T_years
    - steps
    - N
    - seed
    """
    if ctx is None or inputs is None:
        raise ValueError("ctx and inputs must not be None")

    for k in ("S", "strike", "type"):
        if k not in ctx:
            raise ValueError(f"ctx missing required field: {k}")

    for k in ("mu", "sigma", "r", "T_years", "steps", "N", "seed"):
        if k not in inputs:
            raise ValueError(f"inputs missing required field: {k}")

    try:
        S0 = float(ctx["S"])
        K = float(ctx["strike"])
        opt_type = str(ctx["type"]).upper()

        mu = float(inputs["mu"])
        sigma = float(inputs["sigma"])
        r = float(inputs["r"])
        T = float(inputs["T_years"])
        steps = int(inputs["steps"])
        N = int(inputs["N"])
        seed = int(inputs["seed"])
    except (TypeError, ValueError):
        raise ValueError("ctx and inputs contain non-numeric fields where numeric expected")

    if S0 <= 0:
        raise ValueError("ctx['S'] must be > 0")
    if K <= 0:
        raise ValueError("ctx['strike'] must be > 0")
    if opt_type not in ("C", "P"):
        raise ValueError("ctx['type'] must be 'C' or 'P'")
    if sigma < 0:
        raise ValueError("inputs['sigma'] must be >= 0")
    if T <= 0:
        raise ValueError("inputs['T_years'] must be > 0")
    if steps < 1:
        raise ValueError("inputs['steps'] must be >= 1")
    if N < 1:
        raise ValueError("inputs['N'] must be >= 1")

    random.seed(seed)

    dt = T / float(steps)
    sqrt_dt = math.sqrt(dt)
    drift = (mu - 0.5 * sigma * sigma) * dt
    disc = math.exp(-r * T)

    discounted_payoffs: List[float] = []
    paths_subset: List[List[float]] = []

    pick_every = max(1, N // 50)

    for p in range(1, N + 1):
        S_t = S0

        keep_path = (p % pick_every == 0)
        if keep_path:
            path_series: List[float] = [S_t]

        for _ in range(steps):
            z = _randn_box_muller()
            growth = drift + (sigma * sqrt_dt * z)
            S_t = S_t * math.exp(growth)

            if keep_path:
                path_series.append(S_t)

        if opt_type == "C":
            payoff = max(S_t - K, 0.0)
        else:
            payoff = max(K - S_t, 0.0)

        discounted_payoffs.append(float(payoff * disc))

        if keep_path:
            paths_subset.append(path_series)

    return {
        "paths_subset": paths_subset,
        "discounted_payoffs": discounted_payoffs,
        "N": N,
        "steps": steps,
        "discount_factor": float(disc),
    }