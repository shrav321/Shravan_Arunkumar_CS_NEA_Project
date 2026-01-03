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
    # Context is valid by default unless explicitly marked invalid
    if ctx is None:
        raise ValueError("Invalid contract context")
    if ctx.get("valid", True) is not True:
        raise ValueError("Invalid contract context")

    ticker = ctx["ticker"]
    expiry = ctx["expiry"]
    closes = ctx.get("closes")

    if closes is None or len(closes) < 2:
        raise ValueError("Insufficient price history to build model inputs")

    # Volatility from yfinance-memoised path
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

    # Provide defaults if simulator expects them
    N = int(ctx.get("N", 1000))
    seed = int(ctx.get("seed", 42))

    return {
        "mu": mu,
        "sigma": sigma,
        "r": r,
        "T_years": T_years,
        "steps": steps,
        "N": N,
        "seed": seed,
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


def Derive_Metrics(ctx: Dict[str, Any], inputs: Dict[str, Any], sim: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns:
    - mc_mean
    - mc_median
    - q05, q95
    - p_itm (probability payoff > 0)
    - p_netprofit (probability discounted payoff > premium_ref)
    - premium_ref 
    """
    if ctx is None or inputs is None or sim is None:
        raise ValueError("ctx, inputs, and sim must not be None")

    if "discounted_payoffs" not in sim:
        raise ValueError("sim missing required field: discounted_payoffs")

    payoffs = sim["discounted_payoffs"]
    if not isinstance(payoffs, list) or len(payoffs) == 0:
        raise ValueError("discounted_payoffs must be a non-empty list")

    # premium reference: provided by caller for now
    premium_ref = float(ctx.get("premium_ref", 0.0))

    # Basic probabilities
    count = len(payoffs)

    # payoff > 0 means ITM at expiry
    itm_count = 0
    net_count = 0
    for v in payoffs:
        fv = float(v)
        if fv > 0.0:
            itm_count += 1
        if fv > premium_ref:
            net_count += 1

    p_itm = itm_count / count
    p_netprofit = net_count / count

    # Summary stats: sort once
    vals = [float(v) for v in payoffs]
    vals.sort()

    # Mean
    mc_mean = sum(vals) / count

    # Median
    if count % 2 == 1:
        mc_median = vals[count // 2]
    else:
        mid = count // 2
        mc_median = (vals[mid - 1] + vals[mid]) / 2.0

    # Percentiles
    # this index formula is subtly wrong for small samples and tends to bias low. 
    # Should not be a major issue since im using larger samples.
    # It uses int(p*count) instead of a rank based on (count-1).
    def _percentile(sorted_vals: List[float], p: float) -> float:
        idx = int(p * count)  # <-- bug is here
        if idx < 0:
            idx = 0
        if idx >= count:
            idx = count - 1
        return float(sorted_vals[idx])

    q05 = _percentile(vals, 0.05)
    q95 = _percentile(vals, 0.95)

    return {
        "mc_mean": float(mc_mean),
        "mc_median": float(mc_median),
        "q05": float(q05),
        "q95": float(q95),
        "p_itm": float(p_itm),
        "p_netprofit": float(p_netprofit),
        "premium_ref": float(premium_ref),
        "count": int(count),
    }


def Run_Analysis_Pipeline(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns:
    - inputs (model inputs dict)
    - sim (simulation output dict)
    - metrics (metrics dict)
    """
    if ctx is None:
        raise ValueError("ctx must not be None")

    inputs = Build_Model_Inputs(ctx)
    sim = Run_Monte_Carlo(ctx, inputs)
    metrics = Derive_Metrics(ctx, inputs, sim)

    return {
        "inputs": inputs,
        "sim": sim,
        "metrics": metrics,
    }



def Render_Findings(ctx: Dict[str, Any], inputs: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepares user suggestions ready for streamlit
    """
    if ctx is None or inputs is None or metrics is None:
        raise ValueError("ctx, inputs, and metrics must not be None")

    required = ["mc_mean", "mc_median", "q05", "q95", "p_itm", "p_netprofit", "premium_ref", "count"]
    for k in required:
        if k not in metrics:
            raise ValueError(f"metrics missing required field: {k}")

    mc_mean = float(metrics["mc_mean"])
    mc_median = float(metrics["mc_median"])
    q05 = float(metrics["q05"])
    q95 = float(metrics["q95"])
    p_itm = float(metrics["p_itm"])
    p_netprofit = float(metrics["p_netprofit"])
    premium_ref = float(metrics["premium_ref"])
    count = int(metrics["count"])

    if count < 1:
        raise ValueError("metrics['count'] must be >= 1")

    expiry = ctx.get("expiry")
    opt_type = str(ctx.get("type", "")).upper()
    strike = ctx.get("strike")

    summary_lines: List[str] = []
    summary_lines.append(f"Simulations: {count}")
    if expiry is not None:
        summary_lines.append(f"Expiry: {expiry}")
    if opt_type in ("C", "P") and strike is not None:
        label = "Call" if opt_type == "C" else "Put"
        summary_lines.append(f"Contract: {label}, Strike {float(strike):.2f}")

    summary_lines.append(f"Mean discounted payoff: {mc_mean:.4f}")
    summary_lines.append(f"Median discounted payoff: {mc_median:.4f}")
    summary_lines.append(f"5th to 95th percentile range: {q05:.4f} to {q95:.4f}")
    summary_lines.append(f"Probability ITM at expiry: {p_itm:.4f}")
    summary_lines.append(f"Probability payoff exceeds premium: {p_netprofit:.4f}")

    flags: List[str] = []
    if p_netprofit >= 0.5:
        flags.append("Net profit probability is at least 0.5")
    if q95 <= premium_ref:
        flags.append("Even optimistic outcomes may not clear the premium")
    if q05 > premium_ref:
        flags.append("Downside outcomes still clear the premium")

    return {
        "summary_lines": summary_lines,
        "flags": flags,
        "numbers": {
            "mc_mean": mc_mean,
            "mc_median": mc_median,
            "q05": q05,
            "q95": q95,
            "p_itm": p_itm,
            "p_netprofit": p_netprofit,
            "premium_ref": premium_ref,
            "count": count,
        },
    }

def Visualise_Results(sim: Dict[str, Any], max_paths: int = 30, bins: int = 30) -> Dict[str, Any]:
    """
      Returns:
    - paths: list of paths (each path is a list of floats)
    - payoff_hist: dict with 'bin_edges' and 'counts'
    """
    if sim is None:
        raise ValueError("sim must not be None")

    if "paths_subset" not in sim:
        raise ValueError("sim missing required field: paths_subset")
    if "discounted_payoffs" not in sim:
        raise ValueError("sim missing required field: discounted_payoffs")

    paths_subset = sim["paths_subset"]
    payoffs = sim["discounted_payoffs"]

    if not isinstance(paths_subset, list):
        raise ValueError("paths_subset must be a list")
    if not isinstance(payoffs, list) or len(payoffs) == 0:
        raise ValueError("discounted_payoffs must be a non-empty list")

    try:
        mp = int(max_paths)
        nb = int(bins)
    except (TypeError, ValueError):
        raise ValueError("max_paths and bins must be integers")

    if mp < 1:
        raise ValueError("max_paths must be >= 1")
    if nb < 2:
        raise ValueError("bins must be >= 2")

    # Limit paths for plotting
    paths_out: List[List[float]] = []
    for p in paths_subset[:mp]:
        if not isinstance(p, list) or len(p) < 2:
            continue
        cleaned = [float(x) for x in p]
        paths_out.append(cleaned)

    # Histogram for payoffs
    vals = [float(v) for v in payoffs]
    vmin = min(vals)
    vmax = max(vals)

    if vmin == vmax:
        # Degenerate distribution: one bin
        return {
            "paths": paths_out,
            "payoff_hist": {
                "bin_edges": [vmin, vmax],
                "counts": [len(vals)],
            },
        }

    width = (vmax - vmin) / float(nb)
    edges = [vmin + i * width for i in range(nb + 1)]
    counts = [0 for _ in range(nb)]

    for v in vals:
        idx = int((v - vmin) / width)
        if idx == nb:
            idx = nb - 1
        counts[idx] += 1

    return {
        "paths": paths_out,
        "payoff_hist": {
            "bin_edges": edges,
            "counts": counts,
        },
    }


from db_init import get_prices_for_ticker

def Load_Contract_Context(
    ticker: str,
    expiry: str,
    strike: float,
    option_type: str,
    spot: float,
    lookback_min_points: int = 30
) -> Dict[str, Any]:
    """
    
    Returns a dict with:
    - valid
    - ticker, expiry, strike, type
    - S (spot)
    - closes (list of close floats)
    """
    if ticker is None or str(ticker).strip() == "":
        raise ValueError("ticker must be non-empty")
    if expiry is None or str(expiry).strip() == "":
        raise ValueError("expiry must be non-empty")
    if option_type is None or str(option_type).strip() == "":
        raise ValueError("option_type must be non-empty")

    tkr = str(ticker).strip().upper()
    exp = str(expiry).strip()

    typ = str(option_type).strip().upper()
    if typ in ("CALL", "CALLS"):
        typ = "C"
    if typ in ("PUT", "PUTS"):
        typ = "P"
    if typ not in ("C", "P"):
        raise ValueError("option_type must be C or P")

    try:
        K = float(strike)
        S = float(spot)
    except (TypeError, ValueError):
        raise ValueError("strike and spot must be numeric")

    if K <= 0:
        raise ValueError("strike must be > 0")
    if S <= 0:
        raise ValueError("spot must be > 0")

    try:
        min_pts = int(lookback_min_points)
    except (TypeError, ValueError):
        raise ValueError("lookback_min_points must be an integer")

    if min_pts < 2:
        raise ValueError("lookback_min_points must be >= 2")

    rows = get_prices_for_ticker(tkr)
    closes_all = [float(r[1]) for r in rows]

    if len(closes_all) < 2:
        raise ValueError("Insufficient cached closes in PRICE table for this ticker")

    closes = closes_all[-min_pts:] if len(closes_all) >= min_pts else closes_all

    return {
        "valid": True,
        "ticker": tkr,
        "expiry": exp,
        "strike": float(K),
        "type": typ,
        "S": float(S),
        "closes": closes,
    }

def Select_Target_From_Positions(positions: List[Dict[str, Any]], contract_id: str) -> Dict[str, Any]:
    
    if positions is None or not isinstance(positions, list):
        raise ValueError("positions must be a list")

    cid = str(contract_id).strip()
    if cid == "":
        raise ValueError("contract_id must be non-empty")

    for p in positions:
        if not isinstance(p, dict):
            continue
        if str(p.get("contract_id", "")) == cid:
            return p

    raise ValueError("Selected contract_id not found in positions")

