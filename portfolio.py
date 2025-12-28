# portfolio.py

from typing import Dict, List, Tuple, Any


def derive_holdings_from_trade(trades: List[Tuple]) -> Dict[str, int]:
    """
    Derive net contract holdings from a list of trade rows.

    Each trade row is expected in the format:
    (trade_id, contract_id, quantity, price, timestamp, side)

    Returns a dictionary mapping contract_id -> net quantity.
    """
    holdings: Dict[str, int] = {}

    for trade in trades:
        contract_id = trade[1]
        quantity = int(trade[2])
        side = str(trade[5]).upper()

        if contract_id not in holdings:
            holdings[contract_id] = 0

        if side == "BUY":
            holdings[contract_id] += quantity
        elif side == "SELL":
            holdings[contract_id] -= quantity
        else:
            raise ValueError(f"Unknown trade side: {side}")

    return holdings




from db_init import get_contract


def attach_contract_details(holdings: Dict[str, int]) -> Dict[str, Dict[str, Any]]:
    """
    Attach contract metadata from the CONTRACT table to holdings.

    holdings: dict mapping contract_id -> net quantity

    Returns a dict mapping contract_id -> enriched position dict containing:
    - contract_id
    - ticker
    - expiry
    - strike
    - type
    - net_quantity
    """
    if holdings is None:
        raise ValueError("holdings must not be None")

    enriched: Dict[str, Dict[str, Any]] = {}

    for contract_id, net_qty in holdings.items():
        # get_contract returns:
        # (contract_id, ticker, expiry, strike, type) or None if not found
        row = get_contract(contract_id)

        if row is None:
            raise ValueError(f"Contract not found in CONTRACT table: {contract_id}")

        enriched[contract_id] = {
            "contract_id": row[0],
            "ticker": row[1],
            "expiry": row[2],
            "strike": row[3],
            "type": row[4],
            "net_quantity": int(net_qty),
        }

    return enriched





def get_position_metrics(trades: List[Tuple]) -> Dict[str, Any]:
    """
    Compute position metrics from a list of trade rows for a single contract.

    Each trade row is expected in the format:
    (trade_id, contract_id, quantity, price, timestamp, side)

    Cost basis rule used:
    - average_cost is computed from BUY trades only
    - SELL trades reduce net_quantity but do not change average_cost
    - if net_quantity becomes 0, average_cost is set to None

    Returns a dict containing:
    - contract_id
    - net_quantity
    - total_bought_quantity
    - average_cost
    """
    if trades is None:
        raise ValueError("trades must not be None")
    if len(trades) == 0:
        raise ValueError("trades must contain at least one trade row")

    contract_id = trades[0][1]

    net_qty = 0
    total_bought_qty = 0
    total_bought_cost = 0.0

    for row in trades:
        row_contract_id = row[1]
        if row_contract_id != contract_id:
            raise ValueError("All trades must be for the same contract_id")

        qty = int(row[2])
        price = float(row[3])
        side = str(row[5]).upper()

        if qty <= 0:
            raise ValueError("Trade quantity must be > 0")
        if price <= 0:
            raise ValueError("Trade price must be > 0")

        if side == "BUY":
            net_qty += qty
            total_bought_qty += qty
            total_bought_cost += qty * price
        elif side == "SELL":
            net_qty -= qty
        else:
            raise ValueError(f"Unknown trade side: {side}")

    if net_qty < 0:
        raise ValueError("Net quantity became negative - trade history is inconsistent")

    if net_qty == 0:
        avg_cost = None
    else:
        if total_bought_qty == 0:
            raise ValueError("Position has net quantity but no BUY trades - inconsistent state")
        avg_cost = total_bought_cost / total_bought_qty

    return {
        "contract_id": contract_id,
        "net_quantity": net_qty,
        "total_bought_quantity": total_bought_qty,
        "average_cost": avg_cost,
    }





from portfolio import derive_holdings_from_trade, attach_contract_details, get_position_metrics


def _position_key(pos: Dict[str, Any]) -> tuple:
    """
    Build a sortable key for a position dict.
    Sorting priority:
    1) ticker (A-Z)
    2) expiry (ISO date string, so lexical order matches chronological order)
    3) strike (numeric)
    4) type (C before P )
    """
    ticker = str(pos["ticker"])
    expiry = str(pos["expiry"])
    strike = float(pos["strike"])
    opt_type = str(pos["type"])
    return (ticker, expiry, strike, opt_type)


def _insertion_sort_positions(positions: List[Dict[str, Any]]) -> None:
    """
    Sort positions in-place using insertion sort and _position_key.

    """
    for i in range(1, len(positions)):
        current = positions[i]
        current_key = _position_key(current)

        j = i - 1

    
        while j >= 0 and _position_key(positions[j]) > current_key:
            positions[j + 1] = positions[j]
            j -= 1

        positions[j + 1] = current



def build_portfolio_view(all_trades: List[Tuple]) -> List[Dict[str, Any]]:
    """
    Build a complete portfolio view from a list of trade rows across all contracts.

    Each trade row is expected in the format:
    (trade_id, contract_id, quantity, price, timestamp, side)

    Returns a list of position dicts. Each dict contains:
    - contract_id, ticker, expiry, strike, type
    - net_quantity
    - total_bought_quantity
    - average_cost
    """
    if all_trades is None:
        raise ValueError("all_trades must not be None")

    if len(all_trades) == 0:
        return []

    # 1) Derive net holdings across all contracts from trade history
    holdings = derive_holdings_from_trade(all_trades)

    # 2) Remove flat positions (net quantity == 0) from the view
    nonzero_holdings: Dict[str, int] = {}
    for contract_id, net_qty in holdings.items():
        if int(net_qty) > 0:
            nonzero_holdings[contract_id] = int(net_qty)

    if len(nonzero_holdings) == 0:
        return []

    # 3) Attach contract metadata from CONTRACT table
    enriched = attach_contract_details(nonzero_holdings)

    # 4) Compute per-contract position metrics using only the trades for that contract
    positions: List[Dict[str, Any]] = []
    for contract_id in enriched.keys():
        contract_trades = [t for t in all_trades if t[1] == contract_id]
        metrics = get_position_metrics(contract_trades)

        position = {
            "contract_id": enriched[contract_id]["contract_id"],
            "ticker": enriched[contract_id]["ticker"],
            "expiry": enriched[contract_id]["expiry"],
            "strike": enriched[contract_id]["strike"],
            "type": enriched[contract_id]["type"],
            "net_quantity": metrics["net_quantity"],
            "total_bought_quantity": metrics["total_bought_quantity"],
            "average_cost": metrics["average_cost"],
        }
        positions.append(position)

   
    _insertion_sort_positions(positions)


    return positions




def unrealised_pl_for_contract(position: Dict[str, Any], current_price: float, contract_multiplier: int = 100) -> float:
    """
    Compute unrealised P/L for a single position.

    position must contain:
    - net_quantity
    - average_cost

    current_price is the current tradable premium (per share).
    contract_multiplier converts per-share premium to per-contract value.

    Returns unrealised P/L as a float.
    """
    if position is None:
        raise ValueError("position must not be None")

    if "net_quantity" not in position:
        raise ValueError("position must include 'net_quantity'")
    if "average_cost" not in position:
        raise ValueError("position must include 'average_cost'")

    net_qty = int(position["net_quantity"])
    avg_cost = position["average_cost"]

    # Flat positions have no unrealised P/L.
    if net_qty == 0:
        return 0.0

    if avg_cost is None:
        raise ValueError("average_cost is None for a non-flat position")

    try:
        cp = float(current_price)
    except (TypeError, ValueError):
        raise ValueError("current_price must be a number")

    if cp != cp:
        raise ValueError("current_price must not be NaN")
    if cp <= 0:
        raise ValueError("current_price must be > 0")

    ac = float(avg_cost)
    if ac <= 0:
        raise ValueError("average_cost must be > 0")

    pl_per_share = cp - ac
    pl_total = pl_per_share * net_qty * int(contract_multiplier)

    return float(pl_total)

# portfolio.py



from portfolio import build_portfolio_view, unrealised_pl_for_contract


def build_portfolio_view_with_pl(
    all_trades: List[Tuple],
    current_prices: Dict[str, float],
    contract_multiplier: int = 100
) -> List[Dict[str, Any]]:
    """
    Extend the portfolio view by attaching unrealised P/L to each position.

    all_trades: list of trade rows across all contracts
    current_prices: dict mapping contract_id -> current premium (per share)
    contract_multiplier: scaling factor for per-contract value

    Returns the same list of position dicts as build_portfolio_view, plus:
    - current_price
    - unrealised_pl
    """
    if current_prices is None:
        raise ValueError("current_prices must not be None")

    positions = build_portfolio_view(all_trades)

    for pos in positions:
        cid = pos["contract_id"]

        if cid not in current_prices:
            raise ValueError(f"Missing current price for contract_id: {cid}")

        cp = current_prices[cid]
        pos["current_price"] = float(cp)
        pos["unrealised_pl"] = unrealised_pl_for_contract(pos, cp, contract_multiplier)

    return positions



import math
from datetime import datetime, timezone
from typing import Dict, Any
from market import _norm_cdf
RISK_FREE_RATE_DEFAULT = 0.05





def _norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def compute_bs_greeks_for_contract(
    contract: Dict[str, Any],
    spot: float,
    sigma: float,
    r: float = RISK_FREE_RATE_DEFAULT
) -> Dict[str, float]:
    """
    Compute Black-Scholes Greeks for a European option.

    contract must include:
    - strike (numeric)
    - expiry (YYYY-MM-DD)
    - type ('C' or 'P')

    Returns:
    - delta
    - gamma
    - vega
    - theta (per day)
    """
    if contract is None:
        raise ValueError("contract must not be None")

    for key in ("strike", "expiry", "type"):
        if key not in contract:
            raise ValueError(f"contract missing required field: {key}")

    try:
        S = float(spot)
        K = float(contract["strike"])
        vol = float(sigma)
        rate = float(r)
    except (TypeError, ValueError):
        raise ValueError("spot, strike, sigma, and r must be numeric")

    if S <= 0:
        raise ValueError("spot must be > 0")
    if K <= 0:
        raise ValueError("strike must be > 0")
    if vol <= 0:
        raise ValueError("sigma must be > 0")

    opt_type = str(contract["type"]).upper()
    if opt_type not in ("C", "P"):
        raise ValueError("option type must be 'C' or 'P'")

    expiry_str = str(contract["expiry"]).strip()
    try:
        expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        raise ValueError("expiry must be in YYYY-MM-DD format")

    now = datetime.now(timezone.utc)
    T_seconds = (expiry_dt - now).total_seconds()
    T = T_seconds / (365.0 * 24.0 * 3600.0)

    if T <= 0:
        raise ValueError("time to expiry must be positive for Greeks")

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (rate + 0.5 * vol * vol) * T) / (vol * sqrtT)
    d2 = d1 - vol * sqrtT

    Nd1 = _norm_cdf(d1)
    Nd2 = _norm_cdf(d2)
    pdf_d1 = _norm_pdf(d1)

    if opt_type == "C":
        delta = Nd1
        theta_annual = (
            -(S * pdf_d1 * vol) / (2.0 * sqrtT)
            - rate * K * math.exp(-rate * T) * Nd2
        )
    else:
        delta = Nd1 - 1.0
        theta_annual = (
            -(S * pdf_d1 * vol) / (2.0 * sqrtT)
            + rate * K * math.exp(-rate * T) * _norm_cdf(-d2)
        )

    gamma = pdf_d1 / (S * vol * sqrtT)
    vega_annual = S * pdf_d1 * sqrtT

    theta_per_day = theta_annual / 365.0

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega_annual),
        "theta": float(theta_per_day),
    }



CONTRACT_MULTIPLIER_DEFAULT = 100
RISK_FREE_RATE_DEFAULT = 0.05


def attach_greeks_to_portfolio_view(
    positions: List[Dict[str, Any]],
    spot_by_ticker: Dict[str, float],
    sigma_by_ticker: Dict[str, float],
    r: float = RISK_FREE_RATE_DEFAULT,
    contract_multiplier: int = CONTRACT_MULTIPLIER_DEFAULT
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Attach Black-Scholes Greeks to each position and return portfolio totals.
    """
    if positions is None:
        raise ValueError("positions must not be None")
    if spot_by_ticker is None:
        raise ValueError("spot_by_ticker must not be None")
    if sigma_by_ticker is None:
        raise ValueError("sigma_by_ticker must not be None")

    totals = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

    for pos in positions:
        if pos is None:
            raise ValueError("positions must not contain None")

        for k in ("ticker", "expiry", "strike", "type", "net_quantity"):
            if k not in pos:
                raise ValueError(f"position missing required field: {k}")

        ticker = str(pos["ticker"]).strip().upper()
        if ticker == "":
            raise ValueError("ticker must be non-empty")

        if ticker not in spot_by_ticker:
            raise ValueError(f"Missing spot for ticker: {ticker}")
        if ticker not in sigma_by_ticker:
            raise ValueError(f"Missing sigma for ticker: {ticker}")

        try:
            spot = float(spot_by_ticker[ticker])
            sigma = float(sigma_by_ticker[ticker])
        except (TypeError, ValueError):
            raise ValueError("spot and sigma must be numeric")

        if spot <= 0:
            raise ValueError("spot must be > 0")
        if sigma <= 0:
            raise ValueError("sigma must be > 0")

        net_qty = int(pos["net_quantity"])
        if net_qty <= 0:
            raise ValueError("net_quantity must be > 0 for Greeks attachment")

        contract = {
            "strike": pos["strike"],
            "expiry": pos["expiry"],
            "type": pos["type"],
        }

        per_share = compute_bs_greeks_for_contract(contract, spot=spot, sigma=sigma, r=r)

        scale = net_qty * int(contract_multiplier)
        per_position = {
            "delta": per_share["delta"] * scale,
            "gamma": per_share["gamma"] * scale,
            "vega": per_share["vega"] * scale,
            "theta": per_share["theta"] * scale,
        }

        pos["greeks_per_share"] = per_share
        pos["greeks_position"] = per_position

        totals["delta"] += float(per_position["delta"])
        totals["gamma"] += float(per_position["gamma"])
        totals["vega"] += float(per_position["vega"])
        totals["theta"] += float(per_position["theta"])

    return positions, totals


from market import (
    get_underlying_spot,
    compute_historical_volatility,
    bs_price_yf,
    compute_mispricing_for_contract
)
from portfolio import (
    build_portfolio_view,
    compute_bs_greeks_for_contract
)




from typing import Any, Dict, List, Tuple

import market
from portfolio import build_portfolio_view, compute_bs_greeks_for_contract




from typing import Any, Dict, List, Tuple, Optional


def build_portfolio_view_with_risk_metrics(
    all_trades: List[Tuple],
    current_prices: Dict[str, float],
    spot_by_ticker: Optional[Dict[str, float]] = None,
    sigma_by_ticker: Optional[Dict[str, float]] = None,
    r: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Build a portfolio view enriched with theoretical pricing, mispricing, and Greeks.

    Market premiums are injected via current_prices to keep portfolio logic deterministic.
    Spot and sigma can be injected for deterministic testing; otherwise they are fetched via market.py.
    """
    positions = build_portfolio_view(all_trades)
    if not positions:
        return []

    if current_prices is None:
        raise ValueError("current_prices must not be None")

    by_ticker: Dict[str, List[Dict[str, Any]]] = {}
    for pos in positions:
        by_ticker.setdefault(pos["ticker"], []).append(pos)

    for ticker, ticker_positions in by_ticker.items():
        if spot_by_ticker is not None and ticker in spot_by_ticker:
            spot = float(spot_by_ticker[ticker])
        else:
            spot = float(market.get_underlying_spot(ticker))

        if sigma_by_ticker is not None and ticker in sigma_by_ticker:
            sigma = float(sigma_by_ticker[ticker])
        else:
            sigma = float(market.compute_historical_volatility(ticker))

        for pos in ticker_positions:
            cid = pos["contract_id"]
            if cid not in current_prices:
                raise ValueError(f"Missing current price for contract_id: {cid}")

            market_price = float(current_prices[cid])

            contract_view = {
                "ticker": pos["ticker"],
                "expiry": pos["expiry"],
                "strike": pos["strike"],
                "type": pos["type"],
            }

            theoretical = market.bs_price_yf(contract_view, spot=spot, sigma=sigma, r=r)

            diff = market_price - theoretical
            pct = diff / theoretical if theoretical > 0 else 0.0

            greeks = compute_bs_greeks_for_contract(contract_view, spot=spot, sigma=sigma, r=r)

            pos["spot"] = spot
            pos["volatility"] = sigma

            pos["market_price"] = market_price
            pos["theoretical_price"] = float(theoretical)
            pos["mispricing_abs"] = float(diff)
            pos["mispricing_pct"] = float(pct)

            pos["delta"] = greeks["delta"]
            pos["gamma"] = greeks["gamma"]
            pos["vega"] = greeks["vega"]
            pos["theta"] = greeks["theta"]

    return positions
