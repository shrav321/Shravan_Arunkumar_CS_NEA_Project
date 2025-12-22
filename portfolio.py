# portfolio.py

from typing import Dict, List, Tuple


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

# portfolio.py

from typing import Dict, Tuple, Any

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

# portfolio.py



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



from typing import Any, Dict, List, Tuple

from portfolio import derive_holdings_from_trade, attach_contract_details, get_position_metrics

# portfolio.py

from typing import Any, Dict, List


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

# portfolio.py

from typing import Any, Dict


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
