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
