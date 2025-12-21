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
