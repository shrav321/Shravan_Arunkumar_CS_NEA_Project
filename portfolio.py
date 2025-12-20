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
