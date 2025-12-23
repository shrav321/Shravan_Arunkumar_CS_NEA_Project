# trades.py

from db_init import adjust_cash, get_cash


def add_funds(amount: float) -> float:
    """
    Add funds to the simulated cash balance.

    Returns the new balance after the funds are added.
    """
    if amount is None:
        raise ValueError("amount must not be None")

    # Convert to float early so validation is consistent for int, str-numeric, etc.
    try:
        amt = float(amount)
    except (TypeError, ValueError):
        raise ValueError("amount must be a number")

   
    if amt != amt:
        raise ValueError("amount must not be NaN")

    # Adding funds must be positive to prevent this routine being abused as a withdrawal.
    if amt <= 0:
        raise ValueError("amount must be > 0")

    # Adjust the stored cash balance.
    adjust_cash(amt)

    # Read back the stored value and return it so the caller can display it immediately.
    return get_cash()

from datetime import datetime, timezone
from typing import Any, Mapping

from db_init import get_cash, adjust_cash, insert_trade, get_trades_for_contract
from market import build_contract_id, ensure_contract_exists, get_current_option_price

# Standard equity options represent 100 shares per contract.

CONTRACT_MULTIPLIER = 100


def execute_buy_from_market(option_row: Mapping[str, Any], quantity: int) -> dict:
    """
    Execute a BUY from a market option row.

    Returns a small dict of key results for UI display and test assertions.
    """
    if option_row is None:
        raise ValueError("option_row must not be None")

    # Validate quantity early because it affects affordability and trade recording.
    try:
        qty = int(quantity)
    except (TypeError, ValueError):
        raise ValueError("quantity must be an integer")
    if qty <= 0:
        raise ValueError("quantity must be > 0")

    # Extract required contract fields from the option row.
    ticker = option_row.get("ticker", None)
    expiry = option_row.get("expiry", None)
    strike = option_row.get("strike", None)
    option_type = option_row.get("type", None)

    if ticker is None or str(ticker).strip() == "":
        raise ValueError("option_row must contain a non-empty 'ticker'")
    if expiry is None or str(expiry).strip() == "":
        raise ValueError("option_row must contain a non-empty 'expiry'")
    if strike is None:
        raise ValueError("option_row must contain 'strike'")
    if option_type is None or str(option_type).strip() == "":
        raise ValueError("option_row must contain a non-empty 'type'")

    # Build the canonical contract id and ensure CONTRACT row exists.
    contract_id = build_contract_id(str(ticker), str(expiry), float(strike), str(option_type))
    ensure_contract_exists(contract_id, str(ticker), str(expiry), float(strike), str(option_type))

    # Determine current tradable option premium.
    price = get_current_option_price(option_row)

    # Total cash impact for a buy.
    total_cost = price * qty * CONTRACT_MULTIPLIER

    # Affordability check must happen before writing any state.
    cash_before = get_cash()
    if cash_before < total_cost:
        raise ValueError("Insufficient funds to execute buy")

    # Persist the trade and cash update.
    timestamp = datetime.now(timezone.utc).isoformat()
    insert_trade(contract_id, qty, price, timestamp, "BUY")
    adjust_cash(-total_cost)

    cash_after = get_cash()

    return {
        "contract_id": contract_id,
        "side": "BUY",
        "quantity": qty,
        "price": price,
        "multiplier": CONTRACT_MULTIPLIER,
        "total_cost": total_cost,
        "cash_before": cash_before,
        "cash_after": cash_after,
        "timestamp": timestamp,
    }


from portfolio import derive_holdings_from_trade

def execute_sell_from_portfolio(contract_id: str, quantity: int, option_row: Mapping[str, Any]) -> dict:
    """
    Execute a SELL for an existing position.

    contract_id: canonical id of the contract to sell
    quantity: number of contracts to sell (must be positive)
    option_row: mapping containing pricing fields such as ask and lastPrice

    Returns a dict of key results for UI display and test assertions.
    """
    if contract_id is None or str(contract_id).strip() == "":
        raise ValueError("contract_id must be a non-empty string")
    cid = str(contract_id).strip()

    if option_row is None:
        raise ValueError("option_row must not be None")

    try:
        qty = int(quantity)
    except (TypeError, ValueError):
        raise ValueError("quantity must be an integer")
    if qty <= 0:
        raise ValueError("quantity must be > 0")

    # Retrieve trade history and derive current holdings for this contract
    trades = get_trades_for_contract(cid)
    holdings = derive_holdings_from_trade(trades)
    net_qty = holdings.get(cid, 0)

    if net_qty <= 0:
        raise ValueError("No position available to sell for this contract")
    if qty > net_qty:
        raise ValueError("Cannot sell more contracts than currently held")

    # Determine tradable price (same pricing rule used by buys)
    price = get_current_option_price(option_row)

    proceeds = price * qty * CONTRACT_MULTIPLIER

    cash_before = get_cash()

    # Write the sell trade event, then update cash
    timestamp = datetime.now(timezone.utc).isoformat()
    insert_trade(cid, qty, price, timestamp, "SELL")
    adjust_cash(proceeds)

    cash_after = get_cash()

    return {
        "contract_id": cid,
        "side": "SELL",
        "quantity": qty,
        "price": price,
        "multiplier": CONTRACT_MULTIPLIER,
        "proceeds": proceeds,
        "cash_before": cash_before,
        "cash_after": cash_after,
        "net_quantity_before": net_qty,
        "timestamp": timestamp,
    }

# trades.py

from datetime import datetime, timezone
from typing import Dict, Any

from db_init import get_trades_for_contract, insert_trade, adjust_cash
from portfolio import derive_holdings_from_trade


CONTRACT_MULTIPLIER = 100


def execute_exercise_from_portfolio(
    contract: Dict[str, Any],
    underlying_spot: float,
    current_date: str
) -> Dict[str, Any]:
    """
    Exercise an option position if it is eligible.

    contract must contain:
    - contract_id
    - expiry (ISO date string)
    - strike
    - type ('C' or 'P')

    underlying_spot is the current underlying price.
    current_date is an ISO date string used to validate expiry.

    Returns a dict describing the exercise result.
    """
    if contract is None:
        raise ValueError("contract must not be None")

    required = ["contract_id", "expiry", "strike", "type"]
    for key in required:
        if key not in contract:
            raise ValueError(f"contract missing required field: {key}")

    contract_id = str(contract["contract_id"])
    expiry = str(contract["expiry"])
    strike = float(contract["strike"])
    opt_type = str(contract["type"]).upper()

    if opt_type not in ("C", "P"):
        raise ValueError("option type must be 'C' or 'P'")

    try:
        spot = float(underlying_spot)
    except (TypeError, ValueError):
        raise ValueError("underlying_spot must be numeric")

    if spot <= 0:
        raise ValueError("underlying_spot must be > 0")

    if current_date < expiry:
        raise ValueError("option has not yet reached expiry")

    trades = get_trades_for_contract(contract_id)
    holdings = derive_holdings_from_trade(trades)
    net_qty = holdings.get(contract_id, 0)

    if net_qty <= 0:
        raise ValueError("no open position to exercise")

    if opt_type == "C":
        intrinsic = max(spot - strike, 0.0)
    else:
        intrinsic = max(strike - spot, 0.0)

    if intrinsic == 0.0:
        raise ValueError("option is not in the money and cannot be exercised")

    payoff = intrinsic * net_qty * CONTRACT_MULTIPLIER

    timestamp = datetime.now(timezone.utc).isoformat()

    insert_trade(contract_id, net_qty, intrinsic, timestamp, "SELL")
    adjust_cash(payoff)

    return {
        "contract_id": contract_id,
        "net_quantity_exercised": net_qty,
        "intrinsic_value": intrinsic,
        "payoff": payoff,
        "timestamp": timestamp,
    }

# trades.py

from datetime import datetime, timezone
from typing import Dict, Any

from db_init import get_trades_for_contract, insert_trade
from portfolio import derive_holdings_from_trade


def execute_expire_worthless_from_portfolio(
    contract: Dict[str, Any],
    underlying_spot: float,
    current_date: str
) -> Dict[str, Any]:
    """
    Expire an option position worthless at or after expiry.

    The position is closed by inserting a terminal SELL trade
    with price = 0.0 and no cash adjustment.
    """
    if contract is None:
        raise ValueError("contract must not be None")

    required = ["contract_id", "expiry", "strike", "type"]
    for key in required:
        if key not in contract:
            raise ValueError(f"contract missing required field: {key}")

    contract_id = str(contract["contract_id"]).strip()
    expiry = str(contract["expiry"]).strip()
    strike = float(contract["strike"])
    opt_type = str(contract["type"]).upper()

    if contract_id == "":
        raise ValueError("contract_id must be non-empty")
    if opt_type not in ("C", "P"):
        raise ValueError("option type must be 'C' or 'P'")
    if strike <= 0:
        raise ValueError("strike must be > 0")

    try:
        spot = float(underlying_spot)
    except (TypeError, ValueError):
        raise ValueError("underlying_spot must be numeric")
    if spot <= 0:
        raise ValueError("underlying_spot must be > 0")

    # Expiry eligibility
    if current_date < expiry:
        raise ValueError("option has not yet reached expiry")

    # Reconstruct position
    trades = get_trades_for_contract(contract_id)
    holdings = derive_holdings_from_trade(trades)
    net_qty = int(holdings.get(contract_id, 0))

    if net_qty <= 0:
        raise ValueError("no open position to expire")

    # Check intrinsic value is zero
    if opt_type == "C":
        intrinsic = max(spot - strike, 0.0)
    else:
        intrinsic = max(strike - spot, 0.0)

    if intrinsic != 0.0:
        raise ValueError("option is in the money and should be exercised instead")

    timestamp = datetime.now(timezone.utc).isoformat()

    # Close the position with a terminal SELL at zero value
    insert_trade(contract_id, net_qty, 0.0, timestamp, "SELL")

    return {
        "contract_id": contract_id,
        "net_quantity_expired": net_qty,
        "expired_worthless": True,
        "timestamp": timestamp,
    }
