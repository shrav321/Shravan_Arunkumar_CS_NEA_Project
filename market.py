# market.py

def build_contract_id(ticker: str, expiry_iso: str, strike: float, option_type: str) -> str:
 
    #Format: TICKER_YYYY-MM-DD_STRIKE.CC_SIDE
    # Normalise ticker
    ticker_clean = ticker.strip().upper()

    # Assume expiry already supplied in ISO "YYYY-MM-DD"
    expiry_clean = expiry_iso.strip()

    # Normalise option type to single character "C" or "P"
    side_raw = option_type.strip().upper()
    if side_raw in ("C", "CALL", "CALLS"):
        side = "C"
    elif side_raw in ("P", "PUT", "PUTS"):
        side = "P"
    else:
        raise ValueError(f"Unsupported option type: {option_type!r}")

    # Normalise strike to 2 decimal places
    strike_val = float(strike)
    strike_str = f"{strike_val:.2f}"

    return f"{ticker_clean}_{expiry_clean}_{strike_str}_{side}"

# market.py

# market.py

from db_init import insert_contract


def ensure_contract_exists(contract_id: str, ticker: str, expiry: str, strike: float, option_type: str) -> None:
    """
    Ensures a CONTRACT row exists for the given contract_id.
    Intended to be idempotent via INSERT OR IGNORE in db_init.insert_contract.
    """
    if contract_id is None or str(contract_id).strip() == "":
        raise ValueError("contract_id must be a non-empty string")

    ticker_clean = ticker.strip().upper()
    if ticker_clean == "":
        raise ValueError("ticker must be a non-empty string")

    expiry_clean = expiry.strip()
    if expiry_clean == "":
        raise ValueError("expiry must be a non-empty string in ISO format (YYYY-MM-DD)")

    side_raw = option_type.strip().upper()
    if side_raw in ("C", "CALL", "CALLS"):
        side = "C"
    elif side_raw in ("P", "PUT", "PUTS"):
        side = "P"
    else:
        raise ValueError(f"Unsupported option type: {option_type!r}")

    strike_val = float(strike)
    if strike_val <= 0:
        raise ValueError("strike must be > 0")

    
    insert_contract(contract_id, ticker_clean, expiry_clean, strike_val, side)

