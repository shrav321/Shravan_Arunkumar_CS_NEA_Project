# market.py

def build_contract_id(ticker: str, expiry_iso: str, strike: float, option_type: str) -> str:
    """
    Build the canonical contract_id for an option.

    Format: TICKER_YYYY-MM-DD_STRIKE.CC_SIDE
            e.g. "AAPL_2025-12-19_150.00_C"
    """
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
