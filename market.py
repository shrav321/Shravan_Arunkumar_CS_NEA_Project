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

# market.py

from typing import Any, Mapping


def get_current_option_price(option_row: Mapping[str, Any]) -> float:
    """
    Extract a tradable option price from an option-chain row.

    Preference order:
    1) ask (if present and > 0)
    2) lastPrice (if present and > 0)

    Raises ValueError if no usable price can be obtained.
    """
    try:
        ask = option_row.get("ask", None)
        if ask is not None:
            ask_f = float(ask)
            if ask_f > 0:
                return ask_f

        last = option_row.get("lastPrice", None)
        if last is not None:
            last_f = float(last)
            if last_f > 0:
                return last_f

    except (TypeError, ValueError):
        # Handles missing keys, non-numeric values, or unexpected row structures
        pass

    raise ValueError("No usable option price found (ask and lastPrice missing or non-positive).")


# market.py

import yfinance as yf


def get_underlying_spot(ticker: str) -> float:
    """
    Fetch the current spot price of the underlying asset.

    Returns a positive float representing the latest tradable price.
    Raises ValueError if a usable price cannot be obtained.
    """
    if ticker is None or str(ticker).strip() == "":
        raise ValueError("ticker must be a non-empty string")

    ticker_clean = ticker.strip().upper()

    try:
        yf_ticker = yf.Ticker(ticker_clean)

        # Attempt to read fast_info first
        fast_info = getattr(yf_ticker, "fast_info", None)
        if fast_info is not None:
            last_price = fast_info.get("last_price", None)
            if last_price is not None:
                last_f = float(last_price)
                if last_f > 0:
                    return last_f

        # Fallback to info dictionary
        info = yf_ticker.info
        price = info.get("regularMarketPrice", None)
        if price is not None:
            price_f = float(price)
            if price_f > 0:
                return price_f

    except (TypeError, ValueError, KeyError):
        pass

    raise ValueError(f"No usable spot price found for ticker {ticker_clean}")

# market.py

import math
from typing import List, Tuple

from db_init import get_prices_for_ticker, insert_price_row
import yfinance as yf


TRADING_DAYS_PER_YEAR = 252


def compute_historical_volatility(ticker: str) -> float:
    """
    Compute annualised historical volatility for a ticker using daily log returns.

    Uses PRICE table as memoisation. Missing prices are fetched once from yfinance
    and inserted using INSERT OR IGNORE.

    Returns volatility as a float.
    """
    if ticker is None or str(ticker).strip() == "":
        raise ValueError("ticker must be a non-empty string")

    symbol = str(ticker).strip().upper()

    prices: List[Tuple[str, float]] = get_prices_for_ticker(symbol)

    if len(prices) < 2:
        hist = yf.Ticker(symbol).history(period="1y")

        if hist.empty:
            raise ValueError("no historical price data available")

        for date, row in hist.iterrows():
            insert_price_row(symbol, date.strftime("%Y-%m-%d"), float(row["Close"]))

        prices = get_prices_for_ticker(symbol)

    if len(prices) < 2:
        raise ValueError("insufficient data to compute volatility")

    closes = [float(p[1]) for p in prices]

    log_returns = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        curr = closes[i]

        if prev <= 0 or curr <= 0:
            raise ValueError("price data must be positive")

        log_returns.append(math.log(curr / prev))

    mean = sum(log_returns) / len(log_returns)

    variance = 0.0
    for r in log_returns:
        variance += (r - mean) ** 2

    variance /= len(log_returns)
    daily_std = math.sqrt(variance)

    annualised_vol = daily_std * math.sqrt(TRADING_DAYS_PER_YEAR)

    return annualised_vol



