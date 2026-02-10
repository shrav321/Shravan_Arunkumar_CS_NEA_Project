# market.py

import math
import yfinance as yf
from datetime import datetime, timezone
from typing import Dict, Any, Mapping, List, Tuple, Optional, Callable
# Consolidated internal database imports
from db_init import insert_contract, get_prices_for_ticker, insert_price_row

# Global constant for annualising volatility and time-decay
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE_DEFAULT = 0.05  # 5% per annum baseline


def build_contract_id(ticker: str, expiry_iso: str, strike: float, option_type: str) -> str:
    """
    Constructs a unique primary key for the CONTRACT table.
    Format: TICKER_YYYY-MM-DD_STRIKE.CC_SIDE
    """
    ticker_clean = ticker.strip().upper()
    expiry_clean = expiry_iso.strip()

    # Normalise option type to single character "C" or "P"
    side_raw = option_type.strip().upper()
    if side_raw in ("C", "CALL", "CALLS"):
        side = "C"
    elif side_raw in ("P", "PUT", "PUTS"):
        side = "P"
    else:
        raise ValueError(f"Unsupported option type: {option_type!r}")

    # Normalise strike to 2 decimal places for consistent string matching
    strike_val = float(strike)
    strike_str = f"{strike_val:.2f}"

    return f"{ticker_clean}_{expiry_clean}_{strike_str}_{side}"


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


def get_current_option_price(option_row: Mapping[str, Any]) -> float:
    """
    Extract a tradable option price from an option-chain row.
    Preference order: 1) ask, 2) lastPrice.
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
        pass

    raise ValueError("No usable option price found (ask and lastPrice missing or non-positive).")


def get_underlying_spot(ticker: str) -> float:
    """
    Fetch the current spot price of the underlying asset via yfinance.
    """
    if ticker is None or str(ticker).strip() == "":
        raise ValueError("ticker must be a non-empty string")

    ticker_clean = ticker.strip().upper()

    try:
        yf_ticker = yf.Ticker(ticker_clean)

        # Attempt to read fast_info (faster attribute access)
        fast_info = getattr(yf_ticker, "fast_info", None)
        if fast_info is not None:
            last_price = fast_info.get("last_price", None)
            if last_price is not None:
                last_f = float(last_price)
                if last_f > 0:
                    return last_f

        # Fallback to general info dictionary
        info = yf_ticker.info
        price = info.get("regularMarketPrice", None)
        if price is not None:
            price_f = float(price)
            if price_f > 0:
                return price_f

    except (TypeError, ValueError, KeyError):
        pass

    raise ValueError(f"No usable spot price found for ticker {ticker_clean}")


def compute_historical_volatility(ticker: str) -> float:
    """
    Compute annualised historical volatility for a ticker using daily log returns.
    Uses PRICE table as memoisation (caching).
    """
    if ticker is None or str(ticker).strip() == "":
        raise ValueError("ticker must be a non-empty string")

    symbol = str(ticker).strip().upper()

    # Attempt to retrieve cached prices from local database
    prices: List[Tuple[str, float]] = get_prices_for_ticker(symbol)

    # Fetch from API if cache is insufficient (less than 2 data points)
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

    # Calculate log returns: ln(P_t / P_{t-1})
    log_returns = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        curr = closes[i]

        if prev <= 0 or curr <= 0:
            raise ValueError("price data must be positive")

        log_returns.append(math.log(curr / prev))

    mean = sum(log_returns) / len(log_returns)

    # Calculate daily variance and standard deviation
    variance = 0.0
    for r in log_returns:
        variance += (r - mean) ** 2

    variance /= len(log_returns)
    daily_std = math.sqrt(variance)

    # Annualise using the square root of time (sqrt(252))
    annualised_vol = daily_std * math.sqrt(TRADING_DAYS_PER_YEAR)

    return annualised_vol


def _norm_cdf(x: float) -> float:
    """
    Standard normal cumulative distribution function (CDF)
    using the Abramowitz-Stegun rational approximation for math efficiency.
    """
    a1 = 0.319381530
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429
    p  = 0.2316419

    sign = 1
    if x < 0:
        sign = -1
    x = abs(x)

    t = 1.0 / (1.0 + p * x)
    pdf = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    poly = a1*t + a2*t*t + a3*t*t*t + a4*t*t*t*t + a5*t*t*t*t*t

    cdf = 1.0 - pdf * poly

    return cdf if sign == 1 else 1.0 - cdf


def bs_price_yf(
    option_row: Dict[str, Any],
    spot: float,
    sigma: float,
    r: float = RISK_FREE_RATE_DEFAULT
) -> float:
    """
    Calculates theoretical option value using the Black-Scholes model.
    """
    if option_row is None:
        raise ValueError("option_row must not be None")

    for key in ("strike", "expiry", "type"):
        if key not in option_row:
            raise ValueError(f"option_row missing required field: {key}")

    try:
        S = float(spot)
        K = float(option_row["strike"])
        vol = float(sigma)
        rate = float(r)
    except (TypeError, ValueError):
        raise ValueError("spot, strike, sigma, and r must be numeric")

    if S <= 0 or K <= 0 or vol <= 0:
        raise ValueError("spot, strike, and sigma must be positive")

    opt_type = str(option_row["type"]).upper()
    expiry_str = str(option_row["expiry"]).strip()
    
    try:
        expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        raise ValueError("expiry must be in YYYY-MM-DD format")

    # Determine time to expiry (T) as a fraction of a 365-day year
    now = datetime.now(timezone.utc)
    T_seconds = (expiry_dt - now).total_seconds()
    T = T_seconds / (365.0 * 24.0 * 3600.0)

    if T <= 0:
        raise ValueError("time to expiry must be positive for Black-Scholes pricing")

    # Black-Scholes partial derivatives (d1 and d2)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (rate + 0.5 * vol * vol) * T) / (vol * sqrtT)
    d2 = d1 - vol * sqrtT

    discK = K * math.exp(-rate * T)

    # Standard call/put payoff formulas
    if opt_type == "C":
        price = S * _norm_cdf(d1) - discK * _norm_cdf(d2)
    else:
        price = discK * _norm_cdf(-d2) - S * _norm_cdf(-d1)

    return float(max(price, 0.0))



def compute_mispricing_for_contract(
    option_row: Dict[str, Any],
    spot: float,
    sigma: float,
    r: float = RISK_FREE_RATE_DEFAULT
) -> Dict[str, float]:
    """
    Compare observed market premium against theoretical Black-Scholes price.
    """
    if option_row is None:
        raise ValueError("option_row must not be None")

    market_price = get_current_option_price(option_row)
    theoretical = bs_price_yf(option_row, spot=spot, sigma=sigma, r=r)

    diff = market_price - theoretical
    pct = diff / theoretical if theoretical > 0 else 0.0

    return {
        "market_price": float(market_price),
        "theoretical_price": float(theoretical),
        "mispricing_abs": float(diff),
        "mispricing_pct": float(pct),
    }


def _normalise_option_type(option_type: str) -> str:
    """Internal helper to standardise 'Call'/'Put' inputs."""
    t = str(option_type).strip().upper()
    if t in ("C", "CALL", "CALLS"):
        return "C"
    if t in ("P", "PUT", "PUTS"):
        return "P"
    raise ValueError("option_type must be 'C' or 'P'")


def _rows_from_chain_table(table: Any) -> List[Dict[str, Any]]:
    """Convert yfinance dataframe/table into a standard list of dictionaries."""
    if table is None:
        return []

    to_dict = getattr(table, "to_dict", None)
    if callable(to_dict):
        return list(table.to_dict("records"))

    if isinstance(table, list):
        return [dict(item) for item in table if isinstance(item, dict)]

    raise ValueError("Unsupported option chain table type")


def fetch_options_by_ticker_and_type(
    ticker: str,
    option_type: str,
    ticker_provider: Optional[Callable[[str], Any]] = None
) -> List[Dict[str, Any]]:
    """
    Fetches full option chains for all available expiries for a given ticker.
    """
    if ticker is None or str(ticker).strip() == "":
        raise ValueError("ticker must be a non-empty string")

    symbol = str(ticker).strip().upper()
    opt_type = _normalise_option_type(option_type)

    provider = ticker_provider if ticker_provider is not None else yf.Ticker
    tk = provider(symbol)

    expiries = getattr(tk, "options", None)
    if expiries is None or len(expiries) == 0:
        raise ValueError("no option expiries available for ticker")

    results: List[Dict[str, Any]] = []

    # Iterate through all expiry dates and flatten the results
    for exp in expiries:
        chain = tk.option_chain(exp)
        table = chain.calls if opt_type == "C" else chain.puts
        rows = _rows_from_chain_table(table)

        for row in rows:
            normalised = dict(row)
            normalised["ticker"] = symbol
            normalised["expiry"] = str(exp)
            normalised["type"] = opt_type
            results.append(normalised)

    return results


def get_live_option_premium_for_contract(
    ticker: str,
    expiry: str,
    strike: float,
    option_type: str
) -> float:
    """
    Fetch a single specific option's tradable premium from the live market.
    """
    if not (ticker and expiry and option_type):
        raise ValueError("All contract identification fields must be non-empty")

    tkr = str(ticker).strip().upper()
    exp = str(expiry).strip()
    typ = _normalise_option_type(option_type)

    try:
        k = float(strike)
    except (TypeError, ValueError):
        raise ValueError("strike must be numeric")

    yf_tkr = yf.Ticker(tkr)
    chain = yf_tkr.option_chain(exp)
    df = chain.calls if typ == "C" else chain.puts

    # Filter dataframe for the specific strike price
    rows = df[df["strike"].astype(float) == float(k)]
    if rows.empty:
        raise ValueError("Contract not found in chain for given expiry and strike")

    row = rows.iloc[0].to_dict()

    # Reconstruct row for price extraction
    option_row: Dict[str, Any] = {
        "ticker": tkr,
        "expiry": exp,
        "strike": float(k),
        "type": typ,
        "ask": row.get("ask", None),
        "lastPrice": row.get("lastPrice", None),
    }

    return float(get_current_option_price(option_row))