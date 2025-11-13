
import yfinance as yf
import pandas as pd

def get_underlying_spot(ticker: str) -> float:
    yf_ticker: yf.Ticker = yf.Ticker(ticker)
    hist: pd.DataFrame = yf_ticker.history(period="1d")
    spot: float = hist['Close'].iloc[-1]
    return spot
