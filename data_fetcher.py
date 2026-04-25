from datetime import datetime, timedelta
from typing import List, Tuple

import pandas as pd
import yfinance as yf

from utils import get_logger


logger = get_logger(__name__)


def default_tickers_for_geography(geography: str) -> List[str]:
    """
    Return a sensible default universe of large-cap stocks for the given geography.

    - For 'india', we default to a basket of liquid NIFTY 50 large caps.
    - For 'us', we default to a few mega-cap US equities and the S&P 500 ETF.
    - For 'global', we mix broad ETFs.
    """
    if geography == "india":
        # NSE tickers must be suffixed with '.NS' for yfinance.
        return [
            "RELIANCE.NS",
            "TCS.NS",
            "HDFCBANK.NS",
            "ICICIBANK.NS",
            "INFY.NS",
            "ITC.NS",
            "LT.NS",
            "SBIN.NS",
        ]
    if geography == "us":
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "BRK-B", "SPY"]
    # Basic global mix of ETFs; adjust as desired.
    return ["SPY", "EFA", "EEM", "AGG"]


def _resolve_lookback(duration_years: float) -> str:
    """
    Convert the investment horizon into a yfinance-compatible lookback string.

    We keep this intentionally simple and conservative:
    - For horizons <= 1 year, request '1y' of data.
    - For medium horizons, request '3y'.
    - For long horizons, request '5y'.
    """
    if duration_years <= 1:
        return "1y"
    if duration_years <= 3:
        return "3y"
    return "5y"


def fetch_price_history(
    tickers: List[str], duration_years: float
) -> Tuple[pd.DataFrame, str]:
    """
    Download historical adjusted close prices for the given tickers using yfinance.

    Returns:
        prices: DataFrame indexed by date with one column per ticker.
        lookback: The actual lookback period string used for fetching.

    Raises:
        ValueError: if data could not be fetched or is insufficient.
    """
    if not tickers:
        raise ValueError("Ticker universe is empty; cannot fetch price history.")

    lookback = _resolve_lookback(duration_years)
    logger.info(
        "Fetching historical prices for %d tickers over %s.", len(tickers), lookback
    )

    # yfinance returns a DataFrame with multiple columns (Open, High, Low, Close, etc.).
    # We request only adjusted close prices for simplicity and robustness.
    data = yf.download(
        tickers, period=lookback, interval="1d", auto_adjust=True, progress=False
    )

    # If we fetched multiple tickers, yfinance will return a MultiIndex column structure.
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        # Single ticker case.
        prices = data.copy()
        prices = prices.rename(columns={"Close": tickers[0]}) if "Close" in prices else prices

    # Drop any columns that are completely NaN (failed downloads) and forward-fill small gaps.
    prices = prices.dropna(axis=1, how="all")
    prices = prices.ffill().bfill()

    if prices.empty or prices.shape[1] < 2:
        raise ValueError(
            "Insufficient historical data fetched for the selected ticker universe."
        )

    logger.info(
        "Fetched price history with %d records for %d tickers.",
        prices.shape[0],
        prices.shape[1],
    )

    return prices, lookback

