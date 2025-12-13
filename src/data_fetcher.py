"""
Data fetching module for RLIC Enhancement Project.

Fetches price data from Yahoo Finance and economic data from FRED.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)


class YahooFinanceFetcher:
    """Fetch price data from Yahoo Finance."""

    # Asset tickers for Investment Clock analysis
    TICKERS = {
        # Equity indices
        "sp500": "^GSPC",
        "sp500_etf": "SPY",
        "ftse100": "^FTSE",
        "nasdaq": "^IXIC",
        "russell2000": "^RUT",

        # Bond proxies
        "treasury_10y_yield": "^TNX",
        "treasury_20y_etf": "TLT",
        "treasury_short_etf": "SHY",
        "treasury_tips_etf": "TIP",

        # Commodities
        "gold": "GC=F",
        "gold_etf": "GLD",
        "crude_oil": "CL=F",
        "oil_etf": "USO",
        "commodity_etf": "DBC",

        # Currency
        "usd_index": "DX-Y.NYB",

        # Volatility
        "vix": "^VIX",

        # Sector ETFs (for rotation analysis)
        "sector_tech": "XLK",
        "sector_financials": "XLF",
        "sector_energy": "XLE",
        "sector_healthcare": "XLV",
        "sector_consumer_disc": "XLY",
        "sector_consumer_staples": "XLP",
        "sector_industrials": "XLI",
        "sector_materials": "XLB",
        "sector_utilities": "XLU",
        "sector_realestate": "XLRE",
    }

    def __init__(self):
        self.cache_file = CACHE_DIR / "yahoo_prices.parquet"

    def fetch_single(self, ticker: str, start: str = "1990-01-01",
                     end: str = None) -> pd.DataFrame:
        """Fetch data for a single ticker."""
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Fetching {ticker} from {start} to {end}")
        try:
            data = yf.download(ticker, start=start, end=end, progress=False)
            if data.empty:
                logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()
            return data
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return pd.DataFrame()

    def fetch_all(self, start: str = "1990-01-01", end: str = None,
                  tickers: dict = None) -> dict:
        """Fetch data for all defined tickers."""
        if tickers is None:
            tickers = self.TICKERS
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        results = {}
        for name, ticker in tickers.items():
            data = self.fetch_single(ticker, start, end)
            if not data.empty:
                results[name] = data

        return results

    def fetch_adjusted_close(self, start: str = "1990-01-01",
                             end: str = None) -> pd.DataFrame:
        """Fetch adjusted close prices for all tickers as a single DataFrame."""
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        ticker_list = list(self.TICKERS.values())
        logger.info(f"Fetching {len(ticker_list)} tickers...")

        try:
            data = yf.download(ticker_list, start=start, end=end, progress=False)

            # Handle multi-level columns
            if isinstance(data.columns, pd.MultiIndex):
                # Get Adjusted Close prices
                if 'Adj Close' in data.columns.get_level_values(0):
                    prices = data['Adj Close']
                else:
                    prices = data['Close']
            else:
                prices = data

            # Rename columns to friendly names
            reverse_tickers = {v: k for k, v in self.TICKERS.items()}
            prices.columns = [reverse_tickers.get(col, col) for col in prices.columns]

            return prices

        except Exception as e:
            logger.error(f"Error fetching prices: {e}")
            return pd.DataFrame()

    def save_to_cache(self, data: pd.DataFrame):
        """Save data to parquet cache."""
        data.to_parquet(self.cache_file)
        logger.info(f"Saved to {self.cache_file}")

    def load_from_cache(self) -> pd.DataFrame:
        """Load data from parquet cache."""
        if self.cache_file.exists():
            return pd.read_parquet(self.cache_file)
        return pd.DataFrame()


class FREDFetcher:
    """Fetch economic data from FRED."""

    # FRED series for Investment Clock analysis
    SERIES = {
        # Growth indicators
        "gdp_real": "GDPC1",              # Real GDP (Quarterly)
        "gdp_growth": "A191RL1Q225SBEA",  # Real GDP growth rate (Quarterly)
        "industrial_prod": "INDPRO",       # Industrial Production Index (Monthly)
        "capacity_util": "TCU",            # Capacity Utilization (Monthly)

        # Inflation indicators
        "cpi_all": "CPIAUCSL",             # CPI All Urban Consumers (Monthly)
        "cpi_core": "CPILFESL",            # Core CPI (Monthly)
        "pce_price": "PCEPI",              # PCE Price Index (Monthly)
        "ppi_all": "PPIACO",               # PPI All Commodities (Monthly)

        # Labor market
        "unemployment": "UNRATE",           # Unemployment Rate (Monthly)
        "nonfarm_payrolls": "PAYEMS",      # Nonfarm Payrolls (Monthly)
        "initial_claims": "ICSA",          # Initial Jobless Claims (Weekly)

        # Interest rates & yield curve
        "fed_funds": "FEDFUNDS",           # Federal Funds Rate (Monthly)
        "treasury_3m": "TB3MS",            # 3-Month Treasury (Monthly)
        "treasury_2y": "GS2",              # 2-Year Treasury (Monthly)
        "treasury_10y": "GS10",            # 10-Year Treasury (Monthly)
        "treasury_10y2y_spread": "T10Y2Y", # 10Y-2Y Spread (Daily)
        "treasury_10y3m_spread": "T10Y3M", # 10Y-3M Spread (Daily)

        # Money supply
        "m2": "M2SL",                       # M2 Money Supply (Monthly)

        # Surveys & Leading indicators
        "consumer_sentiment": "UMCSENT",   # U. Michigan Consumer Sentiment (Monthly)
        "leading_index": "USSLIND",        # Leading Index (Monthly)

        # Housing
        "housing_starts": "HOUST",         # Housing Starts (Monthly)
        "building_permits": "PERMIT",      # Building Permits (Monthly)

        # Credit spreads
        "baa_spread": "BAA10Y",            # BAA Corporate Bond Spread (Daily)
        "aaa_spread": "AAA10Y",            # AAA Corporate Bond Spread (Daily)

        # Output gap (if available)
        "gdp_gap": "GDPNQDCA188",          # GDP Gap estimate
    }

    def __init__(self, api_key: str = None):
        """
        Initialize FRED fetcher.

        Args:
            api_key: FRED API key. If None, will try to read from environment
                     or .env file. Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html
        """
        self.api_key = api_key
        self.fred = None
        self.cache_file = CACHE_DIR / "fred_data.parquet"

        if api_key:
            self.fred = Fred(api_key=api_key)

    def set_api_key(self, api_key: str):
        """Set FRED API key."""
        self.api_key = api_key
        self.fred = Fred(api_key=api_key)

    def fetch_single(self, series_id: str, start: str = "1990-01-01",
                     end: str = None) -> pd.Series:
        """Fetch a single FRED series."""
        if self.fred is None:
            raise ValueError("FRED API key not set. Call set_api_key() first.")

        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Fetching FRED series {series_id}")
        try:
            data = self.fred.get_series(series_id, observation_start=start,
                                        observation_end=end)
            return data
        except Exception as e:
            logger.error(f"Error fetching {series_id}: {e}")
            return pd.Series()

    def fetch_all(self, start: str = "1990-01-01", end: str = None,
                  series: dict = None) -> pd.DataFrame:
        """Fetch all defined FRED series."""
        if self.fred is None:
            raise ValueError("FRED API key not set. Call set_api_key() first.")

        if series is None:
            series = self.SERIES
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        results = {}
        for name, series_id in series.items():
            data = self.fetch_single(series_id, start, end)
            if not data.empty:
                results[name] = data

        # Combine into DataFrame
        df = pd.DataFrame(results)
        return df

    def save_to_cache(self, data: pd.DataFrame):
        """Save data to parquet cache."""
        data.to_parquet(self.cache_file)
        logger.info(f"Saved to {self.cache_file}")

    def load_from_cache(self) -> pd.DataFrame:
        """Load data from parquet cache."""
        if self.cache_file.exists():
            return pd.read_parquet(self.cache_file)
        return pd.DataFrame()


def compute_returns(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Compute percentage returns."""
    return prices.pct_change(periods) * 100


def compute_yoy_change(data: pd.Series) -> pd.Series:
    """Compute year-over-year percentage change."""
    return data.pct_change(12) * 100  # Assuming monthly data


def compute_momentum(data: pd.Series, short_window: int = 6,
                     long_window: int = 12) -> pd.Series:
    """
    Compute momentum indicator.
    Returns 1 if above both MAs, -1 if below both, 0 otherwise.
    """
    short_ma = data.rolling(short_window).mean()
    long_ma = data.rolling(long_window).mean()

    above_both = (data > short_ma) & (data > long_ma)
    below_both = (data < short_ma) & (data < long_ma)

    momentum = pd.Series(0, index=data.index)
    momentum[above_both] = 1
    momentum[below_both] = -1

    return momentum


if __name__ == "__main__":
    # Example usage
    print("Testing Yahoo Finance fetcher...")
    yf_fetcher = YahooFinanceFetcher()

    # Fetch S&P 500 as a test
    sp500 = yf_fetcher.fetch_single("^GSPC", start="2020-01-01")
    print(f"S&P 500 data shape: {sp500.shape}")
    print(sp500.tail())

    print("\nTo use FRED data, you need an API key.")
    print("Get one free at: https://fred.stlouisfed.org/docs/api/api_key.html")
