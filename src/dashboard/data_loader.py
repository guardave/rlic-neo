"""
Data Loader for RLIC Dashboard.

Implements cache-first data fetching:
1. Check if cached data exists and is fresh
2. If cache is valid, return cached data
3. If cache is invalid/missing, fetch from online sources
4. Save fetched data to cache
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Union
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Default cache expiry (in hours)
DEFAULT_CACHE_EXPIRY_HOURS = 24


# =============================================================================
# Cache Management
# =============================================================================

def get_cache_path(name: str, extension: str = "parquet") -> Path:
    """Get cache file path for a given dataset name."""
    return CACHE_DIR / f"{name}.{extension}"


def is_cache_valid(cache_path: Path, max_age_hours: int = DEFAULT_CACHE_EXPIRY_HOURS) -> bool:
    """
    Check if cache file exists and is not expired.

    Args:
        cache_path: Path to cache file
        max_age_hours: Maximum age in hours before cache is considered stale

    Returns:
        True if cache is valid, False otherwise
    """
    if not cache_path.exists():
        return False

    # Check file modification time
    mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
    age = datetime.now() - mtime

    return age < timedelta(hours=max_age_hours)


def load_from_cache(cache_path: Path) -> Optional[pd.DataFrame]:
    """Load data from cache file."""
    if cache_path.exists():
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_path}: {e}")
    return None


def save_to_cache(df: pd.DataFrame, cache_path: Path) -> None:
    """Save data to cache file."""
    try:
        df.to_parquet(cache_path)
        logger.info(f"Saved cache: {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to save cache {cache_path}: {e}")


# =============================================================================
# FRED Data Fetching
# =============================================================================

# FRED series mappings
FRED_SERIES = {
    # Growth indicators
    "orders_inv_ratio": "AMTMNO/AMTMTI",  # Computed from Orders / Inventories
    "new_orders": "AMTMNO",               # Manufacturers New Orders
    "inventories": "AMTMTI",              # Manufacturers Inventories
    "industrial_prod": "INDPRO",          # Industrial Production Index
    "capacity_util": "TCU",               # Capacity Utilization

    # Inflation indicators
    "ppi_all": "PPIACO",                  # PPI All Commodities
    "cpi_all": "CPIAUCSL",                # CPI All Urban Consumers
    "cpi_core": "CPILFESL",               # Core CPI

    # Labor market
    "unemployment": "UNRATE",             # Unemployment Rate
    "initial_claims": "ICSA",             # Initial Jobless Claims

    # Retail
    "retail_inv_sales": "RETAILIRSA",     # Retail Inventories/Sales Ratio

    # Interest rates
    "fed_funds": "FEDFUNDS",              # Federal Funds Rate
    "treasury_10y": "GS10",               # 10-Year Treasury
    "spread_10y2y": "T10Y2Y",             # 10Y-2Y Spread
    "spread_10y3m": "T10Y3M",             # 10Y-3M Spread

    # Money supply
    "m2": "M2SL",                         # M2 Money Supply

    # Leading indicators
    "lei": "USSLIND",                     # Leading Economic Index
    "cfnai": "CFNAI",                     # Chicago Fed National Activity Index
}


def fetch_fred_series(series_id: str, start_date: str = "1990-01-01",
                      api_key: Optional[str] = None) -> pd.Series:
    """
    Fetch a single FRED series using pandas-datareader.

    Args:
        series_id: FRED series ID
        start_date: Start date string
        api_key: FRED API key (optional, uses env var if not provided)

    Returns:
        Series with FRED data
    """
    try:
        import pandas_datareader.data as web

        end_date = datetime.now().strftime("%Y-%m-%d")
        data = web.DataReader(series_id, 'fred', start_date, end_date)

        if isinstance(data, pd.DataFrame):
            return data.iloc[:, 0]
        return data

    except Exception as e:
        logger.error(f"Failed to fetch FRED series {series_id}: {e}")
        return pd.Series()


def fetch_fred_data(series_dict: Dict[str, str] = None,
                    start_date: str = "1990-01-01",
                    cache_name: str = "fred_indicators",
                    max_cache_age: int = 24) -> pd.DataFrame:
    """
    Fetch multiple FRED series with caching.

    Args:
        series_dict: Dict mapping names to FRED series IDs
        start_date: Start date
        cache_name: Name for cache file
        max_cache_age: Cache expiry in hours

    Returns:
        DataFrame with all series
    """
    if series_dict is None:
        series_dict = FRED_SERIES

    cache_path = get_cache_path(cache_name)

    # Check cache first
    if is_cache_valid(cache_path, max_cache_age):
        logger.info(f"Loading FRED data from cache: {cache_path}")
        cached = load_from_cache(cache_path)
        if cached is not None:
            return cached

    # Fetch fresh data
    logger.info("Fetching fresh FRED data...")
    results = {}

    for name, series_id in series_dict.items():
        # Handle computed series (e.g., ratio)
        if "/" in series_id:
            parts = series_id.split("/")
            num = fetch_fred_series(parts[0], start_date)
            den = fetch_fred_series(parts[1], start_date)
            if not num.empty and not den.empty:
                # Align and compute ratio
                aligned = pd.concat([num, den], axis=1).dropna()
                results[name] = aligned.iloc[:, 0] / aligned.iloc[:, 1]
        else:
            data = fetch_fred_series(series_id, start_date)
            if not data.empty:
                results[name] = data

    if not results:
        logger.error("Failed to fetch any FRED data")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Save to cache
    save_to_cache(df, cache_path)

    return df


# =============================================================================
# Yahoo Finance Data Fetching
# =============================================================================

# Ticker mappings
YAHOO_TICKERS = {
    # Major indices
    "SPY": "SPY",
    "QQQ": "QQQ",

    # Sector ETFs
    "XLK": "XLK",   # Technology
    "XLF": "XLF",   # Financials
    "XLE": "XLE",   # Energy
    "XLV": "XLV",   # Healthcare
    "XLY": "XLY",   # Consumer Discretionary
    "XLP": "XLP",   # Consumer Staples
    "XLI": "XLI",   # Industrials
    "XLB": "XLB",   # Materials
    "XLU": "XLU",   # Utilities
    "XLRE": "XLRE", # Real Estate
    "XLC": "XLC",   # Communication Services

    # Other
    "TLT": "TLT",   # Long-term Treasuries
    "GLD": "GLD",   # Gold
    "VIX": "^VIX",  # Volatility Index
}


def fetch_yahoo_prices(tickers: Union[str, List[str]],
                       start_date: str = "1990-01-01",
                       end_date: str = None) -> pd.DataFrame:
    """
    Fetch price data from Yahoo Finance.

    Args:
        tickers: Single ticker or list of tickers
        start_date: Start date
        end_date: End date (defaults to today)

    Returns:
        DataFrame with adjusted close prices
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    if isinstance(tickers, str):
        tickers = [tickers]

    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)

        if data.empty:
            return pd.DataFrame()

        # Extract Adjusted Close
        if isinstance(data.columns, pd.MultiIndex):
            if 'Adj Close' in data.columns.get_level_values(0):
                prices = data['Adj Close']
            else:
                prices = data['Close']
        else:
            # Single ticker
            if 'Adj Close' in data.columns:
                prices = data[['Adj Close']]
                prices.columns = tickers
            else:
                prices = data[['Close']]
                prices.columns = tickers

        return prices

    except Exception as e:
        logger.error(f"Failed to fetch Yahoo data for {tickers}: {e}")
        return pd.DataFrame()


def fetch_sector_prices(start_date: str = "1999-01-01",
                       cache_name: str = "sector_prices",
                       max_cache_age: int = 24) -> pd.DataFrame:
    """
    Fetch sector ETF prices with caching.

    Args:
        start_date: Start date
        cache_name: Name for cache file
        max_cache_age: Cache expiry in hours

    Returns:
        DataFrame with sector prices
    """
    cache_path = get_cache_path(cache_name)

    # Check cache first
    if is_cache_valid(cache_path, max_cache_age):
        logger.info(f"Loading sector prices from cache: {cache_path}")
        cached = load_from_cache(cache_path)
        if cached is not None:
            return cached

    # Fetch fresh data
    logger.info("Fetching fresh sector price data...")
    sector_tickers = [v for k, v in YAHOO_TICKERS.items() if k.startswith("XL")]

    prices = fetch_yahoo_prices(sector_tickers, start_date)

    if not prices.empty:
        save_to_cache(prices, cache_path)

    return prices


def fetch_spy_prices(start_date: str = "1993-01-01",
                    cache_name: str = "spy_prices",
                    max_cache_age: int = 24) -> pd.DataFrame:
    """
    Fetch SPY price data with caching.

    Args:
        start_date: Start date
        cache_name: Name for cache file
        max_cache_age: Cache expiry in hours

    Returns:
        DataFrame with SPY prices
    """
    cache_path = get_cache_path(cache_name)

    # Check cache first
    if is_cache_valid(cache_path, max_cache_age):
        logger.info(f"Loading SPY prices from cache: {cache_path}")
        cached = load_from_cache(cache_path)
        if cached is not None:
            return cached

    # Fetch fresh data
    logger.info("Fetching fresh SPY price data...")
    prices = fetch_yahoo_prices("SPY", start_date)

    if not prices.empty:
        save_to_cache(prices, cache_path)

    return prices


# =============================================================================
# Combined Data Loading
# =============================================================================

def load_indicator_with_target(indicator_name: str,
                               target_ticker: str,
                               start_date: str = "1990-01-01",
                               cache_name: Optional[str] = None,
                               max_cache_age: int = 24) -> pd.DataFrame:
    """
    Load indicator data merged with target returns.

    Args:
        indicator_name: Name of FRED indicator (key in FRED_SERIES)
        target_ticker: Yahoo ticker for target
        start_date: Start date
        cache_name: Cache file name (auto-generated if None)
        max_cache_age: Cache expiry in hours

    Returns:
        Monthly DataFrame with indicator and target returns
    """
    if cache_name is None:
        cache_name = f"{indicator_name}_{target_ticker}_analysis"

    cache_path = get_cache_path(cache_name)

    # Check cache first
    if is_cache_valid(cache_path, max_cache_age):
        logger.info(f"Loading analysis data from cache: {cache_path}")
        cached = load_from_cache(cache_path)
        if cached is not None:
            return cached

    # Fetch indicator
    if indicator_name in FRED_SERIES:
        series_dict = {indicator_name: FRED_SERIES[indicator_name]}
        indicator_df = fetch_fred_data(series_dict, start_date,
                                       cache_name=f"fred_{indicator_name}",
                                       max_cache_age=max_cache_age)
    else:
        logger.error(f"Unknown indicator: {indicator_name}")
        return pd.DataFrame()

    # Fetch target prices
    target_prices = fetch_yahoo_prices(target_ticker, start_date)
    if target_prices.empty:
        return pd.DataFrame()

    # Resample to monthly
    indicator_monthly = indicator_df.resample('ME').last()
    target_monthly = target_prices.resample('ME').last()

    # Calculate returns
    target_monthly[f"{target_ticker}_return"] = target_monthly.iloc[:, 0].pct_change()

    # Merge
    merged = pd.concat([indicator_monthly, target_monthly], axis=1).dropna()

    # Save to cache
    save_to_cache(merged, cache_path)

    return merged


def load_investment_clock_data(start_date: str = "1990-01-01",
                               cache_name: str = "investment_clock_data",
                               max_cache_age: int = 24) -> pd.DataFrame:
    """
    Load data for Investment Clock analysis.

    Includes:
    - Orders/Inventories Ratio (growth indicator)
    - PPI (inflation indicator)
    - Sector returns (FF-12 or Sector ETFs)

    Returns:
        Monthly DataFrame with indicators and sector returns
    """
    # First, try to load from existing data/ parquet files (primary source)
    existing_data_path = DATA_DIR / "monthly_with_best_phases.parquet"
    if existing_data_path.exists():
        logger.info(f"Loading Investment Clock data from: {existing_data_path}")
        data = pd.read_parquet(existing_data_path)

        # Ensure sector returns are present
        sector_prices_path = DATA_DIR / "sector_prices.parquet"
        if sector_prices_path.exists():
            sectors = pd.read_parquet(sector_prices_path)
            if hasattr(sectors.index, 'tz'):
                sectors.index = sectors.index.tz_localize(None)
            sectors_monthly = sectors.resample('ME').last()
            sector_returns = sectors_monthly.pct_change()
            sector_returns.columns = [f"{col}_return" for col in sector_returns.columns]

            # Merge with data
            data = pd.concat([data, sector_returns], axis=1, join='inner')

        return data

    # Fallback: check our cache
    cache_path = get_cache_path(cache_name)
    if is_cache_valid(cache_path, max_cache_age):
        logger.info(f"Loading Investment Clock data from cache: {cache_path}")
        cached = load_from_cache(cache_path)
        if cached is not None:
            return cached

    # Last resort: try to fetch fresh data
    logger.warning("No cached data found, attempting to fetch fresh data...")

    # Fetch growth and inflation indicators
    indicator_series = {
        "new_orders": "AMTMNO",
        "inventories": "AMTMTI",
        "ppi_all": "PPIACO"
    }
    indicators = fetch_fred_data(indicator_series, start_date,
                                 cache_name="fred_ic_indicators",
                                 max_cache_age=max_cache_age)

    if indicators.empty:
        raise ValueError("No data available. Please ensure cached parquet files exist in data/")

    # Compute Orders/Inv ratio
    if 'new_orders' in indicators.columns and 'inventories' in indicators.columns:
        indicators['orders_inv_ratio'] = indicators['new_orders'] / indicators['inventories']

    # Fetch sector data
    sectors = fetch_sector_prices(start_date="1999-01-01",
                                 max_cache_age=max_cache_age)

    # Resample and merge
    if hasattr(indicators.index, 'freq') or isinstance(indicators.index, pd.DatetimeIndex):
        indicators_monthly = indicators.resample('ME').last()
    else:
        indicators_monthly = indicators

    if hasattr(sectors.index, 'freq') or isinstance(sectors.index, pd.DatetimeIndex):
        sectors_monthly = sectors.resample('ME').last()
    else:
        sectors_monthly = sectors

    # Calculate sector returns
    sector_returns = sectors_monthly.pct_change()
    sector_returns.columns = [f"{col}_return" for col in sector_returns.columns]

    merged = pd.concat([indicators_monthly, sector_returns], axis=1)

    # Save to cache
    save_to_cache(merged, cache_path)

    return merged


# =============================================================================
# Cache Invalidation
# =============================================================================

def invalidate_cache(cache_name: str = None) -> None:
    """
    Invalidate (delete) cached data.

    Args:
        cache_name: Name of cache to invalidate. If None, invalidates all caches.
    """
    if cache_name:
        cache_path = get_cache_path(cache_name)
        if cache_path.exists():
            cache_path.unlink()
            logger.info(f"Invalidated cache: {cache_path}")
    else:
        # Invalidate all caches
        for cache_file in CACHE_DIR.glob("*.parquet"):
            cache_file.unlink()
            logger.info(f"Invalidated cache: {cache_file}")


def get_cache_info() -> pd.DataFrame:
    """
    Get information about all cached files.

    Returns:
        DataFrame with cache file info (name, size, age)
    """
    info = []
    for cache_file in CACHE_DIR.glob("*.parquet"):
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        age = datetime.now() - mtime
        size_mb = cache_file.stat().st_size / (1024 * 1024)

        info.append({
            'name': cache_file.stem,
            'size_mb': round(size_mb, 2),
            'modified': mtime.strftime("%Y-%m-%d %H:%M"),
            'age_hours': round(age.total_seconds() / 3600, 1),
            'valid': age < timedelta(hours=DEFAULT_CACHE_EXPIRY_HOURS)
        })

    return pd.DataFrame(info)


# =============================================================================
# Convenience Functions
# =============================================================================

def load_analysis_data(analysis_id: str, max_cache_age: int = 24) -> pd.DataFrame:
    """
    Load data for a specific analysis.

    Args:
        analysis_id: One of 'investment_clock', 'spy_retailirsa',
                     'spy_indpro', 'xlre_orders_inv'
        max_cache_age: Cache expiry in hours

    Returns:
        DataFrame with analysis data
    """
    # For investment_clock, always use the specialized loader that includes sector returns
    if analysis_id == 'investment_clock':
        return load_investment_clock_data(max_cache_age=max_cache_age)

    # For other analyses, check for existing parquet files in data/ directory
    existing_files = {
        'spy_retailirsa': 'spy_retail_inv_sales.parquet',
        'spy_indpro': 'spy_ip_analysis.parquet',
        'xlre_orders_inv': 'xlre_oi_analysis.parquet',
        'xlp_retailirsa': 'xlp_retail_inv_sales.parquet',
        'xly_retailirsa': 'xly_retail_inv_sales.parquet',
        'xlre_newhomesales': 'xlre_newhomesales_full.parquet',
        'xli_ism_mfg': 'xli_ism_mfg_full.parquet'
    }

    if analysis_id in existing_files:
        data_path = DATA_DIR / existing_files[analysis_id]
        if data_path.exists():
            logger.info(f"Loading {analysis_id} from existing data: {data_path}")
            data = pd.read_parquet(data_path)
            # Ensure datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'date' in data.columns:
                    data.set_index('date', inplace=True)
            return data

    # Fallback to fetching fresh data

    elif analysis_id == 'spy_retailirsa':
        return load_indicator_with_target('retail_inv_sales', 'SPY',
                                          max_cache_age=max_cache_age)

    elif analysis_id == 'spy_indpro':
        return load_indicator_with_target('industrial_prod', 'SPY',
                                          max_cache_age=max_cache_age)

    elif analysis_id == 'xlre_orders_inv':
        return load_indicator_with_target('orders_inv_ratio', 'XLRE',
                                          start_date="2015-01-01",
                                          max_cache_age=max_cache_age)

    elif analysis_id == 'xlp_retailirsa':
        return load_indicator_with_target('retail_inv_sales', 'XLP',
                                          max_cache_age=max_cache_age)

    elif analysis_id == 'xly_retailirsa':
        return load_indicator_with_target('retail_inv_sales', 'XLY',
                                          max_cache_age=max_cache_age)

    else:
        raise ValueError(f"Unknown analysis_id: {analysis_id}")


# For testing
if __name__ == "__main__":
    print("Testing data loader...")

    # Test cache info
    print("\nCache status:")
    print(get_cache_info())

    # Test loading analysis data
    print("\nLoading SPY vs RETAILIRSA data...")
    data = load_analysis_data('spy_retailirsa')
    print(f"Shape: {data.shape}")
    print(data.tail())
