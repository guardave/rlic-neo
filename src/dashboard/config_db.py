"""
Configuration Database for RLIC Dashboard.

SQLite-backed configuration that replaces hardcoded elif chains
for column detection, analysis metadata, and regime definitions.

Design doc: docs/design_dashboard_refactoring.md
"""

import sqlite3
import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

# Database path
PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "rlic_config.db"


# =============================================================================
# Database Connection & Schema
# =============================================================================

def get_connection() -> sqlite3.Connection:
    """Get SQLite connection with Row factory for dict-like access."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS analyses (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        icon TEXT DEFAULT 'analytics',
        short_name TEXT,
        description TEXT,
        caption TEXT,
        home_column INTEGER DEFAULT 1,
        display_order INTEGER DEFAULT 100,
        analysis_type TEXT DEFAULT 'single',
        target_ticker TEXT,
        target_return_col TEXT,
        data_file TEXT NOT NULL,
        phase_labels TEXT,
        created_at TEXT DEFAULT (datetime('now')),
        notes TEXT
    );

    CREATE TABLE IF NOT EXISTS analysis_indicators (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_id TEXT NOT NULL,
        axis TEXT NOT NULL DEFAULT 'primary',
        indicator_pattern TEXT,
        indicator_columns TEXT,
        indicator_filter TEXT,
        indicator_exclude TEXT,
        trading_columns TEXT,
        return_columns TEXT,
        return_pattern TEXT,
        price_column TEXT,
        exclude_from_detection TEXT,
        base_column TEXT,
        default_lag INTEGER DEFAULT 0,
        lag_min INTEGER DEFAULT -12,
        lag_max INTEGER DEFAULT 12,
        regime_method TEXT DEFAULT 'direction',
        regime_threshold REAL,
        regime_labels TEXT,
        regime_source_col TEXT,
        FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_analysis_indicators_analysis
        ON analysis_indicators(analysis_id, axis);
    """)

    conn.commit()
    conn.close()


# =============================================================================
# Analysis Queries
# =============================================================================

def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a sqlite3.Row to a plain dict."""
    return dict(row)


def get_all_analyses() -> List[Dict[str, Any]]:
    """Get all analyses ordered by display_order."""
    conn = get_connection()
    cursor = conn.execute(
        "SELECT * FROM analyses ORDER BY display_order, id"
    )
    results = [_row_to_dict(row) for row in cursor.fetchall()]
    conn.close()
    return results


def get_analysis_config(analysis_id: str) -> Optional[Dict[str, Any]]:
    """Get configuration for a single analysis."""
    conn = get_connection()
    cursor = conn.execute(
        "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
    )
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    return _row_to_dict(row)


def get_indicator_config(
    analysis_id: str,
    axis: str = 'primary'
) -> Optional[Dict[str, Any]]:
    """Get indicator configuration for an analysis axis."""
    conn = get_connection()
    cursor = conn.execute(
        "SELECT * FROM analysis_indicators WHERE analysis_id = ? AND axis = ?",
        (analysis_id, axis)
    )
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    result = _row_to_dict(row)
    # Parse JSON fields
    for field in ('indicator_columns', 'indicator_filter', 'indicator_exclude',
                  'trading_columns', 'return_columns', 'exclude_from_detection',
                  'regime_labels'):
        if result.get(field):
            try:
                result[field] = json.loads(result[field])
            except (json.JSONDecodeError, TypeError):
                pass
    return result


def get_all_indicator_configs(analysis_id: str) -> List[Dict[str, Any]]:
    """Get all indicator configs for an analysis (primary + secondary)."""
    conn = get_connection()
    cursor = conn.execute(
        "SELECT * FROM analysis_indicators WHERE analysis_id = ? ORDER BY axis",
        (analysis_id,)
    )
    results = []
    for row in cursor.fetchall():
        d = _row_to_dict(row)
        for field in ('indicator_columns', 'indicator_filter', 'indicator_exclude',
                      'trading_columns', 'return_columns', 'exclude_from_detection',
                      'regime_labels'):
            if d.get(field):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    pass
        results.append(d)
    conn.close()
    return results


# =============================================================================
# Column Resolution — replaces 64 elif branches
# =============================================================================

def resolve_columns(
    analysis_id: str,
    data: pd.DataFrame,
    context: str = 'default'
) -> Dict[str, Any]:
    """
    Resolve indicator and return columns for a given analysis.

    This single function replaces all elif column-detection branches
    across the 7 dashboard pages.

    Args:
        analysis_id: Analysis identifier (e.g., 'spy_retailirsa')
        data: DataFrame loaded from parquet
        context: 'default' for display pages (Overview, Correlation, Lead-Lag, Forecasts)
                 'trading' for action pages (Regimes, Backtests)

    Returns:
        Dict with:
            'indicator_cols': list of indicator column names
            'return_cols': list of return column names
            'indicator_col': primary indicator column
            'return_col': primary return column
            'regime_method': 'direction' | 'threshold' | 'precomputed'
            'regime_config': dict with regime parameters
            'lag_config': dict with lag slider parameters
            'data': DataFrame (possibly modified with computed columns)
    """
    analysis = get_analysis_config(analysis_id)
    config = get_indicator_config(analysis_id, axis='primary')

    if analysis is None or config is None:
        # Fallback for unknown analyses
        return _resolve_fallback(data)

    # --- Resolve indicator columns ---
    indicator_cols = _resolve_indicator_cols(config, data, context)

    # --- Resolve return columns ---
    return_cols = _resolve_return_cols(config, analysis, data)

    # --- Fallbacks ---
    if not indicator_cols:
        exclude = set(return_cols + ['regime', 'Regime'])
        if config.get('price_column'):
            exclude.add(config['price_column'])
        indicator_cols = [c for c in data.columns if c not in exclude
                         and not c.endswith('_return') and not c.endswith('_Returns')]

    if not return_cols:
        # Try computing from price column
        price_col = config.get('price_column') or analysis.get('target_ticker')
        if price_col and price_col in data.columns:
            ret_col = f'{price_col}_return'
            data[ret_col] = data[price_col].pct_change()
            return_cols = [ret_col]
        else:
            # Generic fallback
            return_cols = [c for c in data.columns if c.endswith('_return') or c.endswith('_Returns')]

    # --- Handle regime ---
    regime_method = config.get('regime_method', 'direction')
    regime_config = {
        'method': regime_method,
        'threshold': config.get('regime_threshold'),
        'labels': config.get('regime_labels'),
        'source_col': config.get('regime_source_col'),
    }

    if regime_method == 'precomputed':
        source_col = config.get('regime_source_col', 'Regime')
        if source_col in data.columns and 'regime' not in data.columns:
            data['regime'] = data[source_col]

    # --- Lag config ---
    lag_config = {
        'base_col': config.get('base_column'),
        'default': config.get('default_lag', 0),
        'min': config.get('lag_min', -12),
        'max': config.get('lag_max', 12),
    }

    # --- Price column (for level/trend charts) ---
    price_col = _resolve_price_col(config, analysis, data)

    return {
        'indicator_cols': indicator_cols,
        'return_cols': return_cols,
        'indicator_col': indicator_cols[0] if indicator_cols else None,
        'return_col': return_cols[0] if return_cols else None,
        'price_col': price_col,
        'regime_method': regime_method,
        'regime_config': regime_config,
        'lag_config': lag_config,
        'data': data,
    }


def _resolve_indicator_cols(
    config: Dict,
    data: pd.DataFrame,
    context: str
) -> List[str]:
    """Resolve indicator columns from config."""

    # Trading context: use trading_columns if available
    if context == 'trading' and config.get('trading_columns'):
        trading_cols = config['trading_columns']
        found = [c for c in trading_cols if c in data.columns]
        if found:
            return found

    # Exact column names
    if config.get('indicator_columns'):
        exact_cols = config['indicator_columns']
        found = [c for c in exact_cols if c in data.columns]
        if found:
            return found

    # Filter-based detection (contains + and_contains)
    if config.get('indicator_filter'):
        filt = config['indicator_filter']
        contains = filt.get('contains', [])
        and_contains = filt.get('and_contains', [])
        cols = []
        for c in data.columns:
            if any(pat in c for pat in contains):
                if not and_contains or any(ac in c for ac in and_contains):
                    cols.append(c)
        if cols:
            return cols

    # Pattern-based detection (case-insensitive, pipe-separated patterns)
    if config.get('indicator_pattern'):
        pattern = config['indicator_pattern']
        # Support pipe-separated patterns: 'indpro|industrial'
        patterns = [p.strip() for p in pattern.split('|')]
        exclude_suffixes = config.get('indicator_exclude', ['_return', '_Returns'])

        cols = []
        for c in data.columns:
            c_lower = c.lower()
            if any(p.lower() in c_lower for p in patterns):
                if not any(c.endswith(ex) for ex in exclude_suffixes):
                    cols.append(c)
        if cols:
            return cols

    return []


def _resolve_return_cols(
    config: Dict,
    analysis: Dict,
    data: pd.DataFrame
) -> List[str]:
    """Resolve return columns from config."""

    # Exact column names
    if config.get('return_columns'):
        exact_cols = config['return_columns']
        found = [c for c in exact_cols if c in data.columns]
        if found:
            return found

    # Pattern-based (suffix match)
    if config.get('return_pattern'):
        pattern = config['return_pattern']
        cols = [c for c in data.columns if c.endswith(pattern)]
        if cols:
            return cols

    # From analysis target_return_col
    if analysis.get('target_return_col') and analysis['target_return_col'] in data.columns:
        return [analysis['target_return_col']]

    return []


def _resolve_price_col(
    config: Dict,
    analysis: Dict,
    data: pd.DataFrame
) -> Optional[str]:
    """Resolve the price/level column for the target asset.

    Tries in order:
      1. config['price_column'] (older analyses: 'SPY', 'XLRE', etc.)
      2. '{target_ticker}_Level' (newer analyses: 'SPY_Level', 'XLI_Level')
      3. target_ticker directly (fallback)
    """
    # Explicit price column from config
    pc = config.get('price_column')
    if pc and pc in data.columns:
        return pc

    # Newer naming: {ticker}_Level
    ticker = analysis.get('target_ticker')
    if ticker:
        level_col = f'{ticker}_Level'
        if level_col in data.columns:
            return level_col
        if ticker in data.columns:
            return ticker

    return None


def _resolve_fallback(data: pd.DataFrame) -> Dict[str, Any]:
    """Fallback resolution for unknown analyses."""
    indicator_cols = [c for c in data.columns
                      if not c.endswith('_return') and not c.endswith('_Returns')
                      and c != 'regime' and c != 'Regime']
    return_cols = [c for c in data.columns
                   if c.endswith('_return') or c.endswith('_Returns')]

    return {
        'indicator_cols': indicator_cols,
        'return_cols': return_cols,
        'indicator_col': indicator_cols[0] if indicator_cols else None,
        'return_col': return_cols[0] if return_cols else None,
        'regime_method': 'direction',
        'regime_config': {'method': 'direction'},
        'lag_config': {'base_col': None, 'default': 0, 'min': -12, 'max': 12},
        'data': data,
    }


# =============================================================================
# Composite 4-Regime Support
# =============================================================================

def combine_regimes(
    primary_regime: pd.Series,
    secondary_regime: pd.Series,
    phase_labels: Dict[str, str]
) -> pd.Series:
    """
    Combine two binary regime signals into 4 phases.

    Args:
        primary_regime: Series of regime labels (e.g., 'Rising'/'Falling')
        secondary_regime: Series of regime labels
        phase_labels: Mapping of "primary+secondary" to phase name

    Returns:
        Series with 4-phase labels
    """
    combined = primary_regime.astype(str) + '+' + secondary_regime.astype(str)
    label_map = {k.lower(): v for k, v in phase_labels.items()}
    return combined.str.lower().map(label_map)


# =============================================================================
# Auto-initialize on import
# =============================================================================

def _ensure_seeded():
    """Create tables and seed from seed script (supports Streamlit Cloud).

    Always re-seeds to pick up any additions/removals in seed data.
    The seed script is idempotent (DELETE + INSERT).
    """
    if not DB_PATH.parent.exists():
        return
    init_db()
    import importlib.util
    seed_path = PROJECT_ROOT / "script" / "seed_config_db.py"
    if seed_path.exists():
        spec = importlib.util.spec_from_file_location("seed_config_db", str(seed_path))
        seed_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(seed_module)
        seed_module.main()

_ensure_seeded()
