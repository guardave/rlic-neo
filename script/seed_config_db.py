#!/usr/bin/env python3
"""
Seed the RLIC configuration database from current hardcoded values.

Populates the analyses and analysis_indicators tables with data
extracted from navigation.py, data_loader.py, and Home.py.

Idempotent: uses INSERT OR REPLACE so it can be re-run safely.

Usage:
    python script/seed_config_db.py
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.config_db import get_connection, init_db

# =============================================================================
# Seed Data
# =============================================================================

ANALYSES_SEED = [
    {
        'id': 'spy_retailirsa',
        'name': 'SPY vs Retail Inv/Sales',
        'icon': 'inventory_2',
        'short_name': 'SPY x Retail Inv/Sales',
        'description': 'S&P 500 vs retail inventory-to-sales ratio',
        'caption': 'RETAILIRSA \u2022 SPY \u2022 Lead-Lag Analysis',
        'home_column': 1,
        'display_order': 20,
        'analysis_type': 'single',
        'target_ticker': 'SPY',
        'target_return_col': None,
        'data_file': 'spy_retail_inv_sales.parquet',
        'phase_labels': None,
    },
    {
        'id': 'spy_indpro',
        'name': 'SPY vs Industrial Production',
        'icon': 'precision_manufacturing',
        'short_name': 'SPY x Indust. Prod.',
        'description': 'S&P 500 vs industrial production index',
        'caption': 'INDPRO \u2022 SPY \u2022 Regime Analysis',
        'home_column': 2,
        'display_order': 30,
        'analysis_type': 'single',
        'target_ticker': 'SPY',
        'target_return_col': None,
        'data_file': 'spy_ip_analysis.parquet',
        'phase_labels': None,
    },
    {
        'id': 'xlre_orders_inv',
        'name': 'XLRE vs Orders/Inventories',
        'icon': 'real_estate_agent',
        'short_name': 'XLRE x Orders/Inv',
        'description': 'Real estate sector vs orders-to-inventories ratio',
        'caption': 'Orders/Inv Ratio \u2022 XLRE \u2022 Backtest',
        'home_column': 2,
        'display_order': 40,
        'analysis_type': 'single',
        'target_ticker': 'XLRE',
        'target_return_col': None,
        'data_file': 'xlre_oi_analysis.parquet',
        'phase_labels': None,
    },
    {
        'id': 'xlp_retailirsa',
        'name': 'XLP vs Retail Inv/Sales',
        'icon': 'shopping_basket',
        'short_name': 'XLP x Retail Inv/Sales',
        'description': 'Consumer Staples sector vs retail inventory-to-sales ratio',
        'caption': 'RETAILIRSA \u2022 XLP \u2022 Lead-Lag Analysis',
        'home_column': 1,
        'display_order': 50,
        'analysis_type': 'single',
        'target_ticker': 'XLP',
        'target_return_col': None,
        'data_file': 'xlp_retail_inv_sales.parquet',
        'phase_labels': None,
    },
    {
        'id': 'xly_retailirsa',
        'name': 'XLY vs Retail Inv/Sales',
        'icon': 'storefront',
        'short_name': 'XLY x Retail Inv/Sales',
        'description': 'Consumer Discretionary sector vs retail inventory-to-sales ratio',
        'caption': 'RETAILIRSA \u2022 XLY \u2022 Lead-Lag Analysis',
        'home_column': 2,
        'display_order': 60,
        'analysis_type': 'single',
        'target_ticker': 'XLY',
        'target_return_col': None,
        'data_file': 'xly_retail_inv_sales.parquet',
        'phase_labels': None,
    },
    {
        'id': 'xlre_newhomesales',
        'name': 'XLRE vs New Home Sales',
        'icon': 'home_work',
        'short_name': 'XLRE x New Home Sales',
        'description': 'Real estate sector vs new home sales (lag +8 significant)',
        'caption': 'New Home Sales \u2022 XLRE \u2022 Lag +8 Significant',
        'home_column': 1,
        'display_order': 70,
        'analysis_type': 'single',
        'target_ticker': 'XLRE',
        'target_return_col': 'XLRE_Returns',
        'data_file': 'xlre_newhomesales_full.parquet',
        'phase_labels': None,
    },
    {
        'id': 'xli_ism_mfg',
        'name': 'XLI vs ISM Manufacturing PMI',
        'icon': 'factory',
        'short_name': 'XLI x ISM Manu. PMI',
        'description': 'Industrials sector vs ISM Manufacturing PMI (confirmatory)',
        'caption': 'ISM Mfg PMI \u2022 XLI \u2022 Confirmatory Signal',
        'home_column': 2,
        'display_order': 80,
        'analysis_type': 'single',
        'target_ticker': 'XLI',
        'target_return_col': 'XLI_Returns',
        'data_file': 'xli_ism_mfg_full.parquet',
        'phase_labels': None,
    },
    {
        'id': 'xli_ism_svc',
        'name': 'XLI vs ISM Services PMI',
        'icon': 'corporate_fare',
        'short_name': 'XLI x ISM Svc. PMI',
        'description': 'Industrials sector vs ISM Services PMI (confirmatory)',
        'caption': 'ISM Svc PMI \u2022 XLI \u2022 Confirmatory Signal',
        'home_column': 1,
        'display_order': 90,
        'analysis_type': 'single',
        'target_ticker': 'XLI',
        'target_return_col': 'XLI_Returns',
        'data_file': 'xli_ism_svc_full.parquet',
        'phase_labels': None,
    },
    {
        'id': 'spy_hy_ig_spread',
        'name': 'SPY vs HY-IG Credit Spread',
        'icon': 'show_chart',
        'short_name': 'SPY x HY-IG Spread',
        'description': 'S&P 500 vs High Yield minus Investment Grade credit spread',
        'caption': 'HY-IG Spread \u2022 SPY \u2022 Credit Risk Signal',
        'home_column': 2,
        'display_order': 100,
        'analysis_type': 'single',
        'target_ticker': 'SPY',
        'target_return_col': 'SPY_Returns',
        'data_file': 'spy_hy_ig_spread_full.parquet',
        'phase_labels': None,
    },
]


INDICATORS_SEED = [
    # --- Pattern-based analyses (older, use _return suffix) ---
    {
        'analysis_id': 'spy_retailirsa',
        'axis': 'primary',
        'indicator_pattern': 'retail',
        'indicator_columns': None,
        'indicator_filter': None,
        'indicator_exclude': json.dumps(['_return']),
        'trading_columns': None,
        'return_columns': None,
        'return_pattern': '_return',
        'price_column': 'SPY',
        'exclude_from_detection': json.dumps(['SPY', 'regime']),
        'base_column': 'retail_inv_sales',
        'default_lag': 0,
        'lag_min': -12,
        'lag_max': 12,
        'regime_method': 'direction',
        'regime_threshold': None,
        'regime_labels': None,
        'regime_source_col': None,
    },
    {
        'analysis_id': 'spy_indpro',
        'axis': 'primary',
        'indicator_pattern': 'indpro|industrial',
        'indicator_columns': None,
        'indicator_filter': None,
        'indicator_exclude': json.dumps(['_return']),
        'trading_columns': None,
        'return_columns': None,
        'return_pattern': '_return',
        'price_column': 'SPY',
        'exclude_from_detection': json.dumps(['SPY', 'regime']),
        'base_column': 'industrial_prod',
        'default_lag': 0,
        'lag_min': -12,
        'lag_max': 12,
        'regime_method': 'direction',
        'regime_threshold': None,
        'regime_labels': None,
        'regime_source_col': None,
    },
    {
        'analysis_id': 'xlre_orders_inv',
        'axis': 'primary',
        'indicator_pattern': 'order|oi',
        'indicator_columns': None,
        'indicator_filter': None,
        'indicator_exclude': json.dumps(['_return']),
        'trading_columns': None,
        'return_columns': None,
        'return_pattern': '_return',
        'price_column': 'XLRE',
        'exclude_from_detection': json.dumps(['XLRE', 'regime']),
        'base_column': 'orders_inv_ratio',
        'default_lag': 0,
        'lag_min': -12,
        'lag_max': 12,
        'regime_method': 'direction',
        'regime_threshold': None,
        'regime_labels': None,
        'regime_source_col': None,
    },
    {
        'analysis_id': 'xlp_retailirsa',
        'axis': 'primary',
        'indicator_pattern': 'retail',
        'indicator_columns': None,
        'indicator_filter': None,
        'indicator_exclude': json.dumps(['_return']),
        'trading_columns': None,
        'return_columns': None,
        'return_pattern': '_return',
        'price_column': 'XLP',
        'exclude_from_detection': json.dumps(['XLP', 'regime']),
        'base_column': 'retail_inv_sales',
        'default_lag': 0,
        'lag_min': -12,
        'lag_max': 12,
        'regime_method': 'direction',
        'regime_threshold': None,
        'regime_labels': None,
        'regime_source_col': None,
    },
    {
        'analysis_id': 'xly_retailirsa',
        'axis': 'primary',
        'indicator_pattern': 'retail',
        'indicator_columns': None,
        'indicator_filter': None,
        'indicator_exclude': json.dumps(['_return']),
        'trading_columns': None,
        'return_columns': None,
        'return_pattern': '_return',
        'price_column': 'XLY',
        'exclude_from_detection': json.dumps(['XLY', 'regime']),
        'base_column': 'retail_inv_sales',
        'default_lag': 0,
        'lag_min': -12,
        'lag_max': 12,
        'regime_method': 'direction',
        'regime_threshold': None,
        'regime_labels': None,
        'regime_source_col': None,
    },
    # --- Exact-column analyses (newer, use _Returns suffix) ---
    {
        'analysis_id': 'xlre_newhomesales',
        'axis': 'primary',
        'indicator_pattern': None,
        'indicator_columns': json.dumps(['NewHomeSales_Level', 'NewHomeSales_YoY']),
        'indicator_filter': json.dumps({'contains': ['NewHomeSales'], 'and_contains': ['Level', 'YoY']}),
        'indicator_exclude': None,
        'trading_columns': json.dumps(['NewHomeSales_YoY_Lagged']),
        'return_columns': json.dumps(['XLRE_Returns']),
        'return_pattern': None,
        'price_column': None,
        'exclude_from_detection': None,
        'base_column': 'NewHomeSales_YoY',
        'default_lag': 8,
        'lag_min': -12,
        'lag_max': 24,
        'regime_method': 'precomputed',
        'regime_threshold': None,
        'regime_labels': None,
        'regime_source_col': 'Regime',
    },
    {
        'analysis_id': 'xli_ism_mfg',
        'axis': 'primary',
        'indicator_pattern': None,
        'indicator_columns': json.dumps(['ISM_Mfg_PMI_Level', 'ISM_Mfg_PMI_YoY']),
        'indicator_filter': json.dumps({'contains': ['ISM_Mfg_PMI'], 'and_contains': ['Level', 'YoY']}),
        'indicator_exclude': None,
        'trading_columns': json.dumps(['ISM_Mfg_PMI_Level_Lagged']),
        'return_columns': json.dumps(['XLI_Returns']),
        'return_pattern': None,
        'price_column': None,
        'exclude_from_detection': None,
        'base_column': 'ISM_Mfg_PMI_Level',
        'default_lag': 0,
        'lag_min': -12,
        'lag_max': 12,
        'regime_method': 'threshold',
        'regime_threshold': 50.0,
        'regime_labels': json.dumps({'above': 'Mfg Expansion', 'below': 'Mfg Contraction'}),
        'regime_source_col': 'Regime',
    },
    {
        'analysis_id': 'xli_ism_svc',
        'axis': 'primary',
        'indicator_pattern': None,
        'indicator_columns': json.dumps(['ISM_Svc_PMI_Level', 'ISM_Svc_PMI_YoY']),
        'indicator_filter': json.dumps({'contains': ['ISM_Svc_PMI'], 'and_contains': ['Level', 'YoY']}),
        'indicator_exclude': None,
        'trading_columns': json.dumps(['ISM_Svc_PMI_Level_Lagged']),
        'return_columns': json.dumps(['XLI_Returns']),
        'return_pattern': None,
        'price_column': None,
        'exclude_from_detection': None,
        'base_column': 'ISM_Svc_PMI_Level',
        'default_lag': 0,
        'lag_min': -12,
        'lag_max': 12,
        'regime_method': 'threshold',
        'regime_threshold': 50.0,
        'regime_labels': json.dumps({'above': 'Svc Expansion', 'below': 'Svc Contraction'}),
        'regime_source_col': 'Regime',
    },
    # --- Credit spread analysis (precomputed regime) ---
    {
        'analysis_id': 'spy_hy_ig_spread',
        'axis': 'primary',
        'indicator_pattern': None,
        'indicator_columns': json.dumps(['HY_IG_Spread_Level', 'HY_IG_Spread_YoY']),
        'indicator_filter': json.dumps({'contains': ['HY_IG_Spread'], 'and_contains': ['Level', 'YoY']}),
        'indicator_exclude': None,
        'trading_columns': json.dumps(['HY_IG_Spread_YoY_Lagged']),
        'return_columns': json.dumps(['SPY_Returns']),
        'return_pattern': None,
        'price_column': None,
        'exclude_from_detection': None,
        'base_column': 'HY_IG_Spread_YoY',
        'default_lag': -1,
        'lag_min': -18,
        'lag_max': 18,
        'regime_method': 'precomputed',
        'regime_threshold': None,
        'regime_labels': None,
        'regime_source_col': 'Regime',
    },
]


# =============================================================================
# Seed Execution
# =============================================================================

def seed_analyses(conn):
    """Insert or replace analyses rows."""
    cursor = conn.cursor()
    # Clear existing to ensure removed analyses are purged
    cursor.execute("DELETE FROM analyses")
    for a in ANALYSES_SEED:
        cursor.execute("""
            INSERT OR REPLACE INTO analyses
            (id, name, icon, short_name, description, caption, home_column,
             display_order, analysis_type, target_ticker, target_return_col,
             data_file, phase_labels)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            a['id'], a['name'], a['icon'], a['short_name'], a['description'],
            a['caption'], a['home_column'], a['display_order'], a['analysis_type'],
            a['target_ticker'], a['target_return_col'], a['data_file'],
            a['phase_labels']
        ))
    conn.commit()
    print(f"  Inserted {len(ANALYSES_SEED)} analyses")


def seed_indicators(conn):
    """Insert or replace indicator config rows."""
    cursor = conn.cursor()
    # Clear existing to avoid duplicates on re-run
    cursor.execute("DELETE FROM analysis_indicators")
    for i in INDICATORS_SEED:
        cursor.execute("""
            INSERT INTO analysis_indicators
            (analysis_id, axis, indicator_pattern, indicator_columns,
             indicator_filter, indicator_exclude, trading_columns,
             return_columns, return_pattern, price_column,
             exclude_from_detection, base_column, default_lag,
             lag_min, lag_max, regime_method, regime_threshold,
             regime_labels, regime_source_col)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            i['analysis_id'], i['axis'], i['indicator_pattern'],
            i['indicator_columns'], i['indicator_filter'],
            i['indicator_exclude'], i['trading_columns'],
            i['return_columns'], i['return_pattern'], i['price_column'],
            i['exclude_from_detection'], i['base_column'],
            i['default_lag'], i['lag_min'], i['lag_max'],
            i['regime_method'], i['regime_threshold'],
            i['regime_labels'], i['regime_source_col']
        ))
    conn.commit()
    print(f"  Inserted {len(INDICATORS_SEED)} indicator configs")


def main():
    print("Seeding RLIC configuration database...")
    print(f"  DB path: {Path(__file__).parent.parent / 'data' / 'rlic_config.db'}")

    # Ensure tables exist
    init_db()

    conn = get_connection()
    try:
        seed_analyses(conn)
        seed_indicators(conn)

        # Verify
        cursor = conn.execute("SELECT COUNT(*) FROM analyses")
        n_analyses = cursor.fetchone()[0]
        cursor = conn.execute("SELECT COUNT(*) FROM analysis_indicators")
        n_indicators = cursor.fetchone()[0]

        print(f"\nVerification:")
        print(f"  analyses: {n_analyses} rows")
        print(f"  analysis_indicators: {n_indicators} rows")

        # Show summary
        print(f"\nAnalyses:")
        cursor = conn.execute("SELECT id, name, analysis_type, data_file FROM analyses ORDER BY display_order")
        for row in cursor.fetchall():
            print(f"  {row[0]:25s} | {row[1]:35s} | {row[2]:10s} | {row[3]}")

    finally:
        conn.close()

    print("\nDone.")


if __name__ == '__main__':
    main()
