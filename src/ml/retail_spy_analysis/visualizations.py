#!/usr/bin/env python3
"""
Visualizations for SPY vs RETAILIRSA Analysis

Creates charts with annotated examples to illustrate key findings.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime
import os


def load_data():
    """Load the analysis dataset."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    data_path = os.path.join(project_dir, 'data', 'spy_retail_recession.parquet')
    return pd.read_parquet(data_path)


def plot_overview_with_regimes(df, save_path=None):
    """
    Plot 1: Overview of SPY and RETAILIRSA with recession shading.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel 1: SPY Price
    ax1 = axes[0]
    ax1.plot(df.index, df['SPY'], 'b-', linewidth=1.5, label='SPY')
    ax1.set_ylabel('SPY Price ($)', fontsize=11)
    ax1.set_title('SPY vs Retail Inventories-to-Sales Ratio (1993-2025)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Shade recessions
    recession_starts = df[df['Recession'].diff() == 1].index
    recession_ends = df[df['Recession'].diff() == -1].index

    for start, end in zip(recession_starts, recession_ends):
        ax1.axvspan(start, end, alpha=0.3, color='gray', label='Recession' if start == recession_starts[0] else '')

    # Panel 2: RETAILIRSA Level
    ax2 = axes[1]
    ax2.plot(df.index, df['Retail_Inv_Sales_Ratio'], 'r-', linewidth=1.5, label='RETAILIRSA')
    ax2.axhline(y=df['Retail_Inv_Sales_Ratio'].median(), color='r', linestyle='--', alpha=0.5, label=f'Median ({df["Retail_Inv_Sales_Ratio"].median():.2f})')
    ax2.set_ylabel('Inv/Sales Ratio', fontsize=11)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    for start, end in zip(recession_starts, recession_ends):
        ax2.axvspan(start, end, alpha=0.3, color='gray')

    # Panel 3: RETAILIRSA YoY Change
    ax3 = axes[2]
    colors = ['green' if x < 0 else 'red' for x in df['RETAILIRSA_YoY'].fillna(0)]
    ax3.bar(df.index, df['RETAILIRSA_YoY'], color=colors, alpha=0.7, width=25)
    ax3.axhline(y=0, color='black', linewidth=1)
    ax3.set_ylabel('RETAILIRSA YoY (%)', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.grid(True, alpha=0.3)

    for start, end in zip(recession_starts, recession_ends):
        ax3.axvspan(start, end, alpha=0.3, color='gray')

    # Add legend for colors
    ax3.text(0.02, 0.95, 'Green = Falling (Bullish)\nRed = Rising (Bearish)',
             transform=ax3.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_correlation_evidence(df, save_path=None):
    """
    Plot 2: Evidence of contemporaneous negative correlation.
    Shows scatter plot and two annotated time periods.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Calculate SPY returns
    df = df.copy()
    df['SPY_MoM'] = df['SPY'].pct_change(1) * 100

    # Panel 1: Scatter plot - RETAILIRSA_QoQ vs SPY_QoQ
    ax1 = axes[0, 0]
    spy_qoq = df['SPY'].pct_change(3) * 100
    valid = pd.DataFrame({'x': df['RETAILIRSA_QoQ'], 'y': spy_qoq}).dropna()

    ax1.scatter(valid['x'], valid['y'], alpha=0.5, s=30, c='steelblue')

    # Add regression line
    z = np.polyfit(valid['x'], valid['y'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid['x'].min(), valid['x'].max(), 100)
    ax1.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Trend (r = -0.34)')

    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('RETAILIRSA QoQ Change (%)', fontsize=11)
    ax1.set_ylabel('SPY QoQ Return (%)', fontsize=11)
    ax1.set_title('Contemporaneous Correlation: r = -0.34', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Example 1 - 2008 GFC (Rising Inventories = Falling SPY)
    ax2 = axes[0, 1]

    start_date = '2007-06-01'
    end_date = '2009-12-31'
    mask = (df.index >= start_date) & (df.index <= end_date)
    subset = df[mask]

    ax2_twin = ax2.twinx()

    line1, = ax2.plot(subset.index, subset['SPY'], 'b-', linewidth=2, label='SPY')
    line2, = ax2_twin.plot(subset.index, subset['Retail_Inv_Sales_Ratio'], 'r-', linewidth=2, label='RETAILIRSA')

    ax2.set_ylabel('SPY Price ($)', color='blue', fontsize=11)
    ax2_twin.set_ylabel('Inv/Sales Ratio', color='red', fontsize=11)
    ax2.set_title('Example 1: 2008 GFC\nRising Inventories → Falling SPY', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2_twin.tick_params(axis='y', labelcolor='red')

    # Add annotations
    ax2.annotate('Lehman\nBankruptcy', xy=(pd.Timestamp('2008-09-15'), 100),
                 xytext=(pd.Timestamp('2008-03-01'), 130),
                 arrowprops=dict(arrowstyle='->', color='black'),
                 fontsize=9, ha='center')

    ax2.annotate('Inv/Sales\nspikes to 1.75', xy=(pd.Timestamp('2009-01-31'), 70),
                 xytext=(pd.Timestamp('2009-06-01'), 90),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=9, ha='center', color='red')

    ax2.legend([line1, line2], ['SPY', 'RETAILIRSA'], loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Example 2 - 2020 COVID (Spike then recovery)
    ax3 = axes[1, 0]

    start_date = '2019-06-01'
    end_date = '2021-06-30'
    mask = (df.index >= start_date) & (df.index <= end_date)
    subset = df[mask]

    ax3_twin = ax3.twinx()

    line1, = ax3.plot(subset.index, subset['SPY'], 'b-', linewidth=2, label='SPY')
    line2, = ax3_twin.plot(subset.index, subset['Retail_Inv_Sales_Ratio'], 'r-', linewidth=2, label='RETAILIRSA')

    ax3.set_ylabel('SPY Price ($)', color='blue', fontsize=11)
    ax3_twin.set_ylabel('Inv/Sales Ratio', color='red', fontsize=11)
    ax3.set_title('Example 2: COVID-19 Pandemic\nInventory Spike → Market Crash, then Reversal', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='y', labelcolor='blue')
    ax3_twin.tick_params(axis='y', labelcolor='red')

    # Add annotations
    ax3.annotate('COVID\nCrash', xy=(pd.Timestamp('2020-03-23'), 220),
                 xytext=(pd.Timestamp('2020-01-01'), 280),
                 arrowprops=dict(arrowstyle='->', color='black'),
                 fontsize=9, ha='center')

    ax3.annotate('Inv/Sales falls\nas demand surges', xy=(pd.Timestamp('2021-01-31'), 380),
                 xytext=(pd.Timestamp('2020-10-01'), 320),
                 arrowprops=dict(arrowstyle='->', color='green'),
                 fontsize=9, ha='center', color='green')

    ax3.legend([line1, line2], ['SPY', 'RETAILIRSA'], loc='upper left')
    ax3.grid(True, alpha=0.3)

    # Panel 4: Example 3 - 2014-2016 (Falling inventories = Strong SPY)
    ax4 = axes[1, 1]

    start_date = '2014-01-01'
    end_date = '2016-12-31'
    mask = (df.index >= start_date) & (df.index <= end_date)
    subset = df[mask]

    ax4_twin = ax4.twinx()

    line1, = ax4.plot(subset.index, subset['SPY'], 'b-', linewidth=2, label='SPY')
    line2, = ax4_twin.plot(subset.index, subset['Retail_Inv_Sales_Ratio'], 'r-', linewidth=2, label='RETAILIRSA')

    ax4.set_ylabel('SPY Price ($)', color='blue', fontsize=11)
    ax4_twin.set_ylabel('Inv/Sales Ratio', color='red', fontsize=11)
    ax4.set_title('Example 3: 2014-2016\nFalling Inventories → Strong SPY', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='y', labelcolor='blue')
    ax4_twin.tick_params(axis='y', labelcolor='red')

    ax4.annotate('Steady decline\nin Inv/Sales', xy=(pd.Timestamp('2015-06-01'), 200),
                 xytext=(pd.Timestamp('2014-06-01'), 180),
                 arrowprops=dict(arrowstyle='->', color='green'),
                 fontsize=9, ha='center', color='green')

    ax4.legend([line1, line2], ['SPY', 'RETAILIRSA'], loc='upper left')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_regime_analysis(df, save_path=None):
    """
    Plot 3: Regime analysis - SPY performance in different RETAILIRSA regimes.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    df = df.copy()
    df['SPY_MoM'] = df['SPY'].pct_change(1) * 100

    # Define regimes
    median_level = df['Retail_Inv_Sales_Ratio'].median()
    df['Level_Regime'] = np.where(df['Retail_Inv_Sales_Ratio'] > median_level, 'High', 'Low')
    df['Direction_Regime'] = np.where(df['RETAILIRSA_YoY'] > 0, 'Rising', 'Falling')

    # Panel 1: Box plot by Level Regime
    ax1 = axes[0, 0]

    high_returns = df[df['Level_Regime'] == 'High']['SPY_MoM'].dropna()
    low_returns = df[df['Level_Regime'] == 'Low']['SPY_MoM'].dropna()

    bp = ax1.boxplot([low_returns, high_returns], labels=['Low Inv/Sales', 'High Inv/Sales'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')

    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('SPY Monthly Return (%)', fontsize=11)
    ax1.set_title('SPY Returns by Inventory Level\n(Low Inv/Sales = Better)', fontsize=12, fontweight='bold')

    # Add stats
    ax1.text(1, ax1.get_ylim()[1] * 0.9, f'Mean: {low_returns.mean():.2f}%\nSharpe: 1.03',
             ha='center', fontsize=10, color='green')
    ax1.text(2, ax1.get_ylim()[1] * 0.9, f'Mean: {high_returns.mean():.2f}%\nSharpe: 0.51',
             ha='center', fontsize=10, color='red')
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel 2: Box plot by Direction Regime
    ax2 = axes[0, 1]

    falling_returns = df[df['Direction_Regime'] == 'Falling']['SPY_MoM'].dropna()
    rising_returns = df[df['Direction_Regime'] == 'Rising']['SPY_MoM'].dropna()

    bp = ax2.boxplot([falling_returns, rising_returns], labels=['Falling Inv/Sales', 'Rising Inv/Sales'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')

    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('SPY Monthly Return (%)', fontsize=11)
    ax2.set_title('SPY Returns by Inventory Direction\n(Falling = Better)', fontsize=12, fontweight='bold')

    ax2.text(1, ax2.get_ylim()[1] * 0.9, f'Mean: {falling_returns.mean():.2f}%\nSharpe: 0.98',
             ha='center', fontsize=10, color='green')
    ax2.text(2, ax2.get_ylim()[1] * 0.9, f'Mean: {rising_returns.mean():.2f}%\nSharpe: 0.52',
             ha='center', fontsize=10, color='red')
    ax2.grid(True, alpha=0.3, axis='y')

    # Panel 3: Cumulative returns by regime
    ax3 = axes[1, 0]

    # Calculate cumulative returns for each regime
    df['Falling_Return'] = np.where(df['Direction_Regime'] == 'Falling', df['SPY_MoM'], 0)
    df['Rising_Return'] = np.where(df['Direction_Regime'] == 'Rising', df['SPY_MoM'], 0)

    cum_falling = (1 + df['Falling_Return']/100).cumprod() - 1
    cum_rising = (1 + df['Rising_Return']/100).cumprod() - 1
    cum_total = (1 + df['SPY_MoM'].fillna(0)/100).cumprod() - 1

    ax3.plot(df.index, cum_falling * 100, 'g-', linewidth=2, label='Falling Inv/Sales Only')
    ax3.plot(df.index, cum_rising * 100, 'r-', linewidth=2, label='Rising Inv/Sales Only')
    ax3.plot(df.index, cum_total * 100, 'b--', linewidth=1, alpha=0.5, label='Buy & Hold')

    ax3.set_ylabel('Cumulative Return (%)', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_title('Cumulative Returns by Regime\n(Falling Inv/Sales Dominates)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    # Panel 4: Recession vs Non-recession
    ax4 = axes[1, 1]

    recession_returns = df[df['Recession'] == 1]['SPY_MoM'].dropna()
    expansion_returns = df[df['Recession'] == 0]['SPY_MoM'].dropna()

    bp = ax4.boxplot([expansion_returns, recession_returns], labels=['Expansion', 'Recession'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')

    ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_ylabel('SPY Monthly Return (%)', fontsize=11)
    ax4.set_title('SPY Returns: Expansion vs Recession\n(NBER Indicator)', fontsize=12, fontweight='bold')

    ax4.text(1, ax4.get_ylim()[1] * 0.85, f'Mean: {expansion_returns.mean():.2f}%\nn={len(expansion_returns)}',
             ha='center', fontsize=10, color='green')
    ax4.text(2, ax4.get_ylim()[1] * 0.85, f'Mean: {recession_returns.mean():.2f}%\nn={len(recession_returns)}',
             ha='center', fontsize=10, color='red')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_lead_lag_analysis(df, save_path=None):
    """
    Plot 4: Lead-lag correlation analysis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    df = df.copy()
    spy_returns = df['SPY'].pct_change(1) * 100

    # Panel 1: Lead-lag correlations for different features
    ax1 = axes[0, 0]

    lags = range(-12, 13)

    for feature, color, label in [
        ('RETAILIRSA_MoM', 'blue', 'MoM'),
        ('RETAILIRSA_QoQ', 'red', 'QoQ'),
        ('RETAILIRSA_YoY', 'green', 'YoY')
    ]:
        correlations = []
        for lag in lags:
            if lag < 0:
                x = df[feature].shift(-lag)
                y = spy_returns
            else:
                x = df[feature]
                y = spy_returns.shift(-lag)

            valid = pd.DataFrame({'x': x, 'y': y}).dropna()
            if len(valid) > 50:
                corr = valid['x'].corr(valid['y'])
                correlations.append(corr)
            else:
                correlations.append(np.nan)

        ax1.plot(lags, correlations, color=color, linewidth=2, marker='o', markersize=4, label=label)

    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.axhspan(-0.1, 0.1, alpha=0.2, color='gray', label='Noise zone')
    ax1.set_xlabel('Lag (months, negative = RETAILIRSA leads)', fontsize=11)
    ax1.set_ylabel('Correlation with SPY Returns', fontsize=11)
    ax1.set_title('Lead-Lag Correlations\n(Peak at Lag=0 = Contemporaneous)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.4, 0.2)

    # Panel 2: Example - RETAILIRSA QoQ leads by showing 2007-2008
    ax2 = axes[0, 1]

    start_date = '2006-01-01'
    end_date = '2009-06-30'
    mask = (df.index >= start_date) & (df.index <= end_date)
    subset = df[mask].copy()
    subset['SPY_QoQ'] = subset['SPY'].pct_change(3) * 100

    ax2_twin = ax2.twinx()

    line1, = ax2.plot(subset.index, subset['SPY_QoQ'], 'b-', linewidth=2, label='SPY QoQ Return')
    line2, = ax2_twin.plot(subset.index, subset['RETAILIRSA_QoQ'], 'r-', linewidth=2, label='RETAILIRSA QoQ')

    ax2.axhline(0, color='blue', linestyle='--', alpha=0.3)
    ax2_twin.axhline(0, color='red', linestyle='--', alpha=0.3)

    ax2.set_ylabel('SPY QoQ Return (%)', color='blue', fontsize=11)
    ax2_twin.set_ylabel('RETAILIRSA QoQ Change (%)', color='red', fontsize=11)
    ax2.set_title('2006-2009: Contemporaneous Relationship\n(Both move together, opposite directions)', fontsize=12, fontweight='bold')

    # Highlight key periods
    ax2.annotate('Both reverse\ntogether', xy=(pd.Timestamp('2008-10-31'), -30),
                 xytext=(pd.Timestamp('2008-01-01'), -20),
                 arrowprops=dict(arrowstyle='->', color='black'),
                 fontsize=9, ha='center')

    ax2.legend([line1, line2], ['SPY QoQ', 'RETAILIRSA QoQ'], loc='lower left')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Rolling correlation over time
    ax3 = axes[1, 0]

    # 24-month rolling correlation
    rolling_corr = df['RETAILIRSA_QoQ'].rolling(24).corr(spy_returns.shift(0))

    ax3.plot(df.index, rolling_corr, 'purple', linewidth=1.5)
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(-0.34, color='red', linestyle='--', alpha=0.5, label='Full-period correlation (-0.34)')

    ax3.fill_between(df.index, rolling_corr, 0, where=rolling_corr < 0, alpha=0.3, color='red')
    ax3.fill_between(df.index, rolling_corr, 0, where=rolling_corr > 0, alpha=0.3, color='green')

    ax3.set_ylabel('24-Month Rolling Correlation', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_title('Rolling Correlation: RETAILIRSA QoQ vs SPY Returns\n(Mostly Negative)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-1, 1)

    # Panel 4: No predictive power evidence
    ax4 = axes[1, 1]

    # Show that lagged RETAILIRSA doesn't predict future SPY
    lag = 3  # 3-month lag
    x = df['RETAILIRSA_QoQ'].shift(lag)  # RETAILIRSA 3 months ago
    y = spy_returns  # Current SPY return

    valid = pd.DataFrame({'x': x, 'y': y}).dropna()

    ax4.scatter(valid['x'], valid['y'], alpha=0.4, s=30, c='gray')

    # Add regression line
    z = np.polyfit(valid['x'], valid['y'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid['x'].min(), valid['x'].max(), 100)
    ax4.plot(x_line, p(x_line), 'r-', linewidth=2)

    corr = valid['x'].corr(valid['y'])
    ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(0, color='gray', linestyle='--', alpha=0.5)

    ax4.set_xlabel('RETAILIRSA QoQ (3 months ago)', fontsize=11)
    ax4.set_ylabel('SPY Monthly Return (now)', fontsize=11)
    ax4.set_title(f'No Predictive Power: r = {corr:.3f}\n(Lagged RETAILIRSA vs Current SPY)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    ax4.text(0.95, 0.05, 'Flat relationship = \nNo predictive value',
             transform=ax4.transAxes, fontsize=10, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def create_all_visualizations():
    """Generate all visualization plots."""
    print("=" * 60)
    print("GENERATING SPY vs RETAILIRSA VISUALIZATIONS")
    print("=" * 60)

    # Load data
    df = load_data()
    print(f"\nData loaded: {len(df)} observations")

    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    output_dir = os.path.join(project_dir, 'data')

    # Generate plots
    print("\n1. Creating overview plot...")
    plot_overview_with_regimes(df, os.path.join(output_dir, 'spy_retailirsa_overview.png'))

    print("\n2. Creating correlation evidence plot...")
    plot_correlation_evidence(df, os.path.join(output_dir, 'spy_retailirsa_correlation.png'))

    print("\n3. Creating regime analysis plot...")
    plot_regime_analysis(df, os.path.join(output_dir, 'spy_retailirsa_regimes.png'))

    print("\n4. Creating lead-lag analysis plot...")
    plot_lead_lag_analysis(df, os.path.join(output_dir, 'spy_retailirsa_leadlag.png'))

    print("\n" + "=" * 60)
    print("ALL VISUALIZATIONS COMPLETE")
    print("=" * 60)

    plt.show()


if __name__ == '__main__':
    create_all_visualizations()
