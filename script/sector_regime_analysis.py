#!/usr/bin/env python3
"""
Sector Performance by Investment Clock Regime - First Principles Analysis

Uses Orders/Inventories Ratio (Growth) + PPI (Inflation) to form 4 regimes:
- Recovery:    Growth Rising, Inflation Falling
- Overheat:    Growth Rising, Inflation Rising
- Stagflation: Growth Falling, Inflation Rising
- Reflation:   Growth Falling, Inflation Falling

Analyzes sector returns in each regime using:
1. Fama-French 12 Industries (1992-2025, limited by Orders/Inv data)
2. S&P Sector ETFs (1999-2025)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

# Fama-French 12 Industries to S&P Sector Mapping
FF_TO_SECTOR = {
    'NoDur': 'Consumer Staples',
    'Durbl': 'Consumer Discretionary',
    'Manuf': 'Industrials',
    'Enrgy': 'Energy',
    'Chems': 'Materials',
    'BusEq': 'Technology',
    'Telcm': 'Communication',
    'Utils': 'Utilities',
    'Shops': 'Retail',
    'Hlth': 'Healthcare',
    'Money': 'Financials',
    'Other': 'Other',
}

# Investment Clock theoretical sector preferences
THEORY_SECTORS = {
    'Recovery': ['Technology', 'Industrials', 'Consumer Discretionary', 'Financials'],
    'Overheat': ['Energy', 'Materials', 'Industrials'],
    'Stagflation': ['Healthcare', 'Utilities', 'Consumer Staples'],
    'Reflation': ['Financials', 'Consumer Discretionary', 'Communication']
}


def load_indicators():
    """Load Orders/Inv Ratio and PPI indicators."""
    indicators = pd.read_parquet(os.path.join(DATA_DIR, 'monthly_all_indicators.parquet'))
    indicators.index = pd.to_datetime(indicators.index)
    return indicators


def load_ff_industries():
    """Load Fama-French 12 Industry portfolios."""
    ff = pd.read_parquet(os.path.join(DATA_DIR, 'ff_12_industries.parquet'))
    ff.index = pd.to_datetime(ff.index)
    # Convert to end-of-month to match indicators
    ff.index = ff.index + pd.offsets.MonthEnd(0)
    # Convert from percentage to decimal returns
    ff = ff / 100
    # Rename to sector names
    ff.columns = [FF_TO_SECTOR.get(col, col) for col in ff.columns]
    return ff


def generate_phases(indicators):
    """
    Generate Investment Clock phases using Orders/Inv MoM + PPI MoM.

    Signal Generation:
    - Growth: Orders/Inv 3MA vs 6MA direction
    - Inflation: PPI 3MA vs 6MA direction
    """
    phases = pd.DataFrame(index=indicators.index)

    # Growth Signal: Orders/Inventories Ratio direction
    oi_ratio = indicators['orders_inv_ratio']
    oi_3ma = oi_ratio.rolling(3).mean()
    oi_6ma = oi_ratio.rolling(6).mean()
    phases['growth_signal'] = np.where(oi_3ma > oi_6ma, 1, -1)

    # Inflation Signal: PPI direction
    ppi = indicators['ppi_all']
    ppi_3ma = ppi.rolling(3).mean()
    ppi_6ma = ppi.rolling(6).mean()
    phases['inflation_signal'] = np.where(ppi_3ma > ppi_6ma, 1, -1)

    # Phase Classification
    def classify_phase(row):
        g, i = row['growth_signal'], row['inflation_signal']
        if pd.isna(g) or pd.isna(i):
            return np.nan
        if g == 1 and i == -1:
            return 'Recovery'
        elif g == 1 and i == 1:
            return 'Overheat'
        elif g == -1 and i == 1:
            return 'Stagflation'
        elif g == -1 and i == -1:
            return 'Reflation'
        return np.nan

    phases['phase'] = phases.apply(classify_phase, axis=1)

    return phases


def analyze_sector_performance(phases, returns, lag=1):
    """
    Analyze sector returns by Investment Clock phase.

    Args:
        phases: DataFrame with 'phase' column
        returns: DataFrame with sector returns
        lag: Signal lag to avoid look-ahead bias
    """
    # Align data
    common_idx = phases.index.intersection(returns.index)
    phases = phases.loc[common_idx].copy()
    returns = returns.loc[common_idx].copy()

    # Apply lag to phase signal
    phases['phase_lagged'] = phases['phase'].shift(lag)

    # Debug: check overlap
    valid_phases = phases['phase_lagged'].dropna()
    print(f"\n  Overlap period: {valid_phases.index[0].strftime('%Y-%m')} to {valid_phases.index[-1].strftime('%Y-%m')}")
    print(f"  Valid months with phase: {len(valid_phases)}")

    results = []

    for phase in ['Recovery', 'Overheat', 'Stagflation', 'Reflation']:
        mask = phases['phase_lagged'] == phase
        phase_returns = returns[mask]

        if len(phase_returns) > 0:
            for sector in returns.columns:
                sector_ret = phase_returns[sector].dropna()
                if len(sector_ret) > 6:  # Minimum 6 months
                    ann_ret = sector_ret.mean() * 12 * 100
                    ann_vol = sector_ret.std() * np.sqrt(12) * 100
                    sharpe = (ann_ret - 2) / ann_vol if ann_vol > 0 else 0  # 2% risk-free

                    # Check if this sector is a theory pick for this phase
                    is_theory = sector in THEORY_SECTORS.get(phase, [])
                    # Also check partial matches (e.g., "Consumer Discretionary" in "Retail")
                    if not is_theory:
                        for theory_sector in THEORY_SECTORS.get(phase, []):
                            if theory_sector in sector or sector in theory_sector:
                                is_theory = True
                                break

                    results.append({
                        'phase': phase,
                        'sector': sector,
                        'months': len(sector_ret),
                        'ann_return': ann_ret,
                        'ann_vol': ann_vol,
                        'sharpe': sharpe,
                        'win_rate': (sector_ret > 0).mean() * 100,
                        'is_theory': is_theory
                    })

    if len(results) == 0:
        print("  WARNING: No results generated - check data alignment")
        return pd.DataFrame(columns=['phase', 'sector', 'months', 'ann_return', 'ann_vol', 'sharpe', 'win_rate', 'is_theory'])

    return pd.DataFrame(results)


def rank_sectors_by_phase(results_df):
    """Rank sectors within each phase by annualized return."""
    ranked = []

    for phase in ['Recovery', 'Overheat', 'Stagflation', 'Reflation']:
        phase_data = results_df[results_df['phase'] == phase].copy()
        phase_data = phase_data.sort_values('ann_return', ascending=False)
        phase_data['rank'] = range(1, len(phase_data) + 1)
        ranked.append(phase_data)

    return pd.concat(ranked, ignore_index=True)


def calculate_phase_statistics(phases):
    """Calculate statistics about phase distribution."""
    phase_counts = phases['phase'].value_counts()
    total = len(phases['phase'].dropna())

    stats = []
    for phase in ['Recovery', 'Overheat', 'Stagflation', 'Reflation']:
        count = phase_counts.get(phase, 0)
        stats.append({
            'phase': phase,
            'months': count,
            'pct': count / total * 100 if total > 0 else 0
        })

    return pd.DataFrame(stats)


def plot_regime_timeline(phases, output_path):
    """Plot the regime timeline with color coding."""
    fig, ax = plt.subplots(figsize=(16, 4))

    colors = {
        'Recovery': '#90EE90',     # Light green
        'Overheat': '#FFB6C1',     # Light pink
        'Stagflation': '#FFA07A',  # Light salmon
        'Reflation': '#87CEEB',    # Light blue
    }

    phases_clean = phases['phase'].dropna()

    for i, (date, phase) in enumerate(phases_clean.items()):
        ax.axvspan(date, date + pd.DateOffset(months=1),
                   color=colors.get(phase, 'gray'), alpha=0.7)

    ax.set_xlim(phases_clean.index[0], phases_clean.index[-1])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Regime')
    ax.set_title('Investment Clock Phases (Orders/Inv + PPI)\n1992-2025', fontsize=14)

    # Legend
    patches = [mpatches.Patch(color=colors[p], label=p) for p in colors]
    ax.legend(handles=patches, loc='upper right', ncol=4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved regime timeline to {output_path}")


def plot_sector_heatmap(ranked_df, output_path, lag=1):
    """Create heatmap of sector performance by phase.

    Args:
        ranked_df: DataFrame with sector performance data
        output_path: Path to save the heatmap
        lag: Implementation lag (0 = Control, 1 = Optimal)
    """
    # Pivot to create matrix
    pivot = ranked_df.pivot(index='sector', columns='phase', values='ann_return')
    pivot = pivot[['Recovery', 'Overheat', 'Stagflation', 'Reflation']]

    # Sort by average performance
    pivot['avg'] = pivot.mean(axis=1)
    pivot = pivot.sort_values('avg', ascending=False)
    pivot = pivot.drop('avg', axis=1)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto',
                   vmin=-15, vmax=25)

    # Labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=12)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=11)

    # Add values
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            color = 'white' if abs(val) > 10 else 'black'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold')

    # Dynamic title based on lag
    lag_label = f"Lag={lag} ({'Control' if lag == 0 else 'Optimal' if lag == 1 else f'{lag}-month'})"
    ax.set_title(f'Sector Annualized Returns by Investment Clock Phase\n'
                 f'(Orders/Inv + PPI, 1992-2025, {lag_label})', fontsize=14)

    plt.colorbar(im, ax=ax, label='Annualized Return (%)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved heatmap to {output_path}")


def plot_best_sectors_by_phase(ranked_df, output_path):
    """Bar chart of top 3 sectors per phase."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors_theory = '#2E8B57'  # Sea green for theory picks
    colors_other = '#4682B4'   # Steel blue for others

    for idx, phase in enumerate(['Recovery', 'Overheat', 'Stagflation', 'Reflation']):
        ax = axes[idx]
        phase_data = ranked_df[ranked_df['phase'] == phase].head(6)

        colors = [colors_theory if row['is_theory'] else colors_other
                  for _, row in phase_data.iterrows()]

        bars = ax.barh(range(len(phase_data)), phase_data['ann_return'].values,
                       color=colors, edgecolor='black', linewidth=0.5)

        ax.set_yticks(range(len(phase_data)))
        ax.set_yticklabels(phase_data['sector'].values)
        ax.invert_yaxis()
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('Annualized Return (%)')
        ax.set_title(f'{phase}\n(Theory: {", ".join(THEORY_SECTORS[phase][:2])}...)',
                     fontsize=12)
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, phase_data['ann_return'].values):
            xpos = val + 0.5 if val >= 0 else val - 0.5
            ha = 'left' if val >= 0 else 'right'
            ax.text(xpos, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                    va='center', ha=ha, fontsize=9)

    # Legend
    theory_patch = mpatches.Patch(color=colors_theory, label='Theory Pick')
    other_patch = mpatches.Patch(color=colors_other, label='Other Sector')
    fig.legend(handles=[theory_patch, other_patch], loc='upper center',
               ncol=2, bbox_to_anchor=(0.5, 0.02))

    plt.suptitle('Top 6 Sectors by Investment Clock Phase\n(1992-2025, 1-month signal lag)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved bar chart to {output_path}")


def analyze_lag_sensitivity(phases, returns, lags=[0, 1, 2, 3]):
    """
    Compare sector performance and theory validation across different signal lags.

    Args:
        phases: DataFrame with 'phase' column
        returns: DataFrame with sector returns
        lags: List of lags to test (0 = no lag, 1 = 1-month lag, etc.)

    Returns:
        DataFrame with theory advantage and key metrics for each lag
    """
    results = []

    for lag in lags:
        lag_results = analyze_sector_performance(phases, returns, lag=lag)
        if len(lag_results) == 0:
            continue

        ranked = rank_sectors_by_phase(lag_results)

        # Calculate overall statistics
        for phase in ['Recovery', 'Overheat', 'Stagflation', 'Reflation']:
            phase_data = ranked[ranked['phase'] == phase]
            theory = phase_data[phase_data['is_theory']]
            other = phase_data[~phase_data['is_theory']]

            if len(theory) > 0 and len(other) > 0:
                theory_avg = theory['ann_return'].mean()
                other_avg = other['ann_return'].mean()
                advantage = theory_avg - other_avg
                best_theory_rank = theory['rank'].min()

                results.append({
                    'lag': lag,
                    'phase': phase,
                    'theory_avg_return': theory_avg,
                    'other_avg_return': other_avg,
                    'theory_advantage': advantage,
                    'best_theory_rank': best_theory_rank,
                    'months': phase_data['months'].iloc[0]
                })

    return pd.DataFrame(results)


def print_lag_comparison(lag_df):
    """Print formatted comparison of lag=0 (control) vs lag=1 (optimal)."""
    print("\n" + "=" * 90)
    print("LAG SENSITIVITY ANALYSIS: CONTROL (lag=0) vs OPTIMAL (lag=1)")
    print("=" * 90)

    print("\nTheory Advantage by Phase and Lag:")
    print("-" * 90)
    print(f"{'Phase':<15} {'Lag=0 (Control)':>20} {'Lag=1 (Optimal)':>20} {'Lag=2':>15} {'Lag=3':>15}")
    print("-" * 90)

    for phase in ['Recovery', 'Overheat', 'Stagflation', 'Reflation']:
        phase_data = lag_df[lag_df['phase'] == phase]
        row_str = f"{phase:<15}"

        for lag in [0, 1, 2, 3]:
            lag_row = phase_data[phase_data['lag'] == lag]
            if len(lag_row) > 0:
                adv = lag_row['theory_advantage'].values[0]
                row_str += f" {adv:>+14.1f}%"
            else:
                row_str += f" {'N/A':>15}"

        print(row_str)

    # Summary statistics
    print("\n" + "-" * 90)
    print("SUMMARY BY LAG:")
    print("-" * 90)

    for lag in [0, 1, 2, 3]:
        lag_data = lag_df[lag_df['lag'] == lag]
        if len(lag_data) > 0:
            avg_advantage = lag_data['theory_advantage'].mean()
            avg_rank = lag_data['best_theory_rank'].mean()
            print(f"  Lag {lag}: Avg Theory Advantage = {avg_advantage:+.1f}%, "
                  f"Avg Best Theory Rank = {avg_rank:.1f}")

    print("\n" + "-" * 90)

    # Compare lag=0 vs lag=1
    lag0_data = lag_df[lag_df['lag'] == 0]
    lag1_data = lag_df[lag_df['lag'] == 1]

    if len(lag0_data) > 0 and len(lag1_data) > 0:
        lag0_avg = lag0_data['theory_advantage'].mean()
        lag1_avg = lag1_data['theory_advantage'].mean()
        diff = lag1_avg - lag0_avg

        print(f"\nCONCLUSION:")
        print(f"  Control (lag=0) avg theory advantage: {lag0_avg:+.1f}%")
        print(f"  Optimal (lag=1) avg theory advantage: {lag1_avg:+.1f}%")
        print(f"  Difference (lag=1 - lag=0):           {diff:+.1f}%")

        if abs(diff) < 1.0:
            print(f"\n  → Results are SIMILAR. Lag=1 is recommended for realistic implementation")
            print(f"    without sacrificing significant predictive power.")
        elif diff > 0:
            print(f"\n  → Lag=1 OUTPERFORMS lag=0 by {diff:.1f}%. This is unexpected if indicators")
            print(f"    have look-ahead bias at lag=0.")
        else:
            print(f"\n  → Lag=0 OUTPERFORMS lag=1 by {-diff:.1f}%. Using lag=1 sacrifices")
            print(f"    {-diff:.1f}% of theory advantage for realistic implementation.")

    return lag0_data, lag1_data


def plot_dual_heatmap(ranked_lag0, ranked_lag1, output_path):
    """Create side-by-side heatmaps for Lag=0 (Control) and Lag=1 (Optimal)."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    for ax, ranked, title in zip(axes, [ranked_lag0, ranked_lag1],
                                  ['Lag=0 (Control)', 'Lag=1 (Optimal)']):
        pivot = ranked.pivot(index='sector', columns='phase', values='ann_return')
        pivot = pivot[['Recovery', 'Overheat', 'Stagflation', 'Reflation']]

        # Sort by average performance
        pivot['avg'] = pivot.mean(axis=1)
        pivot = pivot.sort_values('avg', ascending=False)
        pivot = pivot.drop('avg', axis=1)

        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto',
                       vmin=-15, vmax=35)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=11)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=10)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                color = 'white' if abs(val) > 12 else 'black'
                ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                        fontsize=9, color=color, fontweight='bold')

        ax.set_title(title, fontsize=14, fontweight='bold')

    plt.suptitle('Sector Performance by Phase: Control vs Optimal Lag\n'
                 '(Orders/Inv + PPI, 1992-2025)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved dual heatmap to {output_path}")


def main():
    print("=" * 80)
    print("SECTOR PERFORMANCE BY INVESTMENT CLOCK REGIME")
    print("First Principles Analysis using Orders/Inv Ratio + PPI")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    indicators = load_indicators()
    ff_returns = load_ff_industries()

    print(f"  Indicators: {indicators.index[0].strftime('%Y-%m')} to {indicators.index[-1].strftime('%Y-%m')}")
    print(f"  FF Industries: {ff_returns.index[0].strftime('%Y-%m')} to {ff_returns.index[-1].strftime('%Y-%m')}")

    # Generate phases
    print("\nGenerating Investment Clock phases...")
    phases = generate_phases(indicators)

    # Phase statistics
    phase_stats = calculate_phase_statistics(phases)
    print("\nPhase Distribution (1992-2025):")
    print("-" * 40)
    for _, row in phase_stats.iterrows():
        print(f"  {row['phase']:<12}: {int(row['months']):>4} months ({row['pct']:.1f}%)")

    # Analyze sector performance for BOTH lags
    print("\n" + "=" * 80)
    print("SECTOR PERFORMANCE BY PHASE (Fama-French 12 Industries)")
    print("Comparing Lag=0 (Control) vs Lag=1 (Optimal)")
    print("=" * 80)

    results_lag0 = analyze_sector_performance(phases, ff_returns, lag=0)
    ranked_lag0 = rank_sectors_by_phase(results_lag0)

    results_lag1 = analyze_sector_performance(phases, ff_returns, lag=1)
    ranked_lag1 = rank_sectors_by_phase(results_lag1)

    # SIDE-BY-SIDE SECTOR PERFORMANCE BY PHASE
    for phase in ['Recovery', 'Overheat', 'Stagflation', 'Reflation']:
        phase_data_lag0 = ranked_lag0[ranked_lag0['phase'] == phase]
        phase_data_lag1 = ranked_lag1[ranked_lag1['phase'] == phase]
        theory_picks = THEORY_SECTORS[phase]

        print(f"\n{'=' * 100}")
        print(f"PHASE: {phase.upper()}")
        print(f"Theory recommends: {', '.join(theory_picks)}")
        print("=" * 100)

        # Header for side-by-side
        print(f"\n{'':^50} | {'':^50}")
        print(f"{'LAG=0 (CONTROL)':^50} | {'LAG=1 (OPTIMAL)':^50}")
        print("-" * 100)
        print(f"{'Rk':>2} {'Sector':<20} {'Ret':>7} {'Shrp':>5} {'Th':>3} | "
              f"{'Rk':>2} {'Sector':<20} {'Ret':>7} {'Shrp':>5} {'Th':>3}")
        print("-" * 100)

        # Get top 6 for each lag
        rows_lag0 = list(phase_data_lag0.head(6).iterrows())
        rows_lag1 = list(phase_data_lag1.head(6).iterrows())

        for i in range(6):
            # Lag=0 data
            if i < len(rows_lag0):
                _, r0 = rows_lag0[i]
                th0 = '✓' if r0['is_theory'] else ''
                left = f"{int(r0['rank']):>2} {r0['sector']:<20} {r0['ann_return']:>+6.1f}% {r0['sharpe']:>5.2f} {th0:>3}"
            else:
                left = " " * 48

            # Lag=1 data
            if i < len(rows_lag1):
                _, r1 = rows_lag1[i]
                th1 = '✓' if r1['is_theory'] else ''
                right = f"{int(r1['rank']):>2} {r1['sector']:<20} {r1['ann_return']:>+6.1f}% {r1['sharpe']:>5.2f} {th1:>3}"
            else:
                right = ""

            print(f"{left} | {right}")

    # SIDE-BY-SIDE THEORY VALIDATION SUMMARY
    print("\n" + "=" * 100)
    print("THEORY VALIDATION SUMMARY: LAG=0 (Control) vs LAG=1 (Optimal)")
    print("=" * 100)

    print(f"\n{'':^50} | {'':^50}")
    print(f"{'LAG=0 (CONTROL)':^50} | {'LAG=1 (OPTIMAL)':^50}")
    print("-" * 100)
    print(f"{'Phase':<12} {'ThAvg':>8} {'OthAvg':>8} {'Adv':>8} {'Rk':>4} | "
          f"{'Phase':<12} {'ThAvg':>8} {'OthAvg':>8} {'Adv':>8} {'Rk':>4}")
    print("-" * 100)

    for phase in ['Recovery', 'Overheat', 'Stagflation', 'Reflation']:
        # Lag=0
        pd0 = ranked_lag0[ranked_lag0['phase'] == phase]
        th0 = pd0[pd0['is_theory']]
        ot0 = pd0[~pd0['is_theory']]
        if len(th0) > 0 and len(ot0) > 0:
            th_avg0 = th0['ann_return'].mean()
            ot_avg0 = ot0['ann_return'].mean()
            adv0 = th_avg0 - ot_avg0
            rk0 = int(th0['rank'].min())
            left = f"{phase:<12} {th_avg0:>+7.1f}% {ot_avg0:>+7.1f}% {adv0:>+7.1f}% {rk0:>4}"
        else:
            left = f"{phase:<12} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>4}"

        # Lag=1
        pd1 = ranked_lag1[ranked_lag1['phase'] == phase]
        th1 = pd1[pd1['is_theory']]
        ot1 = pd1[~pd1['is_theory']]
        if len(th1) > 0 and len(ot1) > 0:
            th_avg1 = th1['ann_return'].mean()
            ot_avg1 = ot1['ann_return'].mean()
            adv1 = th_avg1 - ot_avg1
            rk1 = int(th1['rank'].min())
            right = f"{phase:<12} {th_avg1:>+7.1f}% {ot_avg1:>+7.1f}% {adv1:>+7.1f}% {rk1:>4}"
        else:
            right = f"{phase:<12} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>4}"

        print(f"{left} | {right}")

    # LAG SENSITIVITY ANALYSIS (with more lags)
    print("\n" + "=" * 100)
    print("LAG SENSITIVITY ANALYSIS")
    print("Comparing Control (lag=0) vs Optimal (lag=1) vs Conservative (lag=2, 3)")
    print("=" * 100)

    lag_sensitivity = analyze_lag_sensitivity(phases, ff_returns, lags=[0, 1, 2, 3])
    lag0_data, lag1_data = print_lag_comparison(lag_sensitivity)

    # Save lag sensitivity results
    lag_sensitivity.to_csv(os.path.join(DATA_DIR, 'lag_sensitivity_results.csv'), index=False)
    print(f"\n✓ Saved lag sensitivity results to data/lag_sensitivity_results.csv")

    # COMPARATIVE ANALYSIS
    print("\n" + "=" * 100)
    print("COMPARATIVE ANALYSIS: CONTROL (Lag=0) vs OPTIMAL (Lag=1)")
    print("=" * 100)

    # Calculate differences
    print("\nPer-Phase Comparison:")
    print("-" * 80)
    print(f"{'Phase':<15} {'Lag=0 Adv':>12} {'Lag=1 Adv':>12} {'Difference':>12} {'Verdict':>20}")
    print("-" * 80)

    verdicts = []
    for phase in ['Recovery', 'Overheat', 'Stagflation', 'Reflation']:
        pd0 = ranked_lag0[ranked_lag0['phase'] == phase]
        pd1 = ranked_lag1[ranked_lag1['phase'] == phase]

        th0 = pd0[pd0['is_theory']]['ann_return'].mean() if len(pd0[pd0['is_theory']]) > 0 else 0
        ot0 = pd0[~pd0['is_theory']]['ann_return'].mean() if len(pd0[~pd0['is_theory']]) > 0 else 0
        adv0 = th0 - ot0

        th1 = pd1[pd1['is_theory']]['ann_return'].mean() if len(pd1[pd1['is_theory']]) > 0 else 0
        ot1 = pd1[~pd1['is_theory']]['ann_return'].mean() if len(pd1[~pd1['is_theory']]) > 0 else 0
        adv1 = th1 - ot1

        diff = adv1 - adv0
        if diff > 2:
            verdict = "Lag=1 BETTER"
        elif diff < -2:
            verdict = "Lag=0 BETTER"
        else:
            verdict = "SIMILAR"

        verdicts.append((phase, adv0, adv1, diff, verdict))
        print(f"{phase:<15} {adv0:>+11.1f}% {adv1:>+11.1f}% {diff:>+11.1f}% {verdict:>20}")

    # Overall verdict
    avg_adv0 = np.mean([v[1] for v in verdicts])
    avg_adv1 = np.mean([v[2] for v in verdicts])
    overall_diff = avg_adv1 - avg_adv0

    print("-" * 80)
    print(f"{'AVERAGE':<15} {avg_adv0:>+11.1f}% {avg_adv1:>+11.1f}% {overall_diff:>+11.1f}%")

    print("\n" + "-" * 80)
    print("FINAL VERDICT:")
    print("-" * 80)
    if abs(overall_diff) < 1.5:
        print(f"  Results are SIMILAR (difference = {overall_diff:+.1f}%)")
        print(f"  → Lag=1 is RECOMMENDED for realistic implementation")
        print(f"    without sacrificing significant predictive power.")
    elif overall_diff > 0:
        print(f"  Lag=1 OUTPERFORMS Lag=0 by {overall_diff:.1f}%")
        print(f"  → Lag=1 is OPTIMAL and also realistic for implementation.")
    else:
        print(f"  Lag=0 OUTPERFORMS Lag=1 by {-overall_diff:.1f}%")
        print(f"  → Using Lag=1 sacrifices {-overall_diff:.1f}% advantage for realism.")
        print(f"    This is acceptable for practical implementation.")

    # Create visualizations
    print("\n" + "=" * 100)
    print("CREATING VISUALIZATIONS")
    print("=" * 100)

    plot_regime_timeline(phases, os.path.join(DATA_DIR, 'investment_clock_regimes.png'))
    plot_sector_heatmap(ranked_lag1, os.path.join(DATA_DIR, 'sector_phase_heatmap_lag1.png'), lag=1)
    plot_sector_heatmap(ranked_lag0, os.path.join(DATA_DIR, 'sector_phase_heatmap_lag0.png'), lag=0)
    plot_dual_heatmap(ranked_lag0, ranked_lag1, os.path.join(DATA_DIR, 'sector_phase_heatmap_comparison.png'))
    plot_best_sectors_by_phase(ranked_lag1, os.path.join(DATA_DIR, 'sector_phase_barchart.png'))

    # Save results
    ranked_lag0['lag'] = 0
    ranked_lag1['lag'] = 1
    combined_results = pd.concat([ranked_lag0, ranked_lag1], ignore_index=True)
    combined_results.to_csv(os.path.join(DATA_DIR, 'sector_phase_results.csv'), index=False)
    phases.to_parquet(os.path.join(DATA_DIR, 'investment_clock_phases.parquet'))
    print(f"\n✓ Saved results to data/sector_phase_results.csv (includes both lags)")
    print(f"✓ Saved phases to data/investment_clock_phases.parquet")

    # Summary
    print("\n" + "=" * 100)
    print("KEY FINDINGS")
    print("=" * 100)

    print("""
COMPARISON: Lag=0 (Control) vs Lag=1 (Optimal)
""")

    for phase in ['Recovery', 'Overheat', 'Stagflation', 'Reflation']:
        pd0 = ranked_lag0[ranked_lag0['phase'] == phase].head(1)
        pd1 = ranked_lag1[ranked_lag1['phase'] == phase].head(1)

        if len(pd0) > 0 and len(pd1) > 0:
            top0 = f"{pd0.iloc[0]['sector']} ({pd0.iloc[0]['ann_return']:+.1f}%)"
            top1 = f"{pd1.iloc[0]['sector']} ({pd1.iloc[0]['ann_return']:+.1f}%)"
            print(f"  {phase:<12}: Lag=0: {top0:<30} | Lag=1: {top1}")

    return ranked_lag0, ranked_lag1, phases


if __name__ == '__main__':
    ranked_lag0, ranked_lag1, phases = main()
