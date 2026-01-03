# Release Notes - RLIC Enhancement Project

## Session: 2026-01-03

### Bug Fixes

#### Heatmap Title Inconsistency Fix
- **Issue**: `plot_sector_heatmap()` had hardcoded "1-month lag" in title
- **Impact**: Lag=0 heatmap incorrectly displayed "1-month lag" caption
- **Fix**: Added `lag` parameter to function with dynamic title generation
- **Result**: Now correctly shows "Lag=0 (Control)" and "Lag=1 (Optimal)"

### Files Changed
- `script/sector_regime_analysis.py`: Updated `plot_sector_heatmap()` function

### Lessons Learned

1. **Parameterize visualization titles** - When a function generates multiple variants, always pass parameters for dynamic labeling
2. **Side-by-side comparison methodology** - Control vs Optimal comparison strengthens research validity

---

## Session: 2025-12-21

### Features Added

#### Time Series Relationship Analysis Framework
- **Framework Document** (`docs/11_time_series_relationship_framework.md`)
  - Step 0: Qualitative Analysis (literature review, market interpretation, citations)
  - Steps 1-7: Quantitative analysis (correlation, lead-lag, Granger, ML, regime)
  - Step 8: Documentation and Organization (folder structure, naming conventions)
  - Example validation process for visualizations
  - Image embedding best practices

#### SPY vs RETAILIRSA Analysis (`docs/analysis_reports/spy_retailirsa_analysis.md`)
- Qualitative section with bullwhip effect literature, recession indicators
- Key finding: Contemporaneous relationship, no predictive power
- Regime analysis: Falling inventories Sharpe 0.98 vs Rising 0.52
- Citations: MIT Sloan, FRED, NetSuite, CNBC, Rosenberg Research

#### SPY vs Industrial Production Analysis (`docs/analysis_reports/spy_industrial_production_analysis.md`)
- Qualitative section with NBER "Big Four" indicators, academic research
- Key finding: Coincident indicator, no predictive power
- Regime analysis: IP Rising Sharpe 1.03 vs IP Falling 0.87 (not significant)
- Citations: Federal Reserve, Chicago Fed, Fama (1981), Hong et al., Conference Board

#### Visualization Improvements
- Regime background coloring (Green/Pink/Gray)
- Validated example selection process
- Standard naming convention: `{target}_{indicator}_{plot_type}.png`

### Key Discoveries

1. **Qualitative analysis should precede quantitative analysis**
   - Provides context for interpreting statistical results
   - Identifies known limitations upfront
   - Aligns findings with market consensus

2. **Both indicators are coincident, not leading**
   - IP moves with the economy
   - RETAILIRSA reflects current demand conditions
   - Stock markets anticipate production changes (reverse causality)

3. **Recession indicator provides strongest signal**
   - Both analyses show Sharpe -0.64 during recessions
   - More reliable than indicator direction alone

4. **Example validation is critical**
   - Must verify data matches visual before writing descriptions
   - Count regime months, verify dominant regime, check target direction

### Technical Lessons Learned

1. **Literature review sources by type**:
   - Academic: Google Scholar, SSRN (e.g., Fama, Stock & Watson)
   - Professional: Federal Reserve notes, Conference Board, IMF
   - Public: Advisor Perspectives, financial media (CNBC, Bloomberg)

2. **Image paths in nested folders**: Use `../../data/` from `docs/analysis_reports/`

3. **Document organization**: Dedicated folders for analysis reports improve maintainability

### Files Changed
- Moved: `docs/10_spy_retailirsa_analysis.md` → `docs/analysis_reports/spy_retailirsa_analysis.md`
- Moved: `docs/12_spy_industrial_production_analysis.md` → `docs/analysis_reports/spy_industrial_production_analysis.md`
- Updated: `docs/11_time_series_relationship_framework.md` (added Steps 0 and 8)
- New visualizations: 10 PNG files in `data/`

---

## Session: 2025-12-15

### Features Added

#### ML Regime Detection Framework
- **Feature Engineering Module** (`src/ml/feature_engineering.py`)
  - 49 features from growth and inflation indicators
  - Composite features (2D) for simplified regime detection
  - Direction-only features matching Investment Clock quadrant logic

- **Regime Detection Models** (`src/ml/regime_detection.py`)
  - GMM (Gaussian Mixture Model) with optimal regime selection
  - HMM (Hidden Markov Model) with transition dynamics
  - K-Means baseline clustering

- **Supervised Regime Classification** (`src/ml/supervised_regime.py`)
  - Random Forest, XGBoost, Gradient Boosting classifiers
  - HMM with supervised initialization (best performer)
  - Ensemble prediction methods

- **Validation Framework** (`src/ml/validation.py`)
  - Walk-forward cross-validation with purging gap
  - Combinatorial purged CV (Lopez de Prado methodology)
  - Time series backtester with signal lag

#### Backtesting Scripts
- `script/backtest_investment_clock.py` - Main backtest with signal lag
- `script/phase_lag_sensitivity.py` - Phase performance vs lag analysis
- `script/sector_phase_analysis.py` - Sector ETF bias by phase
- `script/ml_regime_baseline.py` - Initial unsupervised ML experiments
- `script/ml_improved_experiment.py` - Improved hybrid approaches

### Key Discoveries

1. **HMM with Supervised Initialization is the winner**
   - Sharpe: 0.62 (vs 0.58 rule-based, 0.60 Buy & Hold)
   - 40% fewer regime changes (smoother transitions)
   - Economically sensible transition dynamics

2. **Random Forest perfectly learns rule-based logic**
   - 100% test accuracy, 95% walk-forward CV accuracy
   - Top features: `orders_inv_dir` (37.7%), `ppi_dir` (29.5%)
   - Confirms rule-based approach is consistent but RF adds no incremental value

3. **2D Composites capture Investment Clock**
   - GMM correctly identifies 4 quadrants
   - But clustering boundaries differ from theoretical thresholds

4. **Transition Matrix Insights**
   - Overheat most persistent: 7 months average
   - Recovery shortest: 3.9 months (transition phase)
   - Overheat → Reflation = 0% (economically sensible)

### Technical Lessons Learned

1. **yfinance API changed** - Now returns MultiIndex columns; added handling for both formats
2. **dropna with many features** - Use `min_valid_ratio` to keep columns with sufficient data
3. **Infinite values break StandardScaler** - Always replace inf with NaN before scaling
4. **Signal lag is critical** - Without lag, backtests have look-ahead bias

### Performance Summary

| Method | Sharpe | Max Drawdown |
|--------|--------|--------------|
| Rule-Based | 0.58 | -12.2% |
| HMM Supervised | 0.62 | -13.7% |
| Buy & Hold SPY | 0.60 | -50.8% |

### Files Changed
- 12 new files added (4,473 lines)
- Committed as: `e80844a Add ML regime detection framework with backtesting infrastructure`
