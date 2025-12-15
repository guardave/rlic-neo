# Release Notes - RLIC Enhancement Project

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
