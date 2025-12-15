# RLIC Enhancement Project - Claude Instructions

## Project Overview
Research project to enhance the Royal London Investment Clock (RLIC) by:
1. Replacing lagging indicators with leading indicators
2. Using ML for regime detection and transition smoothing
3. Proper backtesting with signal lag to avoid look-ahead bias

## Key Concepts

### Investment Clock Phases
- **Recovery** (Growth Rising, Inflation Falling): Favor stocks
- **Overheat** (Growth Rising, Inflation Rising): Favor commodities
- **Stagflation** (Growth Falling, Inflation Rising): Favor cash/defensives
- **Reflation** (Growth Falling, Inflation Falling): Favor bonds

### Best Indicators Found
- **Growth**: Orders/Inventories Ratio (MoM direction)
- **Inflation**: PPI (MoM direction)
- Together achieve 96.8% classification rate vs 66% benchmark

### Recommended ML Approach
- **HMM with Supervised Initialization** - Best Sharpe (0.62), smoothest transitions
- Use 2D composites (Growth + Inflation) as features
- Initialize HMM states from rule-based labels, let EM refine

## Technical Notes

### yfinance API Change (2024-2025)
```python
# Handle both old and new column formats
if 'Adj Close' in data.columns:
    price = data['Adj Close']
elif isinstance(data.columns, pd.MultiIndex):
    price = data[('Adj Close', ticker)]
```

### Feature Engineering Best Practices
- Use `min_valid_ratio=0.7` to filter columns with too many NaNs
- Replace inf values before scaling: `X.replace([np.inf, -np.inf], np.nan)`
- Direction features (+1/-1) work better than continuous for Investment Clock

### Backtesting
- Always use signal lag (minimum 1 month) to avoid look-ahead bias
- Walk-forward CV with purge gap (3 months) for financial ML
- Asset returns calculated as monthly pct_change of adjusted close

## Project Structure
```
rlic/
├── data/           # Parquet files with indicators and prices
├── docs/           # Research documentation (01-09)
├── script/         # Executable analysis scripts
├── src/ml/         # ML modules (feature_engineering, regime_detection, etc.)
└── temp/           # Temporary/experimental files
```

## Common Commands
```bash
# Run ML experiment
python3 script/ml_improved_experiment.py

# Run backtest with lag sensitivity
python3 script/backtest_investment_clock.py

# Fetch fresh data
python3 script/fetch_data.py
```
