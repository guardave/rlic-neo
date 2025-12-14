# ML Regime Detection: Experiment Findings

## Executive Summary

This document presents the findings from our ML experiments to enhance the Investment Clock regime detection. We tested three improvement approaches based on theoretical considerations:

1. **Feature Aggregation**: Reduce 49 features to 2D composites
2. **Feature Selection**: Use direction-only features matching quadrant logic
3. **Hybrid ML**: Use rule-based labels to initialize/train ML models

**Key Finding**: The **HMM with Supervised Initialization** approach delivers the best results, achieving:
- Sharpe ratio of 0.62 (vs 0.58 for rule-based)
- 40% fewer regime changes (smoother transitions)
- Economically sensible transition dynamics

## Background: Why ML Underperformed Initially

Our initial unsupervised ML experiment (GMM, HMM, K-Means on 49 features) showed poor results:
- Only 25% agreement with rule-based phases
- Sharpe ratios of 0.26-0.32 (vs 0.51 for rule-based)
- ML clustered data by statistical variance, not economic meaning

**Root Cause**: With 49 features of varying scales and meanings, clustering algorithms optimized for statistical fit rather than the Growth × Inflation framework.

## Approach 1: Feature Aggregation (2D Composites)

### Methodology

Instead of 49 individual features, we created two composite scores:

**Growth Composite** (average of normalized signals):
- Orders/Inventory ratio direction
- CFNAI 3-month moving average sign
- Yield curve sign (10Y-3M spread)
- LEI 3-month momentum sign
- Initial claims direction (inverted)
- OECD CLI deviation from 100

**Inflation Composite** (average of normalized signals):
- PPI direction (3MA vs 6MA)
- CPI YoY momentum sign
- Breakeven inflation direction
- Commodity index direction
- Oil price direction

Each component is normalized to +1/-1, and the composite is the average.

### Results

GMM on 2D composites correctly identifies the four quadrants:

| Regime | Growth | Inflation | Mapped Phase |
|--------|--------|-----------|--------------|
| 0 | -0.37 | -0.86 | Reflation |
| 1 | +0.65 | -0.06 | Recovery |
| 2 | +0.37 | +1.00 | Overheat |
| 3 | -0.10 | +0.36 | Stagflation |

**Agreement with rule-based**: 27% (low because GMM uses different clustering boundaries than the simple sign-based threshold)

**Insight**: The 2D representation captures the Investment Clock framework, but unsupervised clustering still produces different boundaries than the theoretical quadrants.

## Approach 2: Supervised Classification (Random Forest)

### Methodology

Train a Random Forest classifier using:
- **Features**: 11 direction-only features (binary +1/-1)
- **Labels**: Rule-based phase classifications (ground truth)
- **Split**: 70% train, 30% test (time-based)

### Results

**Test Accuracy: 100%**

Classification Report:
| Phase | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Recovery | 1.00 | 1.00 | 1.00 |
| Overheat | 1.00 | 1.00 | 1.00 |
| Stagflation | 1.00 | 1.00 | 1.00 |
| Reflation | 1.00 | 1.00 | 1.00 |

**Walk-Forward CV Accuracy: 95.0% (+/- 7.5%)**

**Top 5 Important Features**:
1. `orders_inv_dir`: 37.7%
2. `ppi_dir`: 29.5%
3. `commodity_dir`: 11.3%
4. `claims_dir`: 4.9%
5. `breakeven_dir`: 3.8%

### Interpretation

The Random Forest **perfectly learns the rule-based logic**. The top two features (`orders_inv_dir` and `ppi_dir`) are exactly the two indicators used in our rule-based approach, accounting for 67% of feature importance.

**Conclusion**: The rule-based approach is consistent and learnable, but supervised ML adds no value over the rules themselves (same Sharpe ratio of 0.58).

## Approach 3: HMM with Supervised Initialization (Best Approach)

### Methodology

Initialize a Hidden Markov Model using rule-based phases:
1. **Initial state distribution**: From rule-based phase frequencies
2. **Transition matrix**: From rule-based phase transitions
3. **Emission parameters**: Mean/variance per phase from rule-based assignments
4. Let EM algorithm refine parameters while preserving economic structure

### Results

**Agreement with rule-based: 61.3%**

**Regime Changes**:
- Rule-based: 78 transitions
- HMM: 47 transitions
- **Smoothing ratio: 0.60** (40% fewer regime changes)

**Learned Transition Matrix**:

|  | To Recovery | To Overheat | To Stagflation | To Reflation |
|--|-------------|-------------|----------------|--------------|
| From Recovery | 0.74 | 0.11 | 0.08 | 0.07 |
| From Overheat | 0.08 | 0.86 | 0.06 | 0.00 |
| From Stagflation | 0.04 | 0.09 | 0.76 | 0.11 |
| From Reflation | 0.05 | 0.00 | 0.16 | 0.80 |

**Expected Regime Duration (months)**:
- Recovery: 3.9 months (shortest - transition phase)
- Overheat: 7.0 months (longest - economic expansion)
- Stagflation: 4.2 months
- Reflation: 4.9 months

### Economic Interpretation

The transition matrix reveals sensible economic dynamics:

1. **Overheat is most persistent** (86% self-transition, 7 months average)
   - Economic expansions tend to be long-lasting
   - Difficult to predict the end of a boom

2. **Recovery is shortest** (74% self-transition, 3.9 months average)
   - Rapid transition from recession to growth
   - Inflation catches up quickly

3. **Clear cycle pattern**:
   - Overheat → Stagflation (6%) - classic overheating leads to inflation problems
   - Stagflation → Reflation (11%) - policy response to stagflation
   - Reflation → Stagflation (16%) - monetary stimulus can reignite inflation
   - Recovery → Overheat (11%) - growth momentum leads to overheating

4. **Overheat → Reflation is 0%**
   - Makes economic sense: you don't go from boom directly to disinflation
   - Must pass through Stagflation or Recovery first

## Performance Comparison

### Backtest Results (1-Month Signal Lag)

| Method | CAGR | Volatility | Sharpe | Max Drawdown |
|--------|------|------------|--------|--------------|
| Rule-Based | 6.8% | 8.5% | 0.58 | -12.2% |
| GMM 2D | 7.1% | 8.8% | 0.60 | -22.3% |
| Random Forest | 6.8% | 8.5% | 0.58 | -12.2% |
| **HMM Supervised** | **6.9%** | **8.1%** | **0.62** | -13.7% |
| Buy & Hold SPY | 10.2% | 14.7% | 0.60 | -50.8% |

### Analysis

1. **HMM Supervised achieves highest Sharpe (0.62)**
   - Better than rule-based (0.58)
   - Slightly better than Buy & Hold (0.60) with much lower drawdown

2. **GMM 2D has highest CAGR but worst drawdown**
   - The 22.3% drawdown makes it less attractive
   - Clustering boundaries don't align with optimal trading signals

3. **Random Forest matches rule-based exactly**
   - Confirms it's just learning the same rules
   - No incremental value from ML complexity

4. **All strategies dramatically reduce drawdown vs Buy & Hold**
   - Buy & Hold: -50.8%
   - Best strategy (Rule-Based): -12.2%
   - This is the primary value of regime-based investing

## Recommendations

### Primary Recommendation: HMM with Supervised Initialization

Use the HMM-Supervised approach for production because:

1. **Smoother transitions** - 40% fewer regime changes means lower turnover and transaction costs
2. **Probabilistic output** - Provides confidence scores for regime assignments
3. **Economically sensible** - Transition dynamics align with economic theory
4. **Best risk-adjusted returns** - Highest Sharpe ratio among all approaches

### Implementation Guidelines

```python
from src.ml.supervised_regime import HMMWithSupervisedInit
from src.ml.feature_engineering import create_composite_features, create_rule_based_targets

# Create features and labels
composites = create_composite_features(indicators)
X = composites[['growth_composite', 'inflation_composite']]
rule_labels = create_rule_based_targets(indicators)['phase']

# Initialize and fit HMM
hmm = HMMWithSupervisedInit(n_regimes=4, covariance_type='diag')
hmm.fit(X, rule_labels)

# Predict current regime
current_regime = hmm.predict(X.iloc[[-1]])
regime_probs = hmm.predict_proba(X.iloc[[-1]])
```

### When to Use Each Approach

| Use Case | Recommended Approach |
|----------|---------------------|
| Production trading | HMM Supervised |
| Quick analysis | Rule-Based |
| Understanding feature importance | Random Forest |
| Regime boundary exploration | GMM 2D |

## Files Created

| File | Purpose |
|------|---------|
| `src/ml/feature_engineering.py` | Feature creation including composites |
| `src/ml/regime_detection.py` | Unsupervised GMM/HMM/K-Means |
| `src/ml/supervised_regime.py` | Supervised classifiers and HMM with init |
| `src/ml/validation.py` | Walk-forward CV and backtesting |
| `script/ml_regime_baseline.py` | Initial unsupervised experiments |
| `script/ml_improved_experiment.py` | Improved approaches experiment |
| `data/ml_improved_results.csv` | Regime predictions from all methods |
| `data/ml_improved_metrics.csv` | Performance metrics comparison |

## Future Work

1. **Ensemble Methods**: Combine rule-based and HMM predictions for higher confidence signals

2. **Regime Probability Thresholds**: Only trade when regime probability exceeds threshold (e.g., 70%)

3. **Dynamic Allocation**: Use regime probabilities to weight allocations rather than discrete switching

4. **Online Learning**: Update HMM parameters as new data arrives

5. **Alternative Assets**: Test sector ETFs and factor exposures within each regime

## Conclusion

The ML experiments validate that:

1. **The rule-based approach is sound** - Random Forest learns it perfectly
2. **2D representation captures the framework** - Growth and Inflation composites work
3. **HMM adds value through smoothing** - Fewer regime changes, better Sharpe
4. **Economic constraints help ML** - Supervised initialization produces sensible dynamics

The HMM with Supervised Initialization is recommended as the production approach, combining the economic intuition of the Investment Clock with the statistical refinement of machine learning.
