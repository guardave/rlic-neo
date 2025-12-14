# Machine Learning Strategy for RLIC Enhancement

## Executive Summary

Based on research of recent academic literature and industry practices (2024-2025), this document outlines the recommended ML strategy for enhancing the Royal London Investment Clock.

---

## 1. Problem Formulation

### Two Complementary Approaches

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **Supervised Classification** | Predict discrete phases (Recovery, Overheat, Stagflation, Reflation) | Interpretable, aligns with Investment Clock theory | Requires labeled ground truth |
| **Unsupervised Regime Detection** | Let data discover natural market regimes | No labels needed, may find novel regimes | Less interpretable, may not align with theory |

### Recommended: Hybrid Approach

1. **Use unsupervised methods** (GMM, HMM) to discover data-driven regimes
2. **Map discovered regimes** to Investment Clock phases based on characteristics
3. **Compare performance** of ML regimes vs traditional rule-based phases

---

## 2. ML Model Options

### 2.1 Unsupervised Models (Regime Discovery)

| Model | Description | Best For | Python Library |
|-------|-------------|----------|----------------|
| **Gaussian Mixture Model (GMM)** | Clusters data using Gaussian distributions | Cross-sectional regime detection | `sklearn.mixture.GaussianMixture` |
| **Hidden Markov Model (HMM)** | Adds temporal transitions between states | Time series with regime persistence | `hmmlearn.GaussianHMM` |
| **K-Means Clustering** | Simple distance-based clustering | Baseline, interpretable | `sklearn.cluster.KMeans` |
| **Markov Regime Switching** | Statistical regime switching model | Economic time series | `statsmodels.tsa.regime_switching.MarkovRegression` |

**Recommendation**: Start with **HMM** (captures regime persistence) and **GMM** (simpler baseline).

### 2.2 Supervised Models (Phase Prediction)

| Model | Description | Best For | Python Library |
|-------|-------------|----------|----------------|
| **Random Forest** | Ensemble of decision trees | Feature importance, non-linear | `sklearn.ensemble.RandomForestClassifier` |
| **XGBoost/LightGBM** | Gradient boosted trees | Best performance, handles imbalance | `xgboost`, `lightgbm` |
| **Logistic Regression** | Linear classification | Baseline, interpretable | `sklearn.linear_model.LogisticRegression` |
| **LSTM/GRU** | Recurrent neural networks | Sequential patterns | `tensorflow.keras` |

**Recommendation**: Start with **XGBoost** (best performance) and **Random Forest** (interpretability).

### 2.3 End-to-End Models (Direct Asset Allocation)

| Model | Description | Reference |
|-------|-------------|-----------|
| **Reinforcement Learning** | Learn optimal allocation policy | Direct policy optimization |
| **Multi-task Learning** | Jointly predict regime + optimal weights | End-to-end optimization |

**Recommendation**: Advanced stage - pursue after validating regime detection.

---

## 3. Feature Engineering

### 3.1 Economic Indicators (from our analysis)

**Growth Indicators** (14 features):
- Orders/Inv Ratio (YoY, MoM) - Best performer
- CFNAI, CFNAI 3MA
- Yield Curve (10Y-3M, 10Y-2Y)
- LEI (3M, 6M changes)
- Initial Claims YoY (inverted)
- Building Permits YoY
- Capacity Utilization
- OECD CLI

**Inflation Indicators** (19 features):
- PPI (YoY, MoM, 3M annualized) - Best performer
- CPI (YoY, Core YoY)
- Breakeven Inflation (10Y, 5Y)
- M2 (YoY, Lag12, Lag18)
- Oil, Commodity Index (YoY, MoM)
- Import Prices YoY
- Wage Growth YoY

### 3.2 Market-Based Features

| Feature | Description | Rationale |
|---------|-------------|-----------|
| VIX | Volatility index | Risk sentiment |
| Credit Spreads | HY - IG spread | Credit conditions |
| Term Spread | 10Y - 2Y | Growth expectations |
| Dollar Index | USD strength | Global conditions |
| S&P 500 Momentum | 6M/12M returns | Market trend |

### 3.3 Feature Transformations

```python
# Recommended transformations
features = {
    'raw': value,
    'yoy': value.pct_change(12),
    'mom': value.pct_change(1),
    'zscore': (value - value.rolling(60).mean()) / value.rolling(60).std(),
    'momentum': value / value.rolling(12).mean() - 1,
    'direction': np.sign(value.diff(3)),
    'acceleration': value.diff(1).diff(1),
}
```

### 3.4 Dimensionality Reduction

Based on [arXiv research](https://arxiv.org/html/2503.11499v2), use **PCA** to reduce 127 FRED-MD variables to ~61 components (95% variance explained).

---

## 4. Target Variable Design

### Option A: Traditional 4-Phase Classification
```python
# Target: 4 classes
targets = ['Recovery', 'Overheat', 'Stagflation', 'Reflation']
```

### Option B: Binary Growth/Inflation Signals
```python
# Two separate targets
growth_target = [1, -1]  # Rising, Falling
inflation_target = [1, -1]  # Rising, Falling
```

### Option C: Data-Driven Regimes
```python
# Let model discover optimal number of regimes
n_regimes = optimize_via_bic(data)  # Typically 3-5
```

### Option D: Asset Return Prediction (Direct)
```python
# Predict next-month returns directly
target = asset_returns.shift(-1)  # Forward returns
```

**Recommendation**: Start with **Option B** (predicting growth/inflation separately), then combine.

---

## 5. Training & Validation Strategy

### 5.1 The Overfitting Problem

Traditional cross-validation fails for financial time series because:
1. **Temporal leakage**: Future data influences past predictions
2. **Regime persistence**: Adjacent samples are highly correlated
3. **Non-stationarity**: Relationships change over time

### 5.2 Recommended: Walk-Forward Validation with Purging

Based on [QuantInsti](https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/) and [SSRN research](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4686376):

```
Timeline: |----Train----|--Purge--|--Test--|

Walk-Forward:
  Fold 1: [1990-2000] train → [2000-2001] purge → [2001-2002] test
  Fold 2: [1990-2002] train → [2002-2003] purge → [2003-2004] test
  Fold 3: [1990-2004] train → [2004-2005] purge → [2005-2006] test
  ...
```

**Key Parameters**:
- **Training window**: 48-120 months (4-10 years)
- **Purge gap**: 3-6 months (avoid leakage from overlapping indicators)
- **Test window**: 12 months (1 year)
- **Expanding vs Rolling**: Expanding window generally preferred

### 5.3 Combinatorial Purged Cross-Validation (CPCV)

For more robust testing, use CPCV from [Advances in Financial ML](https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/):

```python
from sklearn.model_selection import TimeSeriesSplit

class PurgedWalkForwardCV:
    def __init__(self, n_splits=5, purge_gap=6):
        self.n_splits = n_splits
        self.purge_gap = purge_gap

    def split(self, X, y=None):
        n_samples = len(X)
        test_size = n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            train_end = test_size * (i + 1)
            test_start = train_end + self.purge_gap
            test_end = test_start + test_size

            train_idx = list(range(0, train_end))
            test_idx = list(range(test_start, min(test_end, n_samples)))

            yield train_idx, test_idx
```

---

## 6. Evaluation Metrics

### 6.1 Classification Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Accuracy | Correct / Total | > 50% (random baseline for 4 classes = 25%) |
| F1-Score (macro) | Harmonic mean of P/R | > 0.4 |
| Cohen's Kappa | Agreement beyond chance | > 0.3 |
| Log Loss | Probabilistic accuracy | Lower is better |

### 6.2 Financial Metrics (Most Important)

| Metric | Description | Target |
|--------|-------------|--------|
| **Sharpe Ratio** | Risk-adjusted return | > 0.5 (> benchmark) |
| **Sortino Ratio** | Downside risk-adjusted | > 0.7 |
| **Max Drawdown** | Worst peak-to-trough | < 20% |
| **Win Rate** | % positive months | > 55% |
| **Information Ratio** | Alpha / Tracking Error | > 0.3 |

### 6.3 Overfitting Detection

| Metric | Description | Warning Sign |
|--------|-------------|--------------|
| **Train-Test Gap** | Performance difference | > 30% degradation |
| **Deflated Sharpe** | Sharpe adjusted for trials | Significantly lower |
| **PBO** | Probability of Backtest Overfitting | > 50% |

---

## 7. Experiment Design

### Phase 1: Baseline Comparison

```
Experiment 1.1: Rule-based vs Unsupervised
├── Baseline: Orders/Inv MoM + PPI MoM (current best)
├── GMM with 4 clusters
├── HMM with 4 states
└── K-Means with 4 clusters

Metrics: Classification alignment, Sharpe ratio, Max DD
```

### Phase 2: Feature Selection

```
Experiment 2.1: Feature Importance
├── All 33 indicators
├── Top 10 by mutual information
├── Top 10 by Random Forest importance
└── PCA components (95% variance)

Experiment 2.2: Lag Optimization
├── Features at t-1 (1 month lag)
├── Features at t-3 (3 month lag)
├── Features at t-6 (6 month lag)
└── Multiple lags combined
```

### Phase 3: Model Selection

```
Experiment 3.1: Supervised Classification
├── Logistic Regression (baseline)
├── Random Forest
├── XGBoost
└── LightGBM

Experiment 3.2: Unsupervised + Supervised Hybrid
├── GMM clusters → XGBoost refinement
├── HMM states → regime characteristics
```

### Phase 4: Allocation Optimization

```
Experiment 4.1: Static vs Dynamic Allocation
├── Static: Fixed weights per phase
├── Dynamic: ML-predicted weights
└── Risk-parity within phase

Experiment 4.2: Sector Rotation
├── Phase → Top 3 sectors
├── ML-predicted sector weights
```

---

## 8. Implementation Roadmap

### Stage 1: Data Pipeline (Week 1)
- [ ] Consolidate all indicators into feature matrix
- [ ] Implement feature engineering pipeline
- [ ] Create train/test split with purging

### Stage 2: Unsupervised Baselines (Week 2)
- [ ] Implement GMM regime detection
- [ ] Implement HMM with hmmlearn
- [ ] Compare discovered regimes to Investment Clock phases

### Stage 3: Supervised Models (Week 3)
- [ ] Train XGBoost/Random Forest classifiers
- [ ] Feature importance analysis
- [ ] Hyperparameter tuning with walk-forward CV

### Stage 4: Backtesting Framework (Week 4)
- [ ] Integrate ML predictions with backtest engine
- [ ] Compare ML strategy vs rule-based
- [ ] Robustness testing across time periods

### Stage 5: Optimization (Week 5-6)
- [ ] Sector allocation optimization
- [ ] Position sizing based on confidence
- [ ] Transaction cost modeling

---

## 9. Key References

### Academic Papers
- [Tactical Asset Allocation with Macroeconomic Regime Detection](https://arxiv.org/html/2503.11499v2) - arXiv 2025
- [Explainable ML for Regime-Based Asset Allocation](https://ieeexplore.ieee.org/document/9378332) - IEEE 2021
- [Machine Learning Approach to Risk-Based Asset Allocation](https://www.nature.com/articles/s41598-025-26337-x) - Nature 2025

### Industry Resources
- [Two Sigma: ML Approach to Regime Modeling](https://www.twosigma.com/articles/a-machine-learning-approach-to-regime-modeling/)
- [State Street: Decoding Market Regimes with ML](https://www.ssga.com/library-content/assets/pdf/global/pc/2025/decoding-market-regimes-with-machine-learning.pdf)

### Implementation Guides
- [QuantStart: HMM Market Regime Detection](https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/)
- [QuantInsti: Walk-Forward Optimization](https://blog.quantinsti.com/walk-forward-optimization-introduction/)
- [Cross Validation: Purging & Embargoing](https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/)

### Python Libraries
- `hmmlearn` - Hidden Markov Models
- `sklearn.mixture.GaussianMixture` - GMM
- `statsmodels.tsa.regime_switching` - Markov Switching
- `xgboost`, `lightgbm` - Gradient Boosting
- `mlfinlab` - Financial ML utilities (purged CV, etc.)

---

## 10. Risk Considerations

### Model Risks
1. **Overfitting**: Mitigate with purged CV, out-of-sample testing
2. **Regime change**: Models trained on past may not work in new regimes
3. **Look-ahead bias**: Careful feature engineering with proper lags

### Implementation Risks
1. **Transaction costs**: High turnover can erode returns
2. **Slippage**: Monthly rebalancing should be manageable
3. **Capacity**: Strategy likely has good capacity for personal/institutional use

### Recommendations
- Always compare ML models to simple rule-based baseline
- Use multiple evaluation periods (not just recent history)
- Start with interpretable models before complex ones
- Focus on risk-adjusted returns, not just CAGR

---

*Document created: 2025-12-14*
*Based on research of 2024-2025 academic literature and industry practices*
