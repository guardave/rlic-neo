"""
Walk-Forward Validation Framework.

Implements proper time-series cross-validation with purging for financial ML.
"""

import pandas as pd
import numpy as np
from typing import Generator, Tuple, List, Optional


class PurgedWalkForwardCV:
    """
    Walk-forward cross-validation with purging gap.

    Prevents look-ahead bias by:
    1. Training only on past data
    2. Adding purge gap between train and test to avoid leakage
    3. Expanding or rolling window approach
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_size: Optional[int] = None,
        test_size: int = 12,
        purge_gap: int = 3,
        expanding: bool = True
    ):
        """
        Initialize walk-forward CV.

        Args:
            n_splits: Number of train/test splits
            train_size: Size of training window (None = expanding)
            test_size: Size of test window (months)
            purge_gap: Gap between train and test to prevent leakage (months)
            expanding: If True, training window expands; if False, it rolls
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.purge_gap = purge_gap
        self.expanding = expanding

    def split(self, X: pd.DataFrame, y=None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices.

        Args:
            X: Feature matrix with datetime index
            y: Target (not used, for sklearn compatibility)

        Yields:
            train_indices, test_indices
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate split points
        min_train_size = self.train_size if self.train_size else 48  # Default 4 years minimum
        total_test = self.n_splits * self.test_size
        available = n_samples - min_train_size - self.purge_gap

        if available < total_test:
            # Adjust test size if not enough data
            self.test_size = max(6, available // self.n_splits)
            print(f"Adjusted test_size to {self.test_size} due to limited data")

        for i in range(self.n_splits):
            # Test period
            test_end = n_samples - i * self.test_size
            test_start = test_end - self.test_size

            if test_start <= min_train_size + self.purge_gap:
                break

            # Training period (with purge gap)
            train_end = test_start - self.purge_gap

            if self.expanding:
                train_start = 0
            else:
                train_start = max(0, train_end - self.train_size)

            train_idx = indices[train_start:train_end]
            test_idx = indices[test_start:test_end]

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of splits."""
        return self.n_splits


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV).

    More robust than simple walk-forward by testing multiple paths.
    Based on Lopez de Prado's methodology.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_splits: int = 2,
        purge_gap: int = 3,
        embargo_gap: int = 1
    ):
        """
        Initialize CPCV.

        Args:
            n_splits: Total number of folds
            n_test_splits: Number of folds used for testing in each iteration
            purge_gap: Gap before test period
            embargo_gap: Gap after test period
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_gap = purge_gap
        self.embargo_gap = embargo_gap

    def split(self, X: pd.DataFrame, y=None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate combinatorial train/test splits.

        Yields:
            train_indices, test_indices
        """
        from itertools import combinations

        n_samples = len(X)
        indices = np.arange(n_samples)

        # Divide data into n_splits groups
        fold_size = n_samples // self.n_splits
        folds = []
        for i in range(self.n_splits):
            start = i * fold_size
            end = start + fold_size if i < self.n_splits - 1 else n_samples
            folds.append(indices[start:end])

        # Generate all combinations of test folds
        for test_fold_indices in combinations(range(self.n_splits), self.n_test_splits):
            test_idx = np.concatenate([folds[i] for i in test_fold_indices])

            # Training indices: all folds not in test, with purging
            train_idx = []
            for i, fold in enumerate(folds):
                if i not in test_fold_indices:
                    # Apply purge gap: remove observations close to test periods
                    fold_clean = self._purge_fold(fold, test_idx)
                    train_idx.extend(fold_clean)

            train_idx = np.array(sorted(train_idx))
            test_idx = np.array(sorted(test_idx))

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

    def _purge_fold(self, fold_indices: np.ndarray, test_indices: np.ndarray) -> List[int]:
        """Remove observations from fold that are too close to test period."""
        test_min, test_max = test_indices.min(), test_indices.max()

        purged = []
        for idx in fold_indices:
            # Remove if within purge_gap before test or embargo_gap after test
            if idx < test_min - self.purge_gap or idx > test_max + self.embargo_gap:
                purged.append(idx)
            elif idx < test_min - self.purge_gap:
                purged.append(idx)

        return purged


class TimeSeriesBacktester:
    """
    Backtester for regime-based strategies.
    """

    def __init__(self, signal_lag: int = 1):
        """
        Initialize backtester.

        Args:
            signal_lag: Months to lag signal (1 = realistic implementation)
        """
        self.signal_lag = signal_lag
        self.results = None

    def backtest_regimes(
        self,
        regime_labels: np.ndarray,
        asset_returns: pd.DataFrame,
        regime_allocations: dict,
        index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Backtest regime-based allocation strategy.

        Args:
            regime_labels: Array of regime labels for each period
            asset_returns: DataFrame with asset returns
            regime_allocations: Dict mapping regime -> asset weights
            index: DatetimeIndex for the data

        Returns:
            DataFrame with backtest results
        """
        # Create aligned DataFrame
        results = pd.DataFrame(index=index)
        results['regime'] = regime_labels

        # Lag regimes (use regime at t-lag for allocation at t)
        results['regime_lagged'] = results['regime'].shift(self.signal_lag)

        # Calculate strategy returns
        strategy_returns = []

        for i, date in enumerate(results.index):
            if pd.isna(results.loc[date, 'regime_lagged']):
                strategy_returns.append(np.nan)
                continue

            regime = int(results.loc[date, 'regime_lagged'])

            if regime not in regime_allocations:
                strategy_returns.append(np.nan)
                continue

            weights = regime_allocations[regime]

            # Calculate weighted return
            period_return = 0
            for asset, weight in weights.items():
                if asset in asset_returns.columns and date in asset_returns.index:
                    asset_ret = asset_returns.loc[date, asset]
                    if not pd.isna(asset_ret):
                        period_return += weight * asset_ret

            strategy_returns.append(period_return)

        results['strategy_return'] = strategy_returns

        # Add benchmark (equal weight stocks)
        if 'stocks' in asset_returns.columns:
            results['benchmark_return'] = asset_returns['stocks'].reindex(index)

        self.results = results
        return results

    def calculate_metrics(self, returns: pd.Series, name: str = "Strategy") -> dict:
        """Calculate performance metrics."""
        returns = returns.dropna()

        if len(returns) == 0:
            return {}

        total_return = (1 + returns).prod() - 1
        cagr = (1 + total_return) ** (12 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(12)
        sharpe = (returns.mean() * 12 - 0.02) / volatility if volatility > 0 else 0

        # Drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        return {
            'name': name,
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': (returns > 0).mean(),
            'n_periods': len(returns)
        }


def run_walk_forward_experiment(
    model_class,
    model_params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    asset_returns: pd.DataFrame,
    regime_allocations: dict,
    cv: PurgedWalkForwardCV
) -> dict:
    """
    Run complete walk-forward experiment.

    Args:
        model_class: ML model class (must have fit/predict)
        model_params: Parameters for model
        X: Feature matrix
        y: Target variable
        asset_returns: Asset returns for backtesting
        regime_allocations: Allocation per regime
        cv: Cross-validation splitter

    Returns:
        Dict with experiment results
    """
    all_predictions = pd.Series(index=X.index, dtype=float)
    all_actuals = pd.Series(index=X.index, dtype=float)
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        print(f"Fold {fold_idx + 1}: Train {len(train_idx)} samples, Test {len(test_idx)} samples")

        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Fit model
        model = model_class(**model_params)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Store predictions
        all_predictions.iloc[test_idx] = y_pred
        all_actuals.iloc[test_idx] = y_test.values

        # Calculate fold metrics
        accuracy = (y_pred == y_test.values).mean()
        fold_metrics.append({
            'fold': fold_idx + 1,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'accuracy': accuracy
        })

    # Backtest
    backtester = TimeSeriesBacktester(signal_lag=1)
    valid_mask = all_predictions.notna()

    if valid_mask.sum() > 0:
        backtest_results = backtester.backtest_regimes(
            regime_labels=all_predictions[valid_mask].values.astype(int),
            asset_returns=asset_returns.loc[valid_mask],
            regime_allocations=regime_allocations,
            index=X.index[valid_mask]
        )

        strategy_metrics = backtester.calculate_metrics(
            backtest_results['strategy_return'], "ML Strategy"
        )
    else:
        backtest_results = None
        strategy_metrics = {}

    return {
        'predictions': all_predictions,
        'actuals': all_actuals,
        'fold_metrics': pd.DataFrame(fold_metrics),
        'backtest_results': backtest_results,
        'strategy_metrics': strategy_metrics
    }
