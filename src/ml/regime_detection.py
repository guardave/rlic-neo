"""
Unsupervised Regime Detection Models.

Implements GMM and HMM for discovering market regimes from data.
"""

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("Warning: hmmlearn not available. HMM models will not work.")


class GMMRegimeDetector:
    """
    Gaussian Mixture Model for regime detection.

    Clusters observations into distinct regimes based on feature distributions.
    """

    def __init__(self, n_regimes=4, covariance_type='full', random_state=42):
        """
        Initialize GMM regime detector.

        Args:
            n_regimes: Number of regimes to detect
            covariance_type: 'full', 'tied', 'diag', or 'spherical'
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model = None
        self.regime_characteristics = None

    def fit(self, X):
        """
        Fit GMM to data.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            self
        """
        self.model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_init=10,
            max_iter=200
        )
        self.model.fit(X)

        # Store regime characteristics
        self._compute_regime_characteristics(X)

        return self

    def predict(self, X):
        """
        Predict regime for each observation.

        Args:
            X: Feature matrix

        Returns:
            Array of regime labels
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict regime probabilities.

        Args:
            X: Feature matrix

        Returns:
            Array of regime probabilities (n_samples, n_regimes)
        """
        return self.model.predict_proba(X)

    def _compute_regime_characteristics(self, X):
        """Compute summary statistics for each regime."""
        labels = self.model.predict(X)

        self.regime_characteristics = {}
        for regime in range(self.n_regimes):
            mask = labels == regime
            regime_data = X[mask] if isinstance(X, np.ndarray) else X.values[mask]

            self.regime_characteristics[regime] = {
                'count': mask.sum(),
                'pct': mask.mean() * 100,
                'mean': regime_data.mean(axis=0),
                'std': regime_data.std(axis=0)
            }

    def select_optimal_n_regimes(self, X, min_regimes=2, max_regimes=8):
        """
        Select optimal number of regimes using BIC.

        Args:
            X: Feature matrix
            min_regimes: Minimum number of regimes to try
            max_regimes: Maximum number of regimes to try

        Returns:
            optimal_n, results_df
        """
        results = []

        for n in range(min_regimes, max_regimes + 1):
            gmm = GaussianMixture(
                n_components=n,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                n_init=5
            )
            gmm.fit(X)

            labels = gmm.predict(X)
            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(X, labels)
            else:
                silhouette = -1

            results.append({
                'n_regimes': n,
                'bic': gmm.bic(X),
                'aic': gmm.aic(X),
                'log_likelihood': gmm.score(X) * len(X),
                'silhouette': silhouette
            })

        results_df = pd.DataFrame(results)

        # Optimal by BIC (lower is better)
        optimal_n = results_df.loc[results_df['bic'].idxmin(), 'n_regimes']

        return int(optimal_n), results_df

    def get_regime_summary(self, X, feature_names=None):
        """
        Get summary of regime characteristics.

        Args:
            X: Feature matrix
            feature_names: List of feature names

        Returns:
            DataFrame with regime summaries
        """
        labels = self.model.predict(X)

        if feature_names is None:
            if hasattr(X, 'columns'):
                feature_names = X.columns.tolist()
            else:
                feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

        summaries = []
        for regime in range(self.n_regimes):
            mask = labels == regime
            regime_data = X[mask] if isinstance(X, np.ndarray) else X.values[mask]

            summary = {
                'regime': regime,
                'count': mask.sum(),
                'pct_time': mask.mean() * 100
            }

            # Add mean of each feature
            means = regime_data.mean(axis=0)
            for i, name in enumerate(feature_names[:10]):  # Limit to first 10
                summary[f'{name}_mean'] = means[i]

            summaries.append(summary)

        return pd.DataFrame(summaries)


class HMMRegimeDetector:
    """
    Hidden Markov Model for regime detection.

    Captures temporal dynamics and regime transitions.
    """

    def __init__(self, n_regimes=4, covariance_type='full', n_iter=100, random_state=42):
        """
        Initialize HMM regime detector.

        Args:
            n_regimes: Number of hidden states/regimes
            covariance_type: 'full', 'tied', 'diag', or 'spherical'
            n_iter: Maximum number of EM iterations
            random_state: Random seed
        """
        if not HMM_AVAILABLE:
            raise ImportError("hmmlearn is required for HMM. Install with: pip install hmmlearn")

        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = None
        self.transition_matrix = None

    def fit(self, X):
        """
        Fit HMM to data.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            self
        """
        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state
        )

        # HMM expects 2D array
        X_array = X.values if hasattr(X, 'values') else X
        self.model.fit(X_array)

        # Store transition matrix
        self.transition_matrix = self.model.transmat_

        return self

    def predict(self, X):
        """
        Predict most likely regime sequence (Viterbi).

        Args:
            X: Feature matrix

        Returns:
            Array of regime labels
        """
        X_array = X.values if hasattr(X, 'values') else X
        return self.model.predict(X_array)

    def predict_proba(self, X):
        """
        Predict regime probabilities at each time step.

        Args:
            X: Feature matrix

        Returns:
            Array of regime probabilities (n_samples, n_regimes)
        """
        X_array = X.values if hasattr(X, 'values') else X
        return self.model.predict_proba(X_array)

    def get_transition_matrix(self):
        """
        Get regime transition probability matrix.

        Returns:
            DataFrame with transition probabilities
        """
        if self.transition_matrix is None:
            return None

        regime_names = [f'Regime_{i}' for i in range(self.n_regimes)]

        return pd.DataFrame(
            self.transition_matrix,
            index=[f'From_{name}' for name in regime_names],
            columns=[f'To_{name}' for name in regime_names]
        )

    def get_regime_persistence(self):
        """
        Calculate expected duration of each regime.

        Returns:
            Dict with expected duration in time periods
        """
        if self.transition_matrix is None:
            return None

        persistence = {}
        for i in range(self.n_regimes):
            # Expected duration = 1 / (1 - self-transition probability)
            self_prob = self.transition_matrix[i, i]
            if self_prob < 1:
                expected_duration = 1 / (1 - self_prob)
            else:
                expected_duration = np.inf
            persistence[f'Regime_{i}'] = expected_duration

        return persistence

    def select_optimal_n_regimes(self, X, min_regimes=2, max_regimes=6):
        """
        Select optimal number of regimes using BIC/AIC.

        Args:
            X: Feature matrix
            min_regimes: Minimum number of regimes
            max_regimes: Maximum number of regimes

        Returns:
            optimal_n, results_df
        """
        X_array = X.values if hasattr(X, 'values') else X
        results = []

        for n in range(min_regimes, max_regimes + 1):
            try:
                hmm = GaussianHMM(
                    n_components=n,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    random_state=self.random_state
                )
                hmm.fit(X_array)

                log_likelihood = hmm.score(X_array)

                # Calculate BIC/AIC
                n_samples, n_features = X_array.shape
                n_params = n * n_features + n * n_features * (n_features + 1) / 2 + n * n

                bic = -2 * log_likelihood + n_params * np.log(n_samples)
                aic = -2 * log_likelihood + 2 * n_params

                results.append({
                    'n_regimes': n,
                    'log_likelihood': log_likelihood,
                    'bic': bic,
                    'aic': aic
                })
            except Exception as e:
                print(f"Failed for n_regimes={n}: {e}")

        results_df = pd.DataFrame(results)

        if len(results_df) > 0:
            optimal_n = results_df.loc[results_df['bic'].idxmin(), 'n_regimes']
        else:
            optimal_n = 4

        return int(optimal_n), results_df


class KMeansRegimeDetector:
    """
    Simple K-Means clustering for regime detection (baseline).
    """

    def __init__(self, n_regimes=4, random_state=42):
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model = None

    def fit(self, X):
        self.model = KMeans(
            n_clusters=self.n_regimes,
            random_state=self.random_state,
            n_init=10
        )
        X_array = X.values if hasattr(X, 'values') else X
        self.model.fit(X_array)
        return self

    def predict(self, X):
        X_array = X.values if hasattr(X, 'values') else X
        return self.model.predict(X_array)


def map_regimes_to_phases(regime_labels, feature_df, growth_cols=None, inflation_cols=None):
    """
    Map discovered regimes to Investment Clock phases based on characteristics.

    Args:
        regime_labels: Array of regime labels
        feature_df: DataFrame with features
        growth_cols: Columns representing growth (positive = rising)
        inflation_cols: Columns representing inflation (positive = rising)

    Returns:
        DataFrame with regime-to-phase mapping
    """
    if growth_cols is None:
        growth_cols = ['orders_inv_direction', 'cfnai', 'yield_curve_10y3m']
    if inflation_cols is None:
        inflation_cols = ['ppi_direction', 'cpi_yoy', 'oil_yoy']

    # Filter to available columns
    growth_cols = [c for c in growth_cols if c in feature_df.columns]
    inflation_cols = [c for c in inflation_cols if c in feature_df.columns]

    if not growth_cols or not inflation_cols:
        print("Warning: Missing growth or inflation columns for regime mapping")
        return None

    # Calculate average growth/inflation signal per regime
    mapping = []
    n_regimes = len(np.unique(regime_labels))

    for regime in range(n_regimes):
        mask = regime_labels == regime

        if mask.sum() == 0:
            continue

        regime_data = feature_df[mask]

        # Average growth signal
        growth_avg = regime_data[growth_cols].mean().mean()

        # Average inflation signal
        inflation_avg = regime_data[inflation_cols].mean().mean()

        # Map to phase
        if growth_avg > 0 and inflation_avg <= 0:
            phase = 'Recovery'
        elif growth_avg > 0 and inflation_avg > 0:
            phase = 'Overheat'
        elif growth_avg <= 0 and inflation_avg > 0:
            phase = 'Stagflation'
        else:
            phase = 'Reflation'

        mapping.append({
            'regime': regime,
            'count': mask.sum(),
            'pct_time': mask.mean() * 100,
            'growth_avg': growth_avg,
            'inflation_avg': inflation_avg,
            'mapped_phase': phase
        })

    return pd.DataFrame(mapping)


def compare_regime_methods(X, feature_df, n_regimes=4):
    """
    Compare different regime detection methods.

    Args:
        X: Scaled feature matrix for clustering
        feature_df: Original features for interpretation
        n_regimes: Number of regimes

    Returns:
        Dict with results from each method
    """
    results = {}

    # GMM
    print("Fitting GMM...")
    gmm = GMMRegimeDetector(n_regimes=n_regimes)
    gmm.fit(X)
    gmm_labels = gmm.predict(X)
    results['gmm'] = {
        'model': gmm,
        'labels': gmm_labels,
        'mapping': map_regimes_to_phases(gmm_labels, feature_df)
    }

    # HMM
    if HMM_AVAILABLE:
        print("Fitting HMM...")
        hmm = HMMRegimeDetector(n_regimes=n_regimes)
        hmm.fit(X)
        hmm_labels = hmm.predict(X)
        results['hmm'] = {
            'model': hmm,
            'labels': hmm_labels,
            'mapping': map_regimes_to_phases(hmm_labels, feature_df),
            'transition_matrix': hmm.get_transition_matrix(),
            'persistence': hmm.get_regime_persistence()
        }

    # K-Means
    print("Fitting K-Means...")
    kmeans = KMeansRegimeDetector(n_regimes=n_regimes)
    kmeans.fit(X)
    kmeans_labels = kmeans.predict(X)
    results['kmeans'] = {
        'model': kmeans,
        'labels': kmeans_labels,
        'mapping': map_regimes_to_phases(kmeans_labels, feature_df)
    }

    return results
