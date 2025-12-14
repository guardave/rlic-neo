"""
Supervised Regime Classification.

Uses rule-based phase labels as ground truth to train ML classifiers.
This hybrid approach keeps economic meaning while leveraging ML for:
1. Handling edge cases and transitions
2. Incorporating additional signals
3. Probabilistic phase membership
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class SupervisedRegimeClassifier:
    """
    Supervised classifier for Investment Clock phases.

    Trains on rule-based phase labels to learn the relationship
    between features and phases, then generalizes to new data.
    """

    def __init__(self, model_type='random_forest', **model_params):
        """
        Initialize classifier.

        Args:
            model_type: 'random_forest', 'xgboost', 'gradient_boosting', 'logistic'
            **model_params: Parameters passed to the underlying model
        """
        self.model_type = model_type
        self.model_params = model_params
        self.model = None
        self.feature_importance_ = None
        self.classes_ = None

    def _create_model(self):
        """Create the underlying ML model."""
        if self.model_type == 'random_forest':
            defaults = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_leaf': 5,
                'random_state': 42,
                'class_weight': 'balanced'
            }
            defaults.update(self.model_params)
            return RandomForestClassifier(**defaults)

        elif self.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not installed. Use: pip install xgboost")
            defaults = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'mlogloss'
            }
            defaults.update(self.model_params)
            return XGBClassifier(**defaults)

        elif self.model_type == 'gradient_boosting':
            defaults = {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'random_state': 42
            }
            defaults.update(self.model_params)
            return GradientBoostingClassifier(**defaults)

        elif self.model_type == 'logistic':
            defaults = {
                'max_iter': 1000,
                'random_state': 42,
                'class_weight': 'balanced',
                'multi_class': 'multinomial'
            }
            defaults.update(self.model_params)
            return LogisticRegression(**defaults)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(self, X, y):
        """
        Fit classifier to training data.

        Args:
            X: Feature matrix (DataFrame or array)
            y: Phase labels (0-3)

        Returns:
            self
        """
        self.model = self._create_model()

        # Convert to arrays
        X_arr = X.values if hasattr(X, 'values') else X
        y_arr = y.values if hasattr(y, 'values') else y

        # Remove any NaN
        valid_mask = ~np.isnan(y_arr)
        X_arr = X_arr[valid_mask]
        y_arr = y_arr[valid_mask]

        self.model.fit(X_arr, y_arr.astype(int))
        self.classes_ = self.model.classes_

        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            if hasattr(X, 'columns'):
                self.feature_importance_ = pd.Series(
                    self.model.feature_importances_,
                    index=X.columns
                ).sort_values(ascending=False)
            else:
                self.feature_importance_ = self.model.feature_importances_

        return self

    def predict(self, X):
        """
        Predict phase labels.

        Args:
            X: Feature matrix

        Returns:
            Array of phase labels (0-3)
        """
        X_arr = X.values if hasattr(X, 'values') else X
        return self.model.predict(X_arr)

    def predict_proba(self, X):
        """
        Predict phase probabilities.

        Args:
            X: Feature matrix

        Returns:
            Array of probabilities (n_samples, 4)
        """
        X_arr = X.values if hasattr(X, 'values') else X
        return self.model.predict_proba(X_arr)

    def evaluate(self, X, y_true, phase_names=None):
        """
        Evaluate classifier performance.

        Args:
            X: Feature matrix
            y_true: True phase labels
            phase_names: Optional dict mapping labels to names

        Returns:
            Dict with evaluation metrics
        """
        y_pred = self.predict(X)

        # Handle NaN in y_true
        y_true_arr = y_true.values if hasattr(y_true, 'values') else y_true
        valid_mask = ~np.isnan(y_true_arr)
        y_true_valid = y_true_arr[valid_mask].astype(int)
        y_pred_valid = y_pred[valid_mask]

        accuracy = accuracy_score(y_true_valid, y_pred_valid)

        if phase_names is None:
            phase_names = {0: 'Recovery', 1: 'Overheat', 2: 'Stagflation', 3: 'Reflation'}

        target_names = [phase_names[i] for i in sorted(phase_names.keys())]

        return {
            'accuracy': accuracy,
            'classification_report': classification_report(
                y_true_valid, y_pred_valid,
                target_names=target_names,
                output_dict=True
            ),
            'confusion_matrix': confusion_matrix(y_true_valid, y_pred_valid)
        }

    def get_top_features(self, n=10):
        """Get top n most important features."""
        if self.feature_importance_ is None:
            return None
        return self.feature_importance_.head(n)


class HMMWithSupervisedInit:
    """
    HMM initialized with supervised regime assignments.

    Uses rule-based phases to initialize HMM states, then lets
    HMM refine transition dynamics.
    """

    def __init__(self, n_regimes=4, covariance_type='diag', n_iter=100, random_state=42):
        """
        Initialize HMM with supervised initialization.

        Args:
            n_regimes: Number of hidden states (should be 4 for Investment Clock)
            covariance_type: 'full', 'diag', 'tied', or 'spherical'
            n_iter: Maximum EM iterations
            random_state: Random seed
        """
        try:
            from hmmlearn.hmm import GaussianHMM
            self.HMM_AVAILABLE = True
        except ImportError:
            self.HMM_AVAILABLE = False

        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = None
        self.initial_labels = None

    def fit(self, X, initial_labels):
        """
        Fit HMM using supervised initialization.

        Args:
            X: Feature matrix
            initial_labels: Rule-based phase labels for initialization

        Returns:
            self
        """
        if not self.HMM_AVAILABLE:
            raise ImportError("hmmlearn required. Install with: pip install hmmlearn")

        from hmmlearn.hmm import GaussianHMM

        X_arr = X.values if hasattr(X, 'values') else X
        labels = initial_labels.values if hasattr(initial_labels, 'values') else initial_labels

        # Remove NaN
        valid_mask = ~np.isnan(labels)
        X_arr = X_arr[valid_mask]
        labels = labels[valid_mask].astype(int)

        self.initial_labels = labels

        # Calculate initial parameters from rule-based labels
        n_features = X_arr.shape[1]

        # Initial state distribution
        startprob = np.zeros(self.n_regimes)
        for i in range(self.n_regimes):
            startprob[i] = (labels == i).mean()
        startprob = startprob / startprob.sum()  # Normalize

        # Transition matrix from label sequences
        transmat = np.zeros((self.n_regimes, self.n_regimes))
        for i in range(len(labels) - 1):
            from_state = labels[i]
            to_state = labels[i + 1]
            transmat[from_state, to_state] += 1

        # Normalize rows
        for i in range(self.n_regimes):
            row_sum = transmat[i].sum()
            if row_sum > 0:
                transmat[i] /= row_sum
            else:
                transmat[i] = 1 / self.n_regimes  # Uniform if no transitions

        # Means and covariances per state
        means = np.zeros((self.n_regimes, n_features))
        covars = np.zeros((self.n_regimes, n_features))

        for i in range(self.n_regimes):
            state_data = X_arr[labels == i]
            if len(state_data) > 0:
                means[i] = state_data.mean(axis=0)
                covars[i] = state_data.var(axis=0) + 1e-6  # Add small constant for stability
            else:
                means[i] = X_arr.mean(axis=0)
                covars[i] = X_arr.var(axis=0) + 1e-6

        # Create and initialize HMM
        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            init_params='',  # Don't auto-initialize
            params='stmc'  # Still train all parameters
        )

        # Set initial parameters
        self.model.startprob_ = startprob
        self.model.transmat_ = transmat
        self.model.means_ = means
        self.model.covars_ = covars

        # Fit (refine parameters)
        self.model.fit(X_arr)

        return self

    def predict(self, X):
        """
        Predict regime sequence (Viterbi).

        Args:
            X: Feature matrix

        Returns:
            Array of regime labels
        """
        X_arr = X.values if hasattr(X, 'values') else X
        return self.model.predict(X_arr)

    def predict_proba(self, X):
        """
        Predict regime probabilities.

        Args:
            X: Feature matrix

        Returns:
            Array of probabilities
        """
        X_arr = X.values if hasattr(X, 'values') else X
        return self.model.predict_proba(X_arr)

    def get_transition_matrix(self):
        """Get learned transition matrix."""
        if self.model is None:
            return None

        phase_names = {0: 'Recovery', 1: 'Overheat', 2: 'Stagflation', 3: 'Reflation'}

        return pd.DataFrame(
            self.model.transmat_,
            index=[f"From_{phase_names[i]}" for i in range(self.n_regimes)],
            columns=[f"To_{phase_names[i]}" for i in range(self.n_regimes)]
        )

    def get_regime_persistence(self):
        """Calculate expected duration of each regime."""
        if self.model is None:
            return None

        phase_names = {0: 'Recovery', 1: 'Overheat', 2: 'Stagflation', 3: 'Reflation'}
        persistence = {}

        for i in range(self.n_regimes):
            self_prob = self.model.transmat_[i, i]
            if self_prob < 1:
                duration = 1 / (1 - self_prob)
            else:
                duration = np.inf
            persistence[phase_names[i]] = duration

        return persistence

    def compare_with_initial(self, X):
        """
        Compare HMM predictions with initial rule-based labels.

        Returns:
            Dict with comparison metrics
        """
        if self.initial_labels is None:
            return None

        X_arr = X.values if hasattr(X, 'values') else X

        # Ensure same length
        n = min(len(X_arr), len(self.initial_labels))
        X_arr = X_arr[:n]
        initial = self.initial_labels[:n]

        hmm_pred = self.model.predict(X_arr)

        agreement = (hmm_pred == initial).mean()

        # Transition smoothness (count regime changes)
        initial_changes = np.sum(np.diff(initial) != 0)
        hmm_changes = np.sum(np.diff(hmm_pred) != 0)

        return {
            'agreement_rate': agreement,
            'initial_regime_changes': initial_changes,
            'hmm_regime_changes': hmm_changes,
            'smoothing_ratio': hmm_changes / initial_changes if initial_changes > 0 else 1
        }


def ensemble_regime_prediction(predictions_list, weights=None):
    """
    Combine predictions from multiple models using voting.

    Args:
        predictions_list: List of prediction arrays
        weights: Optional weights for each model

    Returns:
        Array of ensemble predictions
    """
    n_models = len(predictions_list)
    n_samples = len(predictions_list[0])

    if weights is None:
        weights = np.ones(n_models) / n_models

    # Voting
    ensemble_pred = np.zeros(n_samples)

    for i in range(n_samples):
        votes = {}
        for j, preds in enumerate(predictions_list):
            pred = int(preds[i])
            votes[pred] = votes.get(pred, 0) + weights[j]

        ensemble_pred[i] = max(votes, key=votes.get)

    return ensemble_pred.astype(int)
