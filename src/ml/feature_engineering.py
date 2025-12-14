"""
Feature Engineering Pipeline for RLIC ML Models.

This module creates features from economic indicators for regime detection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class FeatureEngineer:
    """Create ML features from economic indicators."""

    def __init__(self, lookback_window=60):
        """
        Initialize feature engineer.

        Args:
            lookback_window: Rolling window for z-score normalization (months)
        """
        self.lookback_window = lookback_window
        self.scaler = StandardScaler()
        self.pca = None
        self.feature_names = []

    def create_growth_features(self, df):
        """Create growth-related features."""
        features = pd.DataFrame(index=df.index)

        # Orders/Inventories Ratio (best growth indicator)
        if 'orders_inv_ratio' in df.columns:
            ratio = df['orders_inv_ratio']
            features['orders_inv_yoy'] = ratio.pct_change(12)
            features['orders_inv_mom'] = ratio.pct_change(1)
            features['orders_inv_3m_mom'] = ratio.pct_change(3)
            features['orders_inv_zscore'] = self._zscore(ratio)
            features['orders_inv_direction'] = np.sign(ratio.rolling(3).mean() - ratio.rolling(6).mean())

        # CFNAI
        if 'cfnai' in df.columns:
            features['cfnai'] = df['cfnai']
            features['cfnai_3ma'] = df['cfnai'].rolling(3).mean()
            features['cfnai_momentum'] = df['cfnai'] - df['cfnai'].shift(3)

        # Yield Curve
        if 'spread_10y3m' in df.columns:
            features['yield_curve_10y3m'] = df['spread_10y3m']
            features['yield_curve_direction'] = np.sign(df['spread_10y3m'].diff(3))

        if 'spread_10y2y' in df.columns:
            features['yield_curve_10y2y'] = df['spread_10y2y']

        # LEI
        if 'lei' in df.columns:
            lei = df['lei']
            features['lei_mom'] = lei.pct_change(1)
            features['lei_3m'] = lei.pct_change(3)
            features['lei_6m'] = lei.pct_change(6)

        # Initial Claims (inverted - lower claims = stronger growth)
        if 'initial_claims' in df.columns:
            claims = df['initial_claims']
            features['claims_yoy_inv'] = -claims.pct_change(12)
            features['claims_zscore_inv'] = -self._zscore(claims)

        # Building Permits
        if 'building_permits' in df.columns:
            permits = df['building_permits']
            features['permits_yoy'] = permits.pct_change(12)
            features['permits_momentum'] = permits.pct_change(3)

        # Capacity Utilization
        if 'capacity_util' in df.columns:
            cu = df['capacity_util']
            features['capacity_util'] = cu
            features['capacity_util_zscore'] = self._zscore(cu)

        # Industrial Production (benchmark)
        if 'industrial_prod' in df.columns:
            ip = df['industrial_prod']
            features['ip_yoy'] = ip.pct_change(12)
            features['ip_momentum'] = ip.pct_change(1)

        # OECD CLI
        if 'oecd_cli' in df.columns:
            cli = df['oecd_cli']
            features['oecd_cli'] = cli
            features['oecd_cli_deviation'] = cli - 100

        # Unemployment (inverted)
        if 'unemployment' in df.columns:
            ur = df['unemployment']
            features['unemployment_inv'] = -ur
            features['unemployment_change'] = -ur.diff(3)

        return features

    def create_inflation_features(self, df):
        """Create inflation-related features."""
        features = pd.DataFrame(index=df.index)

        # PPI (best inflation indicator)
        if 'ppi_all' in df.columns:
            ppi = df['ppi_all']
            features['ppi_yoy'] = ppi.pct_change(12)
            features['ppi_mom'] = ppi.pct_change(1)
            features['ppi_3m_ann'] = ppi.pct_change(3) * 4
            features['ppi_direction'] = np.sign(ppi.rolling(3).mean() - ppi.rolling(6).mean())
            features['ppi_zscore'] = self._zscore(ppi.pct_change(12))

        # CPI (benchmark)
        if 'cpi_all' in df.columns:
            cpi = df['cpi_all']
            features['cpi_yoy'] = cpi.pct_change(12)
            features['cpi_mom'] = cpi.pct_change(1)
            features['cpi_3m_ann'] = cpi.pct_change(3) * 4

        # Core CPI
        if 'cpi_core' in df.columns:
            core = df['cpi_core']
            features['core_cpi_yoy'] = core.pct_change(12)

        # Breakeven Inflation
        if 'breakeven_10y' in df.columns:
            be = df['breakeven_10y']
            features['breakeven_10y'] = be
            features['breakeven_10y_mom'] = be.diff(1)
            features['breakeven_direction'] = np.sign(be.rolling(3).mean() - be.rolling(6).mean())

        if 'breakeven_5y' in df.columns:
            features['breakeven_5y'] = df['breakeven_5y']

        # M2 Money Supply (lagged inflation indicator)
        if 'm2' in df.columns:
            m2 = df['m2']
            features['m2_yoy'] = m2.pct_change(12)
            features['m2_yoy_lag12'] = m2.pct_change(12).shift(12)
            features['m2_yoy_lag18'] = m2.pct_change(12).shift(18)

        # Commodity Index
        if 'commodity_index' in df.columns:
            comm = df['commodity_index']
            features['commodity_yoy'] = comm.pct_change(12)
            features['commodity_mom'] = comm.pct_change(1)
            features['commodity_direction'] = np.sign(comm.rolling(3).mean() - comm.rolling(6).mean())

        # Oil
        if 'oil_wti' in df.columns:
            oil = df['oil_wti']
            features['oil_yoy'] = oil.pct_change(12)
            features['oil_mom'] = oil.pct_change(1)
            features['oil_direction'] = np.sign(oil.rolling(3).mean() - oil.rolling(6).mean())

        # Import Prices
        if 'import_prices' in df.columns:
            imp = df['import_prices']
            features['import_prices_yoy'] = imp.pct_change(12)

        # PCE
        if 'pce_price' in df.columns:
            pce = df['pce_price']
            features['pce_yoy'] = pce.pct_change(12)

        # Inflation Expectations
        if 'inflation_expect' in df.columns:
            features['inflation_expect'] = df['inflation_expect']

        return features

    def create_market_features(self, prices_df):
        """Create market-based features from price data."""
        features = pd.DataFrame(index=prices_df.index)

        # S&P 500 momentum
        spy_cols = ['SPY', 'spy', 'sp500']
        for col in spy_cols:
            if col in prices_df.columns:
                spy = prices_df[col]
                features['spy_1m_ret'] = spy.pct_change(21)  # ~1 month
                features['spy_3m_ret'] = spy.pct_change(63)  # ~3 months
                features['spy_6m_ret'] = spy.pct_change(126)  # ~6 months
                features['spy_12m_ret'] = spy.pct_change(252)  # ~12 months
                features['spy_momentum'] = spy / spy.rolling(252).mean() - 1
                break

        # VIX
        if 'vix' in prices_df.columns:
            vix = prices_df['vix']
            features['vix'] = vix
            features['vix_zscore'] = self._zscore(vix)
            features['vix_regime'] = (vix > 20).astype(int)

        # Gold momentum
        gold_cols = ['GLD', 'gld', 'gold']
        for col in gold_cols:
            if col in prices_df.columns:
                gold = prices_df[col]
                features['gold_3m_ret'] = gold.pct_change(63)
                features['gold_momentum'] = gold / gold.rolling(252).mean() - 1
                break

        # Bond momentum
        bond_cols = ['TLT', 'tlt', 'treasury_10y']
        for col in bond_cols:
            if col in prices_df.columns:
                bond = prices_df[col]
                features['bond_3m_ret'] = bond.pct_change(63)
                break

        return features

    def _zscore(self, series):
        """Calculate rolling z-score."""
        mean = series.rolling(self.lookback_window).mean()
        std = series.rolling(self.lookback_window).std()
        return (series - mean) / std

    def create_all_features(self, indicators_df, prices_df=None):
        """
        Create complete feature set.

        Args:
            indicators_df: DataFrame with economic indicators
            prices_df: Optional DataFrame with price data

        Returns:
            DataFrame with all features
        """
        # Growth features
        growth_features = self.create_growth_features(indicators_df)

        # Inflation features
        inflation_features = self.create_inflation_features(indicators_df)

        # Combine
        features = pd.concat([growth_features, inflation_features], axis=1)

        # Add market features if prices available
        if prices_df is not None and not prices_df.empty:
            # Resample prices to monthly
            monthly_prices = prices_df.resample('ME').last()
            market_features = self.create_market_features(monthly_prices)

            # Align and merge
            features = features.join(market_features, how='left')

        # Remove duplicate columns
        features = features.loc[:, ~features.columns.duplicated()]

        # Store feature names
        self.feature_names = features.columns.tolist()

        return features

    def prepare_ml_data(self, features_df, target_df=None, lag=1, dropna=True, min_valid_ratio=0.7):
        """
        Prepare features and targets for ML model.

        Args:
            features_df: DataFrame with features
            target_df: Optional DataFrame/Series with targets
            lag: Number of periods to lag features (for prediction)
            dropna: Whether to drop rows with NaN values
            min_valid_ratio: Minimum ratio of non-NaN values to keep a column

        Returns:
            X: Feature matrix
            y: Target vector (if target_df provided)
            valid_index: Index of valid samples
        """
        # Lag features (use features at t-lag to predict target at t)
        X = features_df.shift(lag) if lag > 0 else features_df.copy()

        # Remove columns with too many NaN values
        valid_cols = X.columns[X.notna().mean() >= min_valid_ratio]
        X = X[valid_cols]
        print(f"  Kept {len(valid_cols)}/{len(features_df.columns)} features with >={min_valid_ratio*100:.0f}% valid data")

        if target_df is not None:
            # Align
            common_idx = X.index.intersection(target_df.index)
            X = X.loc[common_idx]
            y = target_df.loc[common_idx]

            if dropna:
                valid_mask = X.notna().all(axis=1) & y.notna()
                X = X[valid_mask]
                y = y[valid_mask]

            return X, y, X.index

        # Replace inf with NaN
        X = X.replace([np.inf, -np.inf], np.nan)

        if dropna:
            valid_mask = X.notna().all(axis=1)
            X = X[valid_mask]

        return X, None, X.index

    def scale_features(self, X, fit=True):
        """
        Standardize features.

        Args:
            X: Feature matrix
            fit: Whether to fit scaler (True for training, False for test)

        Returns:
            Scaled feature matrix
        """
        if fit:
            return pd.DataFrame(
                self.scaler.fit_transform(X),
                index=X.index,
                columns=X.columns
            )
        else:
            return pd.DataFrame(
                self.scaler.transform(X),
                index=X.index,
                columns=X.columns
            )

    def apply_pca(self, X, n_components=0.95, fit=True):
        """
        Apply PCA for dimensionality reduction.

        Args:
            X: Feature matrix (should be scaled)
            n_components: Number of components or variance ratio to keep
            fit: Whether to fit PCA (True for training, False for test)

        Returns:
            PCA-transformed feature matrix
        """
        if fit:
            self.pca = PCA(n_components=n_components)
            X_pca = self.pca.fit_transform(X)
        else:
            X_pca = self.pca.transform(X)

        # Create column names
        n_comp = X_pca.shape[1]
        columns = [f'PC{i+1}' for i in range(n_comp)]

        return pd.DataFrame(X_pca, index=X.index, columns=columns)


def create_rule_based_targets(indicators_df, growth_col='orders_inv_ratio', inflation_col='ppi_all'):
    """
    Create target labels using rule-based approach (current best method).

    Args:
        indicators_df: DataFrame with indicators
        growth_col: Column for growth signal
        inflation_col: Column for inflation signal

    Returns:
        DataFrame with growth_signal, inflation_signal, phase
    """
    targets = pd.DataFrame(index=indicators_df.index)

    # Growth signal (Orders/Inv MoM direction)
    if growth_col in indicators_df.columns:
        ratio = indicators_df[growth_col]
        growth_3ma = ratio.rolling(3).mean()
        growth_6ma = ratio.rolling(6).mean()
        targets['growth_signal'] = np.where(growth_3ma > growth_6ma, 1, -1)
    else:
        targets['growth_signal'] = np.nan

    # Inflation signal (PPI MoM direction)
    if inflation_col in indicators_df.columns:
        ppi = indicators_df[inflation_col]
        ppi_3ma = ppi.rolling(3).mean()
        ppi_6ma = ppi.rolling(6).mean()
        targets['inflation_signal'] = np.where(ppi_3ma > ppi_6ma, 1, -1)
    else:
        targets['inflation_signal'] = np.nan

    # Phase classification
    def classify_phase(row):
        g, i = row['growth_signal'], row['inflation_signal']
        if pd.isna(g) or pd.isna(i):
            return np.nan
        if g == 1 and i == -1:
            return 0  # Recovery
        elif g == 1 and i == 1:
            return 1  # Overheat
        elif g == -1 and i == 1:
            return 2  # Stagflation
        elif g == -1 and i == -1:
            return 3  # Reflation
        return np.nan

    targets['phase'] = targets.apply(classify_phase, axis=1)
    targets['phase_name'] = targets['phase'].map({
        0: 'Recovery',
        1: 'Overheat',
        2: 'Stagflation',
        3: 'Reflation'
    })

    return targets


# Phase label mappings
PHASE_NAMES = {0: 'Recovery', 1: 'Overheat', 2: 'Stagflation', 3: 'Reflation'}
PHASE_IDS = {'Recovery': 0, 'Overheat': 1, 'Stagflation': 2, 'Reflation': 3}


def create_composite_features(indicators_df, prices_df=None):
    """
    Create simplified 2D composite features aligned with Investment Clock theory.

    Instead of 49 features, creates just 2 composite scores:
    - Growth Composite: Average of normalized growth indicators
    - Inflation Composite: Average of normalized inflation indicators

    This directly maps to the Investment Clock's 2D framework.

    Args:
        indicators_df: DataFrame with economic indicators
        prices_df: Optional price data (not used for composites)

    Returns:
        DataFrame with 'growth_composite' and 'inflation_composite'
    """
    composites = pd.DataFrame(index=indicators_df.index)

    # Growth indicators (higher = stronger growth)
    growth_components = []

    # Orders/Inventory ratio direction
    if 'orders_inv_ratio' in indicators_df.columns:
        ratio = indicators_df['orders_inv_ratio']
        direction = np.sign(ratio.rolling(3).mean() - ratio.rolling(6).mean())
        growth_components.append(direction)

    # CFNAI (already normalized around 0)
    if 'cfnai' in indicators_df.columns:
        cfnai = indicators_df['cfnai']
        growth_components.append(np.sign(cfnai.rolling(3).mean()))

    # Yield curve (positive = growth, inverted = recession)
    if 'spread_10y3m' in indicators_df.columns:
        spread = indicators_df['spread_10y3m']
        growth_components.append(np.sign(spread))

    # LEI momentum
    if 'lei' in indicators_df.columns:
        lei = indicators_df['lei']
        lei_mom = lei.pct_change(3)
        growth_components.append(np.sign(lei_mom))

    # Claims (inverted - lower claims = stronger growth)
    if 'initial_claims' in indicators_df.columns:
        claims = indicators_df['initial_claims']
        claims_mom = -claims.pct_change(3)  # Inverted
        growth_components.append(np.sign(claims_mom))

    # OECD CLI
    if 'oecd_cli' in indicators_df.columns:
        cli = indicators_df['oecd_cli']
        growth_components.append(np.sign(cli - 100))

    # Inflation indicators (higher = rising inflation)
    inflation_components = []

    # PPI direction
    if 'ppi_all' in indicators_df.columns:
        ppi = indicators_df['ppi_all']
        direction = np.sign(ppi.rolling(3).mean() - ppi.rolling(6).mean())
        inflation_components.append(direction)

    # CPI YoY momentum
    if 'cpi_all' in indicators_df.columns:
        cpi = indicators_df['cpi_all']
        cpi_yoy = cpi.pct_change(12)
        cpi_mom = cpi_yoy - cpi_yoy.shift(3)
        inflation_components.append(np.sign(cpi_mom))

    # Breakeven direction
    if 'breakeven_10y' in indicators_df.columns:
        be = indicators_df['breakeven_10y']
        direction = np.sign(be.rolling(3).mean() - be.rolling(6).mean())
        inflation_components.append(direction)

    # Commodity direction
    if 'commodity_index' in indicators_df.columns:
        comm = indicators_df['commodity_index']
        direction = np.sign(comm.rolling(3).mean() - comm.rolling(6).mean())
        inflation_components.append(direction)

    # Oil direction
    if 'oil_wti' in indicators_df.columns:
        oil = indicators_df['oil_wti']
        direction = np.sign(oil.rolling(3).mean() - oil.rolling(6).mean())
        inflation_components.append(direction)

    # Calculate composites as average of components
    if growth_components:
        growth_df = pd.concat(growth_components, axis=1)
        composites['growth_composite'] = growth_df.mean(axis=1)
        composites['growth_n_signals'] = growth_df.notna().sum(axis=1)

    if inflation_components:
        inflation_df = pd.concat(inflation_components, axis=1)
        composites['inflation_composite'] = inflation_df.mean(axis=1)
        composites['inflation_n_signals'] = inflation_df.notna().sum(axis=1)

    # Also create binary signals for direct comparison
    composites['growth_signal'] = np.sign(composites['growth_composite'])
    composites['inflation_signal'] = np.sign(composites['inflation_composite'])

    return composites


def select_phase_correlated_features(features_df, phase_labels, min_correlation=0.1):
    """
    Select features that correlate with rule-based phase labels.

    Keeps only features that "agree" with the theoretical framework.

    Args:
        features_df: DataFrame with all features
        phase_labels: Series with rule-based phase labels (0-3)
        min_correlation: Minimum absolute correlation to keep feature

    Returns:
        List of selected feature names
    """
    from scipy.stats import pointbiserialr

    selected = []

    # Create binary indicators for each phase
    for phase in range(4):
        phase_binary = (phase_labels == phase).astype(int)

        for col in features_df.columns:
            feat = features_df[col].dropna()
            common_idx = feat.index.intersection(phase_binary.dropna().index)

            if len(common_idx) < 50:
                continue

            try:
                corr, pval = pointbiserialr(
                    phase_binary.loc[common_idx],
                    feat.loc[common_idx]
                )
                if abs(corr) >= min_correlation and pval < 0.05:
                    if col not in selected:
                        selected.append(col)
            except Exception:
                continue

    return selected


def create_direction_only_features(indicators_df):
    """
    Create only direction-based features (binary +1/-1).

    These directly map to the Investment Clock quadrant logic.

    Args:
        indicators_df: DataFrame with economic indicators

    Returns:
        DataFrame with direction features only
    """
    features = pd.DataFrame(index=indicators_df.index)

    # Growth directions
    if 'orders_inv_ratio' in indicators_df.columns:
        ratio = indicators_df['orders_inv_ratio']
        features['orders_inv_dir'] = np.sign(ratio.rolling(3).mean() - ratio.rolling(6).mean())

    if 'cfnai' in indicators_df.columns:
        features['cfnai_dir'] = np.sign(indicators_df['cfnai'].rolling(3).mean())

    if 'spread_10y3m' in indicators_df.columns:
        features['yield_curve_dir'] = np.sign(indicators_df['spread_10y3m'])

    if 'lei' in indicators_df.columns:
        features['lei_dir'] = np.sign(indicators_df['lei'].pct_change(3))

    if 'initial_claims' in indicators_df.columns:
        features['claims_dir'] = -np.sign(indicators_df['initial_claims'].pct_change(3))

    if 'oecd_cli' in indicators_df.columns:
        features['cli_dir'] = np.sign(indicators_df['oecd_cli'] - 100)

    # Inflation directions
    if 'ppi_all' in indicators_df.columns:
        ppi = indicators_df['ppi_all']
        features['ppi_dir'] = np.sign(ppi.rolling(3).mean() - ppi.rolling(6).mean())

    if 'cpi_all' in indicators_df.columns:
        cpi_yoy = indicators_df['cpi_all'].pct_change(12)
        features['cpi_dir'] = np.sign(cpi_yoy - cpi_yoy.shift(3))

    if 'breakeven_10y' in indicators_df.columns:
        be = indicators_df['breakeven_10y']
        features['breakeven_dir'] = np.sign(be.rolling(3).mean() - be.rolling(6).mean())

    if 'commodity_index' in indicators_df.columns:
        comm = indicators_df['commodity_index']
        features['commodity_dir'] = np.sign(comm.rolling(3).mean() - comm.rolling(6).mean())

    if 'oil_wti' in indicators_df.columns:
        oil = indicators_df['oil_wti']
        features['oil_dir'] = np.sign(oil.rolling(3).mean() - oil.rolling(6).mean())

    return features
