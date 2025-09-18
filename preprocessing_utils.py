import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class CTRPreprocessingUtils:
    """Utility functions for advanced CTR data preprocessing"""

    @staticmethod
    def analyze_feature_importance(X, y, feature_names=None, method='chi2', k=20):
        """Analyze feature importance using various methods"""
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        if method == 'chi2':
            # Make sure features are non-negative for chi2
            X_pos = X - X.min(axis=0) + 1
            selector = SelectKBest(chi2, k=k)
            selector.fit(X_pos, y)
            scores = selector.scores_
        elif method == 'f_classif':
            selector = SelectKBest(f_classif, k=k)
            selector.fit(X, y)
            scores = selector.scores_
        elif method == 'mutual_info':
            scores = mutual_info_classif(X, y)
        else:
            raise ValueError("Method must be 'chi2', 'f_classif', or 'mutual_info'")

        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': scores
        }).sort_values('importance', ascending=False)

        return feature_importance

    @staticmethod
    def detect_data_drift(train_df, test_df, features, threshold=0.05):
        """Detect data drift between train and test sets using KS test"""
        drift_results = []

        for feature in features:
            if feature in train_df.columns and feature in test_df.columns:
                # Kolmogorov-Smirnov test
                ks_stat, p_value = stats.ks_2samp(
                    train_df[feature].dropna(),
                    test_df[feature].dropna()
                )

                drift_results.append({
                    'feature': feature,
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'has_drift': p_value < threshold
                })

        return pd.DataFrame(drift_results).sort_values('ks_statistic', ascending=False)

    @staticmethod
    def create_frequency_encoding(df, columns, smoothing=1):
        """Create frequency encoding for categorical variables"""
        df = df.copy()
        encoders = {}

        for col in columns:
            if col in df.columns:
                # Calculate frequency
                freq_map = df[col].value_counts().to_dict()

                # Apply smoothing
                total_count = len(df)
                for key in freq_map:
                    freq_map[key] = (freq_map[key] + smoothing) / (total_count + smoothing * len(freq_map))

                df[f'{col}_freq_enc'] = df[col].map(freq_map)
                encoders[col] = freq_map

        return df, encoders

    @staticmethod
    def create_interaction_features(df, feature_pairs, max_combinations=50):
        """Create interaction features between feature pairs"""
        df = df.copy()

        interaction_count = 0
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns and interaction_count < max_combinations:
                # Multiplicative interaction
                df[f'{feat1}_{feat2}_mult'] = df[feat1] * df[feat2]

                # Additive interaction
                df[f'{feat1}_{feat2}_add'] = df[feat1] + df[feat2]

                # Ratio interaction (avoid division by zero)
                df[f'{feat1}_{feat2}_ratio'] = df[feat1] / (df[feat2] + 1e-8)

                interaction_count += 3

                if interaction_count >= max_combinations:
                    break

        return df

    @staticmethod
    def apply_pca_to_features(X_train, X_test=None, n_components=0.95, feature_prefix='feat_'):
        """Apply PCA to reduce dimensionality of specific feature groups"""
        # Select features for PCA
        feature_cols = [col for col in X_train.columns if col.startswith(feature_prefix)]

        if len(feature_cols) < 2:
            print(f"Not enough features with prefix '{feature_prefix}' for PCA")
            return X_train, X_test

        # Fit PCA
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train[feature_cols])

        # Create new column names
        pca_cols = [f'{feature_prefix}pca_{i}' for i in range(X_train_pca.shape[1])]

        # Replace original features with PCA components
        X_train_new = X_train.drop(columns=feature_cols).copy()
        X_train_pca_df = pd.DataFrame(X_train_pca, columns=pca_cols, index=X_train.index)
        X_train_new = pd.concat([X_train_new, X_train_pca_df], axis=1)

        X_test_new = None
        if X_test is not None:
            X_test_pca = pca.transform(X_test[feature_cols])
            X_test_new = X_test.drop(columns=feature_cols).copy()
            X_test_pca_df = pd.DataFrame(X_test_pca, columns=pca_cols, index=X_test.index)
            X_test_new = pd.concat([X_test_new, X_test_pca_df], axis=1)

        print(f"PCA applied: {len(feature_cols)} -> {X_train_pca.shape[1]} components")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")

        return X_train_new, X_test_new, pca

    @staticmethod
    def create_statistical_features(df, numeric_columns):
        """Create statistical features from numeric columns"""
        df = df.copy()

        if len(numeric_columns) > 1:
            numeric_data = df[numeric_columns]

            # Statistical aggregations
            df['numeric_mean'] = numeric_data.mean(axis=1)
            df['numeric_std'] = numeric_data.std(axis=1)
            df['numeric_min'] = numeric_data.min(axis=1)
            df['numeric_max'] = numeric_data.max(axis=1)
            df['numeric_range'] = df['numeric_max'] - df['numeric_min']
            df['numeric_skew'] = numeric_data.skew(axis=1)
            df['numeric_kurt'] = numeric_data.kurtosis(axis=1)

            # Count of zeros and non-zeros
            df['numeric_zeros_count'] = (numeric_data == 0).sum(axis=1)
            df['numeric_nonzeros_count'] = (numeric_data != 0).sum(axis=1)
            df['numeric_positive_count'] = (numeric_data > 0).sum(axis=1)
            df['numeric_negative_count'] = (numeric_data < 0).sum(axis=1)

        return df

    @staticmethod
    def handle_sequence_features(df, seq_column='seq', max_length=100):
        """Advanced sequence feature engineering"""
        df = df.copy()

        if seq_column not in df.columns:
            return df

        # Parse sequences
        def parse_sequence(seq_str):
            if pd.isna(seq_str):
                return []
            if isinstance(seq_str, str):
                return [int(x) for x in seq_str.split(',') if x.strip().isdigit()]
            return list(seq_str) if hasattr(seq_str, '__iter__') else [seq_str]

        df['parsed_seq'] = df[seq_column].apply(parse_sequence)

        # Sequence statistics
        df['seq_length'] = df['parsed_seq'].apply(len)
        df['seq_unique_count'] = df['parsed_seq'].apply(lambda x: len(set(x)) if x else 0)
        df['seq_repetition_rate'] = df.apply(
            lambda row: 1 - row['seq_unique_count'] / row['seq_length'] if row['seq_length'] > 0 else 0,
            axis=1
        )

        # First and last elements
        df['seq_first'] = df['parsed_seq'].apply(lambda x: x[0] if x else -1)
        df['seq_last'] = df['parsed_seq'].apply(lambda x: x[-1] if x else -1)

        # Most common element
        def get_most_common(seq):
            if not seq:
                return -1
            from collections import Counter
            return Counter(seq).most_common(1)[0][0]

        df['seq_most_common'] = df['parsed_seq'].apply(get_most_common)

        # Sequence trend (increasing/decreasing)
        def sequence_trend(seq):
            if len(seq) < 2:
                return 0
            diffs = [seq[i] - seq[i-1] for i in range(1, len(seq))]
            if not diffs:
                return 0
            return np.mean(diffs)

        df['seq_trend'] = df['parsed_seq'].apply(sequence_trend)

        # Drop the parsed sequence column
        df = df.drop('parsed_seq', axis=1)

        return df

    @staticmethod
    def create_time_based_aggregations(df, time_cols=['hour', 'day_of_week'], agg_features=None):
        """Create time-based aggregation features"""
        df = df.copy()

        if agg_features is None:
            agg_features = [col for col in df.columns if col.startswith(('feat_', 'history_'))]

        for time_col in time_cols:
            if time_col in df.columns:
                for agg_feat in agg_features:
                    if agg_feat in df.columns:
                        # Mean by time period
                        time_means = df.groupby(time_col)[agg_feat].mean()
                        df[f'{agg_feat}_{time_col}_mean'] = df[time_col].map(time_means)

                        # Standard deviation by time period
                        time_stds = df.groupby(time_col)[agg_feat].std()
                        df[f'{agg_feat}_{time_col}_std'] = df[time_col].map(time_stds)

                        # Deviation from time mean
                        df[f'{agg_feat}_{time_col}_dev'] = df[agg_feat] - df[f'{agg_feat}_{time_col}_mean']

        return df

    @staticmethod
    def plot_preprocessing_results(train_df, processed_df, sample_features=5):
        """Plot before/after preprocessing comparison"""
        # Select sample features
        numeric_features = train_df.select_dtypes(include=[np.number]).columns[:sample_features]

        fig, axes = plt.subplots(2, sample_features, figsize=(20, 10))

        for i, feature in enumerate(numeric_features):
            if feature in processed_df.columns:
                # Original distribution
                axes[0, i].hist(train_df[feature].dropna(), bins=50, alpha=0.7)
                axes[0, i].set_title(f'Original: {feature}')
                axes[0, i].set_ylabel('Frequency')

                # Processed distribution
                axes[1, i].hist(processed_df[feature].dropna(), bins=50, alpha=0.7, color='orange')
                axes[1, i].set_title(f'Processed: {feature}')
                axes[1, i].set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def generate_preprocessing_report(original_df, processed_df):
        """Generate comprehensive preprocessing report"""
        report = {
            'original_shape': original_df.shape,
            'processed_shape': processed_df.shape,
            'features_added': processed_df.shape[1] - original_df.shape[1],
            'memory_usage_mb': {
                'original': original_df.memory_usage(deep=True).sum() / 1024 / 1024,
                'processed': processed_df.memory_usage(deep=True).sum() / 1024 / 1024
            },
            'missing_values': {
                'original': original_df.isnull().sum().sum(),
                'processed': processed_df.isnull().sum().sum()
            }
        }

        print("="*50)
        print("PREPROCESSING REPORT")
        print("="*50)
        print(f"Original shape: {report['original_shape']}")
        print(f"Processed shape: {report['processed_shape']}")
        print(f"Features added: {report['features_added']}")
        print(f"Memory usage:")
        print(f"  Original: {report['memory_usage_mb']['original']:.2f} MB")
        print(f"  Processed: {report['memory_usage_mb']['processed']:.2f} MB")
        print(f"Missing values:")
        print(f"  Original: {report['missing_values']['original']}")
        print(f"  Processed: {report['missing_values']['processed']}")
        print("="*50)

        return report