import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class CTRDataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_stats = {}

    def load_data(self, train_path, test_path=None):
        """Load train and test data"""
        print("Loading data...")
        self.train_df = pd.read_parquet(train_path)
        if test_path:
            self.test_df = pd.read_parquet(test_path)
        print(f"Train shape: {self.train_df.shape}")
        if test_path:
            print(f"Test shape: {self.test_df.shape}")
        return self

    def handle_missing_values(self, df):
        """Handle missing values based on EDA findings"""
        df = df.copy()

        # Handle gender and age_group missing values (0.16% missing rate)
        # Fill with mode or create 'unknown' category
        if 'gender' in df.columns:
            df['gender'] = df['gender'].fillna(df['gender'].mode()[0] if not df['gender'].mode().empty else 1.0)

        if 'age_group' in df.columns:
            df['age_group'] = df['age_group'].fillna(df['age_group'].mode()[0] if not df['age_group'].mode().empty else 1.0)

        # Handle numeric features - use median for robustness
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())

        return df

    def engineer_features(self, df):
        """Feature engineering based on EDA insights"""
        df = df.copy()

        # Time-based features
        if 'hour' in df.columns:
            # Ensure hour is numeric
            df['hour'] = pd.to_numeric(df['hour'], errors='coerce')
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['is_peak_hour'] = df['hour'].isin([9, 10, 11, 19, 20, 21]).astype(int)

        if 'day_of_week' in df.columns:
            # Ensure day_of_week is numeric
            df['day_of_week'] = pd.to_numeric(df['day_of_week'], errors='coerce')
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Sequence length features (based on seq_length_stats)
        if 'seq' in df.columns:
            # Parse sequence if it's string format
            if df['seq'].dtype == 'object':
                df['seq_length'] = df['seq'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
            else:
                df['seq_length'] = df['seq'].apply(lambda x: len(x) if hasattr(x, '__len__') else 0)

            # Sequence length binning based on percentiles
            df['seq_length_bin'] = pd.cut(df['seq_length'],
                                        bins=[0, 439, 779, 1124, 1345, float('inf')],
                                        labels=['short', 'medium', 'long', 'very_long', 'extreme'])

        # Feature interactions
        if 'gender' in df.columns and 'age_group' in df.columns:
            df['gender_age_interaction'] = df['gender'].astype(str) + '_' + df['age_group'].astype(str)

        # Numeric feature transformations
        feat_cols = [col for col in df.columns if col.startswith(('feat_', 'history_'))]

        for col in feat_cols:
            if col in df.columns:
                # Ensure column is numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')

                # Log transform for highly skewed features
                if df[col].min() >= 0:
                    df[f'{col}_log1p'] = np.log1p(df[col])

                # Square root transform
                if df[col].min() >= 0:
                    df[f'{col}_sqrt'] = np.sqrt(df[col])

                # Binning based on quantiles
                try:
                    df[f'{col}_bin'] = pd.qcut(df[col], q=5, labels=False, duplicates='drop')
                except:
                    # Fallback to simple binning if qcut fails
                    df[f'{col}_bin'] = pd.cut(df[col], bins=5, labels=False)

        return df

    def encode_categorical_features(self, df, is_training=True):
        """Encode categorical features"""
        df = df.copy()

        categorical_cols = ['gender', 'age_group', 'inventory_id', 'l_feat_14',
                          'seq_length_bin', 'gender_age_interaction']

        for col in categorical_cols:
            if col in df.columns:
                if is_training:
                    self.encoders[col] = LabelEncoder()
                    df[col] = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.encoders:
                        # Handle unseen categories
                        unique_vals = set(self.encoders[col].classes_)
                        df[col] = df[col].astype(str).apply(
                            lambda x: x if x in unique_vals else 'unknown'
                        )
                        df[col] = self.encoders[col].transform(df[col])

        return df

    def scale_numeric_features(self, df, is_training=True):
        """Scale numeric features"""
        df = df.copy()

        # Use RobustScaler for better handling of outliers
        numeric_cols = [col for col in df.columns if col.startswith(('feat_', 'history_'))
                       and not col.endswith(('_bin', '_log1p', '_sqrt'))]

        if is_training:
            self.scalers['robust'] = RobustScaler()
            df[numeric_cols] = self.scalers['robust'].fit_transform(df[numeric_cols])
        else:
            if 'robust' in self.scalers:
                df[numeric_cols] = self.scalers['robust'].transform(df[numeric_cols])

        return df

    def handle_outliers(self, df, method='iqr', factor=1.5):
        """Handle outliers using IQR method"""
        df = df.copy()

        numeric_cols = [col for col in df.columns if col.startswith(('feat_', 'history_'))]

        for col in numeric_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR

                # Cap outliers instead of removing them
                df[col] = df[col].clip(lower_bound, upper_bound)

        return df

    def create_target_encoding(self, df, target_col='clicked', is_training=True):
        """Create target encoding for high cardinality categorical features"""
        df = df.copy()

        high_card_cols = ['inventory_id']

        if is_training and target_col in df.columns:
            for col in high_card_cols:
                if col in df.columns:
                    # Calculate mean target by category with smoothing
                    target_mean = df[target_col].mean()
                    category_stats = df.groupby(col)[target_col].agg(['mean', 'count'])

                    # Smoothing factor
                    smoothing = 100
                    category_stats['smoothed_mean'] = (
                        (category_stats['mean'] * category_stats['count'] +
                         target_mean * smoothing) /
                        (category_stats['count'] + smoothing)
                    )

                    self.feature_stats[f'{col}_target_enc'] = category_stats['smoothed_mean'].to_dict()
                    df[f'{col}_target_enc'] = df[col].map(self.feature_stats[f'{col}_target_enc'])
        else:
            for col in high_card_cols:
                if col in df.columns and f'{col}_target_enc' in self.feature_stats:
                    target_mean = list(self.feature_stats[f'{col}_target_enc'].values())[0]  # fallback
                    df[f'{col}_target_enc'] = df[col].map(self.feature_stats[f'{col}_target_enc']).fillna(target_mean)

        return df

    def preprocess_pipeline(self, df, is_training=True, target_col='clicked', pbar=None):
        """Complete preprocessing pipeline"""
        print(f"Starting preprocessing... Shape: {df.shape}")

        # 1. Handle missing values
        df = self.handle_missing_values(df)
        print("✓ Missing values handled")
        if pbar: pbar.update(1)

        # 2. Feature engineering
        df = self.engineer_features(df)
        print("✓ Feature engineering completed")
        if pbar: pbar.update(1)

        # 3. Target encoding
        df = self.create_target_encoding(df, target_col, is_training)
        print("✓ Target encoding completed")
        if pbar: pbar.update(1)

        # 4. Handle outliers
        df = self.handle_outliers(df)
        print("✓ Outliers handled")
        if pbar: pbar.update(1)

        # 5. Encode categorical features
        df = self.encode_categorical_features(df, is_training)
        print("✓ Categorical features encoded")
        if pbar: pbar.update(1)

        # 6. Scale numeric features
        df = self.scale_numeric_features(df, is_training)
        print("✓ Numeric features scaled")
        if pbar: pbar.update(1)

        print(f"Preprocessing completed! Final shape: {df.shape}")
        return df

    def prepare_train_validation_split(self, test_size=0.2, random_state=42):
        """Prepare train/validation split"""
        if not hasattr(self, 'train_df_processed'):
            raise ValueError("Please run preprocess_pipeline on training data first")

        # Separate features and target
        feature_cols = [col for col in self.train_df_processed.columns if col != 'clicked']
        X = self.train_df_processed[feature_cols]
        y = self.train_df_processed['clicked']

        # Stratified split to maintain class balance
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        return X_train, X_val, y_train, y_val

# Usage example
def main():
    # Initialize preprocessor
    preprocessor = CTRDataPreprocessor()

    # Load data
    train_path = '/Users/gimchaeyeon/Documents/2025/class_datas/data/train.parquet'
    test_path = '/Users/gimchaeyeon/Documents/2025/class_datas/data/test.parquet'

    preprocessor.load_data(train_path, test_path)

    # Preprocess training data
    print("Processing training data...")
    train_processed = preprocessor.preprocess_pipeline(
        preprocessor.train_df,
        is_training=True,
        target_col='clicked'
    )
    preprocessor.train_df_processed = train_processed

    # Preprocess test data
    print("Processing test data...")
    test_processed = preprocessor.preprocess_pipeline(
        preprocessor.test_df,
        is_training=False
    )
    preprocessor.test_df_processed = test_processed

    # Create train/validation split
    X_train, X_val, y_train, y_val = preprocessor.prepare_train_validation_split()

    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {test_processed.shape}")

    # Save processed data
    X_train.to_parquet('data/X_train_processed.parquet')
    X_val.to_parquet('data/X_val_processed.parquet')
    y_train.to_parquet('data/y_train.parquet')
    y_val.to_parquet('data/y_val.parquet')
    test_processed.to_parquet('data/test_processed.parquet')

    print("Processed data saved!")

    return preprocessor, X_train, X_val, y_train, y_val, test_processed

if __name__ == "__main__":
    preprocessor, X_train, X_val, y_train, y_val, test_processed = main()