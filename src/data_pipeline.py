"""
Data Preprocessing Pipeline

Transforms raw data into ML-ready features.
All steps are reproducible and tracked by DVC.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_raw_data(data_path: str) -> pd.DataFrame:
    """
    Load raw data from CSV.
    
    Args:
        data_path: Path to raw CSV file
        
    Returns:
        DataFrame with raw data
    """
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} samples from {data_path}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare data for feature engineering.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Handle missing values
    null_counts = df_clean.isnull().sum()
    if null_counts.any():
        print(f"⚠️  Found null values:\n{null_counts[null_counts > 0]}")
        df_clean = df_clean.dropna()
        print(f"✅ Dropped rows with nulls. New shape: {df_clean.shape}")
    
    # Remove duplicates
    original_len = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = original_len - len(df_clean)
    if duplicates_removed > 0:
        print(f"✅ Removed {duplicates_removed} duplicate rows")
    
    # Remove outliers (basic approach)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'target']
    
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
        if outliers > 0:
            df_clean = df_clean[
                (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            ]
            print(f"✅ Removed {outliers} outliers from '{col}'")
    
    print(f"✅ Cleaning complete. Final shape: {df_clean.shape}")
    return df_clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing ones.
    
    Args:
        df: Cleaned DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    df_features = df.copy()
    
    # Polynomial features
    df_features['feature_1_squared'] = df_features['feature_1'] ** 2
    df_features['feature_2_squared'] = df_features['feature_2'] ** 2
    
    # Interaction features
    df_features['feature_1_x_feature_2'] = (
        df_features['feature_1'] * df_features['feature_2']
    )
    df_features['feature_1_x_feature_4'] = (
        df_features['feature_1'] * df_features['feature_4']
    )
    
    # Ratio features
    df_features['feature_4_per_feature_5'] = (
        df_features['feature_4'] / (df_features['feature_5'] + 1)
    )
    
    # Log transforms (for skewed features)
    df_features['feature_4_log'] = np.log1p(df_features['feature_4'])
    
    # Binning
    df_features['feature_4_bin'] = pd.cut(
        df_features['feature_4'], 
        bins=[0, 25, 50, 75, 100],
        labels=['low', 'medium', 'high', 'very_high']
    )
    
    # One-hot encode categorical features
    df_features = pd.get_dummies(
        df_features, 
        columns=['feature_4_bin'],
        prefix='bin',
        drop_first=True
    )
    
    print(f"✅ Feature engineering complete. New shape: {df_features.shape}")
    return df_features


def scale_features(
    df: pd.DataFrame,
    scaler_path: str = "models/scaler.pkl",
    fit_scaler: bool = True
) -> pd.DataFrame:
    """
    Scale numeric features using StandardScaler.
    
    Args:
        df: DataFrame with features
        scaler_path: Path to save/load scaler
        fit_scaler: Whether to fit a new scaler or load existing
        
    Returns:
        DataFrame with scaled features
    """
    df_scaled = df.copy()
    
    # Identify numeric columns (exclude target)
    numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'target']
    
    if fit_scaler:
        # Fit and save scaler
        scaler = StandardScaler()
        df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
        
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"✅ Fitted and saved scaler to {scaler_path}")
    else:
        # Load existing scaler
        scaler = joblib.load(scaler_path)
        df_scaled[numeric_cols] = scaler.transform(df_scaled[numeric_cols])
        print(f"✅ Loaded and applied scaler from {scaler_path}")
    
    return df_scaled


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """
    Split data into train and test sets.
    
    Args:
        df: DataFrame with features and target
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintain class distribution
    )
    
    print(f"✅ Split data:")
    print(f"   Train set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    print(f"   Train target distribution:\n{y_train.value_counts()}")
    print(f"   Test target distribution:\n{y_test.value_counts()}")
    
    return X_train, X_test, y_train, y_test


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str = "data/processed"
):
    """
    Save processed datasets to CSV files.
    
    Args:
        X_train, X_test: Feature DataFrames
        y_train, y_test: Target Series
        output_dir: Directory to save files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save train set
    train_df = X_train.copy()
    train_df['target'] = y_train.values
    train_path = output_path / "train.csv"
    train_df.to_csv(train_path, index=False)
    print(f"✅ Saved training data to {train_path}")
    
    # Save test set
    test_df = X_test.copy()
    test_df['target'] = y_test.values
    test_path = output_path / "test.csv"
    test_df.to_csv(test_path, index=False)
    print(f"✅ Saved test data to {test_path}")
    
    # Save metadata
    metadata = {
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': X_train.shape[1],
        'feature_names': list(X_train.columns),
        'train_target_distribution': y_train.value_counts().to_dict(),
        'test_target_distribution': y_test.value_counts().to_dict()
    }
    
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved metadata to {metadata_path}")


def run_preprocessing_pipeline(config: dict):
    """
    Run the complete preprocessing pipeline.
    
    Args:
        config: Configuration dictionary
    """
    print("\n" + "="*60)
    print("🔄 STARTING PREPROCESSING PIPELINE")
    print("="*60 + "\n")
    
    # Load raw data
    print("📥 STEP 1: Loading raw data...")
    df = load_raw_data(config['data']['raw_data_path'] + "/dataset.csv")
    
    # Clean data
    print("\n🧹 STEP 2: Cleaning data...")
    df_clean = clean_data(df)
    
    # Engineer features
    print("\n🔧 STEP 3: Engineering features...")
    df_features = engineer_features(df_clean)
    
    # Scale features
    print("\n📏 STEP 4: Scaling features...")
    df_scaled = scale_features(df_features, fit_scaler=True)
    
    # Split data
    print("\n✂️  STEP 5: Splitting data...")
    X_train, X_test, y_train, y_test = split_data(
        df_scaled,
        test_size=config['data']['train_test_split'],
        random_state=config['data']['random_seed']
    )
    
    # Save processed data
    print("\n💾 STEP 6: Saving processed data...")
    save_processed_data(
        X_train, X_test, y_train, y_test,
        output_dir=config['data']['processed_data_path']
    )
    
    print("\n" + "="*60)
    print("✅ PREPROCESSING PIPELINE COMPLETE")
    print("="*60)


if __name__ == "__main__":
    config = load_config()
    run_preprocessing_pipeline(config)
