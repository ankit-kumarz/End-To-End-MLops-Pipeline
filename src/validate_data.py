"""
Data Validation Module

Industry-grade data validation checks before training.
Prevents bad data from entering the ML pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json


class DataValidator:
    """Validates data quality and schema compliance."""
    
    def __init__(self, config: Dict):
        """
        Initialize validator with configuration.
        
        Args:
            config: Dictionary with validation rules
        """
        self.config = config
        self.errors = []
        self.warnings = []
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Check if DataFrame matches expected schema.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if schema is valid, False otherwise
        """
        expected_columns = self.config.get('expected_columns', [])
        
        # Check missing columns
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            self.errors.append(f"Missing columns: {missing_cols}")
            return False
        
        # Check extra columns
        extra_cols = set(df.columns) - set(expected_columns)
        if extra_cols:
            self.warnings.append(f"Extra columns found: {extra_cols}")
        
        return True
    
    def validate_nulls(self, df: pd.DataFrame) -> bool:
        """
        Check for null values.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if null check passes, False otherwise
        """
        null_counts = df.isnull().sum()
        max_null_pct = self.config.get('max_null_percentage', 0.1)
        
        for col, null_count in null_counts.items():
            null_pct = null_count / len(df)
            if null_pct > max_null_pct:
                self.errors.append(
                    f"Column '{col}' has {null_pct:.1%} nulls (max: {max_null_pct:.1%})"
                )
                return False
        
        return True
    
    def validate_ranges(self, df: pd.DataFrame) -> bool:
        """
        Check if numeric values are within expected ranges.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if ranges are valid, False otherwise
        """
        ranges = self.config.get('value_ranges', {})
        
        for col, (min_val, max_val) in ranges.items():
            if col not in df.columns:
                continue
            
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min < min_val or col_max > max_val:
                self.errors.append(
                    f"Column '{col}' out of range: [{col_min}, {col_max}] "
                    f"expected [{min_val}, {max_val}]"
                )
                return False
        
        return True
    
    def validate_data_drift(self, df: pd.DataFrame, reference_stats: Dict) -> bool:
        """
        Check for data drift compared to reference statistics.
        
        Args:
            df: DataFrame to validate
            reference_stats: Reference statistics to compare against
            
        Returns:
            True if no significant drift detected, False otherwise
        """
        drift_threshold = self.config.get('drift_threshold', 0.3)
        
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in reference_stats:
                continue
            
            current_mean = df[col].mean()
            reference_mean = reference_stats[col]['mean']
            
            # Calculate relative change
            if reference_mean != 0:
                drift = abs(current_mean - reference_mean) / abs(reference_mean)
                
                if drift > drift_threshold:
                    self.warnings.append(
                        f"Potential drift in '{col}': {drift:.1%} change from baseline"
                    )
        
        return True
    
    def validate_target_distribution(self, df: pd.DataFrame, target_col: str) -> bool:
        """
        Check if target variable distribution is reasonable.
        
        Args:
            df: DataFrame to validate
            target_col: Name of target column
            
        Returns:
            True if distribution is valid, False otherwise
        """
        if target_col not in df.columns:
            self.errors.append(f"Target column '{target_col}' not found")
            return False
        
        # Check for class imbalance
        target_counts = df[target_col].value_counts()
        total = len(df)
        
        min_class_pct = self.config.get('min_class_percentage', 0.05)
        
        for class_label, count in target_counts.items():
            class_pct = count / total
            if class_pct < min_class_pct:
                self.warnings.append(
                    f"Class imbalance: Class '{class_label}' has only {class_pct:.1%} "
                    f"of samples (min: {min_class_pct:.1%})"
                )
        
        return True
    
    def run_all_validations(
        self, 
        df: pd.DataFrame,
        target_col: str = 'target',
        reference_stats: Dict = None
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Run all validation checks.
        
        Args:
            df: DataFrame to validate
            target_col: Name of target column
            reference_stats: Optional reference statistics for drift detection
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        # Run all checks
        checks = [
            self.validate_schema(df),
            self.validate_nulls(df),
            self.validate_ranges(df),
            self.validate_target_distribution(df, target_col)
        ]
        
        if reference_stats:
            checks.append(self.validate_data_drift(df, reference_stats))
        
        is_valid = all(checks) and len(self.errors) == 0
        
        return is_valid, self.errors, self.warnings


def save_reference_stats(df: pd.DataFrame, output_path: str):
    """
    Save reference statistics for future drift detection.
    
    Args:
        df: DataFrame to extract stats from
        output_path: Path to save statistics JSON
    """
    stats = {}
    
    for col in df.select_dtypes(include=[np.number]).columns:
        stats[col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'median': float(df[col].median())
        }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ Saved reference statistics to {output_path}")


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("data/raw/dataset.csv")
    
    config = {
        'expected_columns': ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'target'],
        'max_null_percentage': 0.05,
        'value_ranges': {
            'feature_1': (-5, 5),
            'feature_2': (-5, 5),
            'feature_4': (0, 100)
        },
        'min_class_percentage': 0.1
    }
    
    validator = DataValidator(config)
    is_valid, errors, warnings = validator.run_all_validations(df)
    
    print("\n" + "="*60)
    print("📊 DATA VALIDATION REPORT")
    print("="*60)
    print(f"Dataset shape: {df.shape}")
    print(f"Validation status: {'✅ PASSED' if is_valid else '❌ FAILED'}")
    
    if errors:
        print(f"\n🚨 Errors ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")
    
    if warnings:
        print(f"\n⚠️  Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  - {warning}")
    
    if is_valid:
        save_reference_stats(df, "data/processed/reference_stats.json")
