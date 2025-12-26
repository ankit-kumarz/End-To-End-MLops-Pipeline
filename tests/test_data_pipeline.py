"""
Unit tests for data pipeline
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_pipeline import clean_data, engineer_features


class TestDataPipeline:
    """Test data pipeline functions"""
    
    def test_clean_data_removes_nulls(self):
        """Test that clean_data removes null values"""
        # Create test data with nulls
        df = pd.DataFrame({
            'feature_1': [1.0, 2.0, np.nan, 4.0],
            'feature_2': [5.0, 6.0, 7.0, 8.0],
            'target': [0, 1, 0, 1]
        })
        
        df_clean = clean_data(df)
        
        # Check no nulls remain
        assert df_clean.isnull().sum().sum() == 0
        # Check we have fewer rows
        assert len(df_clean) < len(df)
    
    def test_clean_data_removes_duplicates(self):
        """Test that clean_data removes duplicate rows"""
        # Create test data with duplicates
        df = pd.DataFrame({
            'feature_1': [1.0, 2.0, 1.0, 4.0],
            'feature_2': [5.0, 6.0, 5.0, 8.0],
            'target': [0, 1, 0, 1]
        })
        
        df_clean = clean_data(df)
        
        # Check no duplicates
        assert len(df_clean) == len(df_clean.drop_duplicates())
    
    def test_engineer_features_creates_expected_features(self):
        """Test that feature engineering creates all expected features"""
        # Create test data
        df = pd.DataFrame({
            'feature_1': [1.0, 2.0],
            'feature_2': [3.0, 4.0],
            'feature_3': [5.0, 6.0],
            'feature_4': [50.0, 75.0],
            'feature_5': [5, 7],
            'target': [0, 1]
        })
        
        df_features = engineer_features(df)
        
        # Check expected features exist
        expected_features = [
            'feature_1_squared',
            'feature_2_squared',
            'feature_1_x_feature_2',
            'feature_1_x_feature_4',
            'feature_4_per_feature_5',
            'feature_4_log',
            'bin_medium',  # One-hot encoded
            'bin_high',
            'bin_very_high'
        ]
        
        for feature in expected_features:
            assert feature in df_features.columns, f"Missing feature: {feature}"
    
    def test_engineer_features_polynomial(self):
        """Test polynomial feature creation"""
        df = pd.DataFrame({
            'feature_1': [2.0],
            'feature_2': [3.0],
            'feature_3': [1.0],
            'feature_4': [50.0],
            'feature_5': [5],
            'target': [0]
        })
        
        df_features = engineer_features(df)
        
        # Check polynomial features
        assert df_features['feature_1_squared'].iloc[0] == 4.0
        assert df_features['feature_2_squared'].iloc[0] == 9.0
    
    def test_engineer_features_interactions(self):
        """Test interaction feature creation"""
        df = pd.DataFrame({
            'feature_1': [2.0],
            'feature_2': [3.0],
            'feature_3': [1.0],
            'feature_4': [50.0],
            'feature_5': [5],
            'target': [0]
        })
        
        df_features = engineer_features(df)
        
        # Check interaction features
        assert df_features['feature_1_x_feature_2'].iloc[0] == 6.0
        assert df_features['feature_1_x_feature_4'].iloc[0] == 100.0
    
    def test_engineer_features_handles_edge_cases(self):
        """Test feature engineering with edge case values"""
        df = pd.DataFrame({
            'feature_1': [0.0],
            'feature_2': [0.0],
            'feature_3': [0.0],
            'feature_4': [0.0],
            'feature_5': [1],
            'target': [0]
        })
        
        df_features = engineer_features(df)
        
        # Should not raise errors
        assert not df_features.isnull().any().any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
