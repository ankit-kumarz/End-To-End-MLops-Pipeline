"""
Unit tests for data validation
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from validate_data import DataValidator


class TestDataValidator:
    """Test DataValidator class"""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance"""
        return DataValidator()
    
    @pytest.fixture
    def valid_data(self):
        """Create valid test data"""
        return pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'feature_3': np.random.randn(100),
            'feature_4': np.random.uniform(0, 100, 100),
            'feature_5': np.random.randint(1, 10, 100),
            'target': np.random.randint(0, 2, 100)
        })
    
    def test_validate_schema_success(self, validator, valid_data):
        """Test schema validation with valid data"""
        expected_schema = {
            'feature_1': 'float64',
            'feature_2': 'float64',
            'feature_3': 'float64',
            'feature_4': 'float64',
            'feature_5': 'int',
            'target': 'int'
        }
        
        result = validator.validate_schema(valid_data, expected_schema)
        assert result is True
    
    def test_validate_schema_failure(self, validator):
        """Test schema validation with invalid data"""
        df = pd.DataFrame({
            'feature_1': ['string', 'data'],  # Wrong type
            'target': [0, 1]
        })
        
        expected_schema = {
            'feature_1': 'float64',
            'target': 'int'
        }
        
        result = validator.validate_schema(df, expected_schema)
        assert result is False
    
    def test_validate_nulls_success(self, validator, valid_data):
        """Test null validation with no nulls"""
        result = validator.validate_nulls(valid_data)
        assert result is True
    
    def test_validate_nulls_failure(self, validator):
        """Test null validation with nulls present"""
        df = pd.DataFrame({
            'feature_1': [1.0, 2.0, np.nan],
            'target': [0, 1, 0]
        })
        
        result = validator.validate_nulls(df)
        assert result is False
    
    def test_validate_ranges_success(self, validator):
        """Test range validation with valid ranges"""
        df = pd.DataFrame({
            'feature_4': [50.0, 75.0, 25.0],
            'feature_5': [5, 7, 3],
            'target': [0, 1, 0]
        })
        
        range_constraints = {
            'feature_4': (0, 100),
            'feature_5': (1, 10)
        }
        
        result = validator.validate_ranges(df, range_constraints)
        assert result is True
    
    def test_validate_ranges_failure(self, validator):
        """Test range validation with out-of-range values"""
        df = pd.DataFrame({
            'feature_4': [50.0, 150.0, 25.0],  # 150 is out of range
            'target': [0, 1, 0]
        })
        
        range_constraints = {
            'feature_4': (0, 100)
        }
        
        result = validator.validate_ranges(df, range_constraints)
        assert result is False
    
    def test_validate_target_distribution_success(self, validator):
        """Test target distribution validation"""
        df = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'target': np.random.choice([0, 1], 100, p=[0.6, 0.4])
        })
        
        result = validator.validate_target_distribution(df, 'target', 
                                                       min_samples_per_class=10)
        assert result is True
    
    def test_validate_target_distribution_failure(self, validator):
        """Test target distribution with imbalanced classes"""
        df = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'target': [0] * 99 + [1]  # Only 1 sample of class 1
        })
        
        result = validator.validate_target_distribution(df, 'target', 
                                                       min_samples_per_class=10)
        assert result is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
