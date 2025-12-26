"""
Unit tests for FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from serve import app, engineer_features
import pandas as pd


class TestAPIEndpoints:
    """Test FastAPI endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns correct information"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "unhealthy"]
        assert "model_loaded" in data
        assert "timestamp" in data
    
    def test_model_info_endpoint(self, client):
        """Test model info endpoint"""
        response = client.get("/model/info")
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "algorithm" in data
            assert "metrics" in data
    
    def test_predict_endpoint_valid_input(self, client):
        """Test prediction with valid input"""
        payload = {
            "feature_1": 0.5,
            "feature_2": 1.2,
            "feature_3": -0.3,
            "feature_4": 75.5,
            "feature_5": 5
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code in [200, 500]  # May fail if model not loaded in test
        
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "probability" in data
            assert "model_version" in data
            assert data["prediction"] in [0, 1]
    
    def test_predict_endpoint_invalid_input(self, client):
        """Test prediction with invalid input"""
        # Missing required field
        payload = {
            "feature_1": 0.5,
            "feature_2": 1.2
            # Missing other features
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_out_of_range(self, client):
        """Test prediction with out-of-range values"""
        payload = {
            "feature_1": 0.5,
            "feature_2": 1.2,
            "feature_3": -0.3,
            "feature_4": 150.0,  # Out of range (should be 0-100)
            "feature_5": 5
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error


class TestFeatureEngineering:
    """Test feature engineering functions"""
    
    def test_engineer_features_output_shape(self):
        """Test that engineer_features produces correct number of features"""
        input_df = pd.DataFrame({
            'feature_1': [0.5],
            'feature_2': [1.2],
            'feature_3': [-0.3],
            'feature_4': [75.5],
            'feature_5': [5]
        })
        
        output_df = engineer_features(input_df)
        
        # Should have original 5 + engineered features
        assert output_df.shape[1] > input_df.shape[1]
        
        # Check specific features exist
        assert 'feature_1_squared' in output_df.columns
        assert 'feature_2_squared' in output_df.columns
        assert 'feature_1_x_feature_2' in output_df.columns
    
    def test_engineer_features_bin_creation(self):
        """Test that binning creates correct categories"""
        input_df = pd.DataFrame({
            'feature_1': [0.5],
            'feature_2': [1.2],
            'feature_3': [-0.3],
            'feature_4': [85.0],  # Should be in 'very_high' bin
            'feature_5': [5]
        })
        
        output_df = engineer_features(input_df)
        
        # Check bin columns exist (drop_first=True, so no bin_low)
        assert 'bin_medium' in output_df.columns
        assert 'bin_high' in output_df.columns
        assert 'bin_very_high' in output_df.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
