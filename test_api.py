"""
Test script for the Model Serving API

This demonstrates how to call the API endpoints programmatically.
"""

import requests
import json
from pprint import pprint


# API base URL
    BASE_URL = "http://localhost:8001"


def test_root():
    """Test the root endpoint."""
    print("\n" + "="*60)
    print("🔍 Testing Root Endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    pprint(response.json())


def test_health():
    """Test the health check endpoint."""
    print("\n" + "="*60)
    print("💚 Testing Health Check")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    pprint(response.json())


def test_model_info():
    """Test the model info endpoint."""
    print("\n" + "="*60)
    print("📊 Testing Model Info")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    pprint(response.json())


def test_single_prediction():
    """Test single prediction endpoint."""
    print("\n" + "="*60)
    print("🎯 Testing Single Prediction")
    print("="*60)
    
    # Example input (these are just sample values)
    payload = {
        "feature_1": 0.5,
        "feature_2": 1.2,
        "feature_3": -0.3,
        "feature_4": 75.5,
        "feature_5": 5
    }
    
    print("Input:")
    pprint(payload)
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"\nStatus Code: {response.status_code}")
    print("Output:")
    pprint(response.json())


def test_batch_prediction():
    """Test batch prediction endpoint."""
    print("\n" + "="*60)
    print("📦 Testing Batch Prediction")
    print("="*60)
    
    # Example batch input
    payload = {
        "instances": [
            {
                "feature_1": 0.5,
                "feature_2": 1.2,
                "feature_3": -0.3,
                "feature_4": 75.5,
                "feature_5": 5
            },
            {
                "feature_1": -0.3,
                "feature_2": 0.8,
                "feature_3": 0.2,
                "feature_4": 45.0,
                "feature_5": 3
            },
            {
                "feature_1": 1.0,
                "feature_2": -0.5,
                "feature_3": 0.8,
                "feature_4": 90.0,
                "feature_5": 8
            }
        ]
    }
    
    print(f"Input: {len(payload['instances'])} instances")
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=payload)
    print(f"\nStatus Code: {response.status_code}")
    print("Output:")
    result = response.json()
    print(f"Predictions count: {result['count']}")
    for i, pred in enumerate(result['predictions'], 1):
        print(f"\nPrediction {i}:")
        pprint(pred)


def test_invalid_input():
    """Test error handling with invalid input."""
    print("\n" + "="*60)
    print("❌ Testing Error Handling (Invalid Input)")
    print("="*60)
    
    # Invalid input: feature_4 out of range
    payload = {
        "feature_1": 0.5,
        "feature_2": 1.2,
        "feature_3": -0.3,
        "feature_4": 150.0,  # Invalid: should be 0-100
        "feature_5": 5
    }
    
    print("Input (with invalid feature_4=150):")
    pprint(payload)
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"\nStatus Code: {response.status_code}")
    print("Output:")
    pprint(response.json())


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("🚀 STARTING API TESTS")
    print("="*60)
    print("\nMake sure the API server is running:")
    print("  python src/serve.py")
    print("\nOr:")
    print("  uvicorn src.serve:app --reload --port 8000")
    
    try:
        # Run tests
        test_root()
        test_health()
        test_model_info()
        test_single_prediction()
        test_batch_prediction()
        test_invalid_input()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\n" + "="*60)
        print("❌ ERROR: Cannot connect to API server")
        print("="*60)
        print("\nPlease start the API server first:")
        print("  python src/serve.py")
        print("\nOr:")
        print("  uvicorn src.serve:app --reload --port 8000")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
