"""Quick inline API test"""
import requests
import json

def test_api():
    base_url = "http://localhost:8001"
    
    print("\n" + "="*60)
    print("🧪 QUICK API TEST")
    print("="*60)
    
    # Test 1: Health
    print("\n1️⃣ Health Check:")
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        print(f"   Status: {r.status_code}")
        print(f"   Response: {json.dumps(r.json(), indent=2)}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Test 2: Single Prediction
    print("\n2️⃣ Single Prediction:")
    payload = {
        "feature_1": 0.5,
        "feature_2": 1.2,
        "feature_3": -0.3,
        "feature_4": 75.5,
        "feature_5": 5
    }
    try:
        r = requests.post(f"{base_url}/predict", json=payload, timeout=5)
        print(f"   Status: {r.status_code}")
        print(f"   Response: {json.dumps(r.json(), indent=2)}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "="*60)
    print("✅ Tests Complete!")
    print("="*60)

if __name__ == "__main__":
    test_api()
