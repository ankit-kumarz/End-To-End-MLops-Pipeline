"""Debug script to check scaler feature names"""
import joblib

scaler = joblib.load("models/scaler.pkl")

print("Scaler feature names:")
if hasattr(scaler, 'feature_names_in_'):
    print(scaler.feature_names_in_)
    print(f"\nTotal features: {len(scaler.feature_names_in_)}")
else:
    print("Scaler doesn't have feature_names_in_ attribute")
