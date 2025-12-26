"""Check the training data"""
import pandas as pd

train = pd.read_csv("data/processed/train.csv")
print("Train data dtypes:")
print(train.dtypes)
print(f"\nTotal columns: {len(train.columns)}")
print(f"\nNumeric columns: {train.select_dtypes(include=['number']).columns.tolist()}")
print(f"\nTotal numeric: {len(train.select_dtypes(include=['number']).columns)}")
