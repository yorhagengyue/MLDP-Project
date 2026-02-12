"""
Test script to verify real model predictions are working
"""
import sys
import os
import pandas as pd
import numpy as np
import pickle

print("="*60)
print("TESTING REAL MODEL PREDICTIONS")
print("="*60)

# Load model
model_path = 'models/best_model_xgboost.pkl'
print(f"\n1. Loading model from: {model_path}")
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"   [OK] Model loaded successfully")
    print(f"   Model type: {type(model).__name__}")
except Exception as e:
    print(f"   [ERROR] {e}")
    sys.exit(1)

# Load processed features
x_train_path = 'data/X_train_clean.csv'
print(f"\n2. Loading processed features from: {x_train_path}")
try:
    X_train = pd.read_csv(x_train_path)
    print(f"   [OK] Loaded features: {X_train.shape}")
except Exception as e:
    print(f"   [ERROR] {e}")
    sys.exit(1)

# Load original data for comparison
train_path = 'data/train.csv'
print(f"\n3. Loading original data from: {train_path}")
try:
    train_original = pd.read_csv(train_path)
    print(f"   [OK] Loaded original data: {train_original.shape}")
except Exception as e:
    print(f"   [ERROR] {e}")
    sys.exit(1)

# Make predictions for first 5 houses
print("\n4. Testing real predictions:")
print("-"*60)
print(f"{'House':<8} {'Actual Price':<15} {'Predicted Price':<18} {'Error %':<10}")
print("-"*60)

for idx in range(5):
    # Get actual price
    actual = train_original.iloc[idx]['SalePrice']

    # Get processed features and predict
    features = X_train.iloc[[idx]]
    pred_log = model.predict(features)[0]
    pred_price = np.expm1(pred_log)

    # Calculate error
    error_pct = abs(pred_price - actual) / actual * 100

    print(f"{idx:<8} ${actual:>12,.0f}  ${pred_price:>15,.0f}  {error_pct:>8.2f}%")

print("-"*60)

# Summary
print("\n5. Summary:")
print("   [OK] Real XGBoost predictions are working!")
print("   [OK] Model is loaded from best_model_xgboost.pkl")
print("   [OK] Features are loaded from X_train_clean.csv")
print("   [OK] Predictions are NOT simulated - they are REAL")
print("\n" + "="*60)
