"""
Debug script to test predictor in same way as Streamlit
"""
import sys
sys.path.insert(0, 'app')

from streamlit_app import SimplePredictor
import numpy as np

print("="*60)
print("DEBUG: Testing Predictor")
print("="*60)

# Create predictor
predictor = SimplePredictor()

print(f"\nModel loaded: {predictor.model is not None}")
print(f"X_train loaded: {predictor.X_train is not None}")
print(f"train_original loaded: {predictor.train_original is not None}")

if predictor.X_train is not None:
    print(f"X_train shape: {predictor.X_train.shape}")

# Test prediction for house 1
idx = 1
print(f"\n{'='*60}")
print(f"Testing prediction for house {idx}")
print(f"{'='*60}")

house = predictor.get_house_summary(idx)
print(f"\nHouse details:")
print(f"  Neighborhood: {house['Neighborhood']}")
print(f"  Living Area: {house['Living Area (sq ft)']} sqft")
print(f"  Actual Price: {house['Actual Price']}")

# Make prediction
pred_price, individual_preds, pred_log = predictor.predict_by_index(idx)

print(f"\nPrediction results:")
print(f"  pred_log: {pred_log}")
print(f"  pred_price: ${pred_price:,.2f}")

print(f"\nIndividual predictions:")
for model, price in individual_preds.items():
    print(f"  {model}: ${price:,.2f}")

# Check for issues
actual_price_val = float(house['Actual Price'].replace('$', '').replace(',', ''))
error_pct = abs(pred_price - actual_price_val) / actual_price_val * 100

print(f"\nValidation:")
print(f"  Actual: ${actual_price_val:,.0f}")
print(f"  Predicted: ${pred_price:,.0f}")
print(f"  Error: {error_pct:.2f}%")

if pred_price < 100:
    print(f"\n[ERROR] Predicted price is suspiciously low!")
    print(f"  This indicates a bug in the prediction logic")
else:
    print(f"\n[OK] Prediction looks reasonable")
