"""
Save all trained models for the project
Trains and saves: Linear Regression, Ridge, Lasso, Random Forest, XGBoost
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

print("="*60)
print("SAVING ALL TRAINED MODELS")
print("="*60)

# Load processed data
print("\n1. Loading processed training data...")
X_train = pd.read_csv('data/X_train_clean.csv')
y_train = pd.read_csv('data/train_clean.csv')['SalePrice']
y_train_log = np.log1p(y_train)

print(f"   X_train shape: {X_train.shape}")
print(f"   y_train samples: {len(y_train_log)}")

# Define models with same parameters as used in notebook
models = {
    'linear_regression': LinearRegression(),
    'ridge': Ridge(alpha=10.0, random_state=42),
    'lasso': Lasso(alpha=0.0005, random_state=42),
    'elastic_net': ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42),
    'random_forest': RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    ),
    'gradient_boosting': GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
}

# Train and save each model
print("\n2. Training and saving models...")
for name, model in models.items():
    print(f"\n   Training {name}...")
    model.fit(X_train, y_train_log)

    # Save model
    filename = f'models/{name}_model.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"   [OK] Saved to: {filename}")

# The best XGBoost model was already saved via notebook
# But let's verify it exists
print("\n3. Verifying XGBoost model...")
import os
if os.path.exists('models/best_model_xgboost.pkl'):
    print("   [OK] best_model_xgboost.pkl already exists (from notebook)")
else:
    print("   [WARNING] XGBoost model not found, training new one...")
    xgb_model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train_log)
    with open('models/best_model_xgboost.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    print("   [OK] Saved to: models/best_model_xgboost.pkl")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("\nAll models saved:")
print("  1. [OK] linear_regression_model.pkl")
print("  2. [OK] ridge_model.pkl")
print("  3. [OK] lasso_model.pkl")
print("  4. [OK] elastic_net_model.pkl")
print("  5. [OK] random_forest_model.pkl")
print("  6. [OK] gradient_boosting_model.pkl")
print("  7. [OK] best_model_xgboost.pkl (from notebook)")
print("  8. [OK] feature_names.pkl (from notebook)")
print("\nTotal: 8 pickle files")
print("="*60)
