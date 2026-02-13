import pandas as pd
import numpy as np
import pickle
import os

class SimplePredictor:
    def __init__(self, model_path='../models/clean_ensemble.pkl', data_path='../data'):
        """Initialize the predictor with model and data paths"""
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.X_train = None
        self.y_train = None
        self.train_original = None
        
        # Load resources
        self._load_model()
        self._load_data()
        
    def _load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
            else:
                print(f"Warning: Model not found at {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")

    def _load_data(self):
        """Load training data for sampling"""
        try:
            # Load processed data
            x_train_path = os.path.join(self.data_path, 'X_train_clean.csv')
            if os.path.exists(x_train_path):
                self.X_train = pd.read_csv(x_train_path)
            
            # Load original data for display
            train_orig_path = os.path.join(self.data_path, 'train.csv')
            if os.path.exists(train_orig_path):
                self.train_original = pd.read_csv(train_orig_path)
                
        except Exception as e:
            print(f"Error loading data: {str(e)}")

    def get_sample_indices(self, n=20):
        """Get indices of n sample houses"""
        if self.train_original is not None:
            return list(range(min(n, len(self.train_original))))
        return []

    def get_house_summary(self, idx):
        """Get summary stats for a specific house"""
        if self.train_original is None:
            return {}
            
        house = self.train_original.iloc[idx]
        return {
            'Neighborhood': house['Neighborhood'],
            'House Style': house['HouseStyle'],
            'Year Built': int(house['YearBuilt']),
            'Overall Quality': int(house['OverallQual']),
            'Living Area (sq ft)': int(house['GrLivArea']),
            'Total Rooms': int(house['TotRmsAbvGrd']),
            'Bedrooms': int(house['BedroomAbvGr']),
            'Bathrooms': int(house['FullBath']) + 0.5 * int(house['HalfBath']),
            'Garage Cars': int(house['GarageCars']),
            'Actual Price': f"${house['SalePrice']:,.0f}"
        }

    def predict_by_index(self, idx):
        """Make prediction for a house by its index in training set"""
        if self.model is None or self.X_train is None:
            return 0, {}, 0
            
        # Get features for this house
        features = self.X_train.iloc[[idx]]
        
        # Make prediction (log scale)
        pred_log = self.model.predict(features)[0]
        
        # Convert back to price
        pred_price = np.expm1(pred_log)
        
        # Mock individual model predictions for demo (since we loaded an ensemble)
        # In a real scenario, we would access base models from the ensemble
        individual_preds = {
            'Ridge': np.expm1(pred_log * 0.98),
            'Lasso': np.expm1(pred_log * 1.01),
            'ElasticNet': np.expm1(pred_log * 0.99),
            'GradientBoosting': np.expm1(pred_log * 1.02)
        }
        
        return pred_price, individual_preds, pred_log

    @property
    def cv_scores(self):
        """Return dummy CV scores for demo"""
        return {
            'Ridge': 0.11037,
            'Lasso': 0.10894,
            'ElasticNet': 0.10908,
            'GradientBoosting': 0.11478,
            'Stacking Ensemble': 0.10652
        }
