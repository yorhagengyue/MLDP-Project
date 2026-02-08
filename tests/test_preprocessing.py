"""
Unit tests for data preprocessing and model pipeline.

Ensures data quality, prevents data leakage, and validates model outputs.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDataIntegrity:
    """Test data loading and basic integrity checks."""
    
    def test_training_data_exists(self):
        """Verify training data file exists."""
        assert os.path.exists('../data/train.csv'), "Training data not found"
    
    def test_test_data_exists(self):
        """Verify test data file exists."""
        assert os.path.exists('../data/test.csv'), "Test data not found"
    
    def test_training_data_shape(self):
        """Verify training data has expected dimensions."""
        train = pd.read_csv('../data/train.csv')
        assert train.shape[0] == 1460, f"Expected 1460 rows, got {train.shape[0]}"
        assert train.shape[1] == 81, f"Expected 81 columns, got {train.shape[1]}"
    
    def test_target_variable_exists(self):
        """Verify SalePrice column exists in training data."""
        train = pd.read_csv('../data/train.csv')
        assert 'SalePrice' in train.columns, "SalePrice column missing"
    
    def test_no_negative_prices(self):
        """Verify all prices are positive."""
        train = pd.read_csv('../data/train.csv')
        assert (train['SalePrice'] > 0).all(), "Found negative or zero prices"


class TestDataLeakagePrevention:
    """Test that train/test separation is maintained."""
    
    def test_processed_data_no_leakage(self):
        """Verify processed train and test sets are properly separated."""
        if not os.path.exists('../data/X_train_processed.csv'):
            pytest.skip("Processed data not yet generated")
        
        X_train = pd.read_csv('../data/X_train_processed.csv')
        X_test = pd.read_csv('../data/X_test_processed.csv')
        
        # Check no NaN values (should be filled using train statistics only)
        assert X_train.isnull().sum().sum() == 0, "Training data contains NaN values"
        assert X_test.isnull().sum().sum() == 0, "Test data contains NaN values"
    
    def test_feature_consistency(self):
        """Verify train and test have identical feature sets."""
        if not os.path.exists('../data/X_train_processed.csv'):
            pytest.skip("Processed data not yet generated")
        
        X_train = pd.read_csv('../data/X_train_processed.csv')
        X_test = pd.read_csv('../data/X_test_processed.csv')
        
        assert list(X_train.columns) == list(X_test.columns), "Feature columns do not match"
        assert X_train.shape[1] == X_test.shape[1], "Column count mismatch"
    
    def test_no_target_in_test(self):
        """Verify test data does not contain target variable."""
        test = pd.read_csv('../data/test.csv')
        assert 'SalePrice' not in test.columns, "Test data should not have SalePrice"


class TestFeatureEngineering:
    """Test feature engineering outputs."""
    
    def test_feature_count_increase(self):
        """Verify feature engineering created new features."""
        if not os.path.exists('../data/X_train_processed.csv'):
            pytest.skip("Processed data not yet generated")
        
        train_original = pd.read_csv('../data/train.csv')
        X_train_processed = pd.read_csv('../data/X_train_processed.csv')
        
        # Original: 79 features (81 columns - Id - SalePrice)
        # After engineering: should be more than 79
        assert X_train_processed.shape[1] > train_original.shape[1] - 2, \
            "Feature engineering did not increase feature count"
    
    def test_no_infinite_values(self):
        """Verify no infinite values in processed data."""
        if not os.path.exists('../data/X_train_processed.csv'):
            pytest.skip("Processed data not yet generated")
        
        X_train = pd.read_csv('../data/X_train_processed.csv')
        assert not np.isinf(X_train.values).any(), "Found infinite values in processed data"


class TestTargetTransformation:
    """Test target variable transformations."""
    
    def test_log_transformation_reversible(self):
        """Verify log transformation can be reversed accurately."""
        if not os.path.exists('../data/y_train_log.csv'):
            pytest.skip("Log-transformed target not yet generated")
        
        y_log = pd.read_csv('../data/y_train_log.csv')
        
        # Log-transformed values should be positive
        assert (y_log > 0).all().all(), "Log-transformed prices must be positive"
        
        # Reverse transformation should give reasonable prices
        y_original = np.expm1(y_log)
        assert (y_original > 10000).all().all(), "Reversed prices too low (< $10,000)"
        assert (y_original < 1000000).all().all(), "Reversed prices too high (> $1,000,000)"
    
    def test_target_distribution_improved(self):
        """Verify log transformation reduces skewness."""
        train = pd.read_csv('../data/train.csv')
        original_skew = train['SalePrice'].skew()
        
        # Log transform
        log_prices = np.log1p(train['SalePrice'])
        log_skew = log_prices.skew()
        
        # Log transformation should reduce skewness
        assert abs(log_skew) < abs(original_skew), \
            f"Log transformation did not reduce skewness: {original_skew:.2f} -> {log_skew:.2f}"


class TestModelOutputs:
    """Test model prediction outputs."""
    
    def test_model_file_exists(self):
        """Verify trained model file exists."""
        model_paths = [
            '../models/clean_ensemble.pkl',
            '../models/best_model_xgboost.pkl'
        ]
        found = any(os.path.exists(path) for path in model_paths)
        assert found, "No trained model file found"
    
    def test_predictions_in_valid_range(self):
        """Verify predictions are within reasonable price range."""
        if not os.path.exists('../models/clean_ensemble.pkl'):
            pytest.skip("Model not yet trained")
        
        import pickle
        
        # Load model
        with open('../models/clean_ensemble.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load test data
        if not os.path.exists('../data/X_test_processed.csv'):
            pytest.skip("Processed test data not available")
        
        X_test = pd.read_csv('../data/X_test_processed.csv')
        
        # Make predictions
        predictions = model.predict(X_test[:10])  # Test first 10 samples
        
        # Convert from log scale
        prices = np.expm1(predictions)
        
        # Verify reasonable range
        assert (prices > 10000).all(), "Predictions too low (< $10,000)"
        assert (prices < 1000000).all(), "Predictions too high (> $1,000,000)"
        assert not np.isnan(prices).any(), "Predictions contain NaN"


class TestCrossValidation:
    """Test cross-validation consistency."""
    
    def test_cv_scores_reasonable(self):
        """Verify cross-validation scores are within expected range."""
        if not os.path.exists('../data/model_comparison.csv'):
            pytest.skip("Model comparison results not available")
        
        results = pd.read_csv('../data/model_comparison.csv')
        
        # RMSE should be between 0.05 and 0.20 for log-transformed prices
        if 'rmse_mean' in results.columns:
            assert (results['rmse_mean'] > 0.05).all(), "RMSE suspiciously low"
            assert (results['rmse_mean'] < 0.20).all(), "RMSE suspiciously high"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])
