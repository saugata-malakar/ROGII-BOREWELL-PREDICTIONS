#!/usr/bin/env python3
import pandas as pd
from src.data import load_all_data
from src.preprocess import preprocess_data, add_features
from src.model import load_model
import joblib
from pathlib import Path

# Load test data
print("Loading test data...")
test_df = load_all_data('.', 'test')
print(f"Test shape: {test_df.shape}")
print(f"Test columns: {test_df.columns.tolist()}")
print(f"Sample IDs: {test_df['id'].head().tolist()}")

# Load model to check its feature names
print("\nLoading model...")
model = load_model('models/lgb_model.pkl')
print(f"Model type: {type(model)}")
if hasattr(model, 'feature_names_in_'):
    print(f"Model feature names: {model.feature_names_in_[:10]}...")  # First 10
    print(f"Total model features: {len(model.feature_names_in_)}")

# Feature engineer test data
print("\nFeature engineering...")
test_featured = add_features(test_df)
print(f"After features: {test_featured.shape}")
print(f"Columns: {test_featured.columns.tolist()}")

# Get training feature names from model
if hasattr(model, 'feature_names_in_'):
    expected_features = model.feature_names_in_.tolist()
    print(f"\nExpected {len(expected_features)} features for model")
    
    # Check which features are missing
    missing = set(expected_features) - set(test_featured.columns)
    print(f"Missing features: {list(missing)[:10]}...")  # Show first 10
