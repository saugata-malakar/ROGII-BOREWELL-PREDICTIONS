#!/usr/bin/env python3
"""
FAST Kaggle Submission Generator
Uses pre-trained model to generate predictions only (no retraining)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("⚡ FAST PREDICTION MODE - Generating Submission")
print("=" * 70)

# Load modules
from src.data import load_all_data
from src.preprocess import preprocess_data, add_features
from src.model import load_model

# Paths
data_dir = Path('.')
model_path = data_dir / "models" / "lgb_model.pkl"
scaler_path = data_dir / "models" / "scaler.pkl"
submission_path = data_dir / "submission.csv"

print("\n[1/5] Loading test data...")
try:
    test_df = load_all_data(str(data_dir), 'test')
    print(f"  ✓ Test data loaded: {test_df.shape}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)

print("\n[2/5] Engineering features...")
try:
    test_featured = add_features(test_df)
    print(f"  ✓ Features created: {test_featured.shape}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)

print("\n[3/5] Loading scaler & preprocessing...")
try:
    scaler = joblib.load(scaler_path)
    test_processed = preprocess_data(test_featured, is_train=False, scaler=scaler)
    print(f"  ✓ Preprocessed shape: {test_processed.shape}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)

print("\n[4/5] Loading model & generating predictions...")
try:
    model = load_model(model_path)
    
    # Get feature columns
    exclude_cols = ['TVT', 'well_id', 'TVT_input', 'Geology', 'GR_type', 'id']
    feature_cols = [col for col in test_processed.columns if col not in exclude_cols]
    
    X_test = test_processed[feature_cols]
    predictions = model.predict(X_test)
    
    print(f"  ✓ Predictions generated: {len(predictions)}")
    print(f"  ✓ Min: {predictions.min():.4f}, Max: {predictions.max():.4f}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)

print("\n[5/5] Creating submission file...")
try:
    submission = pd.DataFrame({
        'id': test_featured['id'],
        'tvt': predictions
    })
    
    submission.to_csv(submission_path, index=False)
    print(f"  ✓ Submission saved: {submission_path}")
    print(f"  ✓ Rows: {len(submission)}")
    print(f"  ✓ Sample:\n{submission.head()}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)

print("\n" + "=" * 70)
print("✅ DONE! Ready to submit to Kaggle")
print("=" * 70)
print(f"\nNext command:\nkaggle competitions submit -c rogii-wellbore-geology-prediction -f submission.csv -m 'Submission'")
