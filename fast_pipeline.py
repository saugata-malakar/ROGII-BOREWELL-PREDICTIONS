#!/usr/bin/env python3
"""
FAST Complete Pipeline - Retrain model and generate predictions
Only uses columns that exist in BOTH train and test
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("⚡ ROGII - FAST RETRAIN + PREDICT")
print("=" * 70)

start_total = time.time()

try:
    from src.data import load_all_data
    from src.preprocess import preprocess_data, add_features
    from src.model import train_model, save_model, load_model
    print("\n✓ Modules loaded")
except Exception as e:
    print(f"\n✗ Error: {e}")
    exit(1)

data_dir = Path('.')
model_path = data_dir / "models" / "lgb_model_v2.pkl"
scaler_path = data_dir / "models" / "scaler_v2.pkl"
imputer_path = data_dir / "models" / "imputer_v2.pkl"
submission_path = data_dir / "submission.csv"
data_dir.mkdir(exist_ok=True)
model_path.parent.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: Load & Prepare Train Data
# ============================================================================
print("\n[1/6] Loading training data...")
step_start = time.time()
try:
    train_df = load_all_data(str(data_dir), 'train')
    train_df = train_df.drop(columns=['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA'], errors='ignore')
    print(f"  ✓ Shape: {train_df.shape}")
    print(f"  ✓ Time: {time.time() - step_start:.2f}s")
except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)

# ============================================================================
# STEP 2: Feature Engineering
# ============================================================================
print("\n[2/6] Engineering features...")
step_start = time.time()
try:
    train_featured = add_features(train_df)
    print(f"  ✓ Shape: {train_featured.shape}")
    print(f"  ✓ Time: {time.time() - step_start:.2f}s")
except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)

# ============================================================================
# STEP 3: Preprocessing
# ============================================================================
print("\n[3/6] Preprocessing...")
step_start = time.time()
try:
    train_processed, scaler, imputer = preprocess_data(train_featured, is_train=True)
    joblib.dump(scaler, scaler_path)
    joblib.dump(imputer, imputer_path)
    print(f"  ✓ Time: {time.time() - step_start:.2f}s")
except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)

# ============================================================================
# STEP 4: Prepare Features & Train
# ============================================================================
print("\n[4/6] Training model...")
step_start = time.time()
try:
    exclude_cols = ['TVT', 'well_id', 'TVT_input', 'Geology', 'GR_type']
    feature_cols = [col for col in train_processed.columns if col not in exclude_cols]
    X = train_processed[feature_cols]
    y = train_processed['TVT']
    groups = train_processed['well_id']
    
    model, _ = train_model(X, y, groups, model_type='lgb')
    save_model(model, str(model_path))
    print(f"  ✓ Features: {len(feature_cols)}")
    print(f"  ✓ Time: {time.time() - step_start:.2f}s")
except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)

# ============================================================================
# STEP 5: Load Test Data & Predict
# ============================================================================
print("\n[5/6] Predicting on test data...")
step_start = time.time()
try:
    test_df = load_all_data(str(data_dir), 'test')
    test_df = test_df.drop(columns=['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA'], errors='ignore')
    
    test_featured = add_features(test_df)
    test_processed = preprocess_data(test_featured, is_train=False, scaler=scaler, imputer=imputer)
    
    # Use same feature columns as training
    X_test = test_processed[feature_cols]
    model = load_model(str(model_path))
    predictions = model.predict(X_test)
    
    print(f"  ✓ Predictions: {len(predictions)}")
    print(f"  ✓ Range: {predictions.min():.2f} - {predictions.max():.2f}")
    print(f"  ✓ Time: {time.time() - step_start:.2f}s")
except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)

# ============================================================================
# STEP 6: Create Submission
# ============================================================================
print("\n[6/6] Creating submission...")
step_start = time.time()
try:
    submission = pd.DataFrame({
        'id': test_featured['id'],
        'tvt': predictions
    })
    submission.to_csv(submission_path, index=False)
    print(f"  ✓ Saved: {submission_path}")
    print(f"  ✓ Rows: {len(submission)}")
    print(f"  ✓ Sample:\n{submission.head()}")
    print(f"  ✓ Time: {time.time() - step_start:.2f}s")
except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)

total_time = time.time() - start_total
print("\n" + "=" * 70)
print(f"✅ COMPLETE in {total_time:.2f}s")
print("=" * 70)
print(f"\nNext: kaggle competitions submit -c rogii-wellbore-geology-prediction -f submission.csv -m 'Real predictions'")
