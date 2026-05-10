#!/usr/bin/env python3
"""
ROGII Complete Pipeline - GPU Optimized with Continuous Logging
Generates Kaggle submission in minimal time
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import all modules at once to catch errors early
print("=" * 70)
print("🚀 ROGII Wellbore Geology Prediction - Complete Pipeline")
print("=" * 70)

print("\n[01/10] Loading dependencies...")
start_total = time.time()

try:
    from src.data import load_all_data
    print("  ✓ Data module loaded")
except Exception as e:
    print(f"  ✗ Error loading data module: {e}")
    exit(1)

try:
    from src.preprocess import preprocess_data, add_features
    print("  ✓ Preprocessing module loaded")
except Exception as e:
    print(f"  ✗ Error loading preprocessing module: {e}")
    exit(1)

try:
    from src.model import train_model, save_model, load_model
    print("  ✓ Model module loaded")
except Exception as e:
    print(f"  ✗ Error loading model module: {e}")
    exit(1)

# Configuration
data_dir = Path('.')
model_type = 'lgb'
model_path = data_dir / "models" / f"{model_type}_model.pkl"
scaler_path = data_dir / "models" / "scaler.pkl"
submission_path = data_dir / "submission.csv"
model_path.parent.mkdir(exist_ok=True)

print(f"  ✓ Configuration ready: model_type={model_type}")

# ============================================================================
# STEP 1: Load Training Data
# ============================================================================
print("\n[02/10] Loading training data...")
step_start = time.time()

try:
    train_df = load_all_data(str(data_dir), 'train')
    print(f"  ✓ Loaded shape: {train_df.shape}")
    print(f"  ✓ Wells: {train_df['well_id'].nunique()}")
    elapsed = time.time() - step_start
    print(f"  ✓ Time: {elapsed:.2f}s")
    print(f"  ⏳ Progress: 20% complete")
except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)

# ============================================================================
# STEP 2: Feature Engineering (BEFORE preprocessing - so imputer sees engineered columns)
# ============================================================================
print("\n[03/10] Engineering features...")
step_start = time.time()

try:
    train_featured = add_features(train_df)
    print(f"  ✓ Features created: {len(train_featured.columns)}")
    print(f"  ✓ Shape: {train_featured.shape}")
    elapsed = time.time() - step_start
    print(f"  ✓ Time: {elapsed:.2f}s")
    print(f"  ⏳ Progress: 30% complete")
except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)

# ============================================================================
# STEP 3: Preprocessing (AFTER feature engineering)
# ============================================================================
print("\n[04/10] Preprocessing data...")
step_start = time.time()

try:
    train_processed, scaler, imputer = preprocess_data(train_featured, is_train=True)
    print(f"  ✓ Missing values handled")
    elapsed = time.time() - step_start
    print(f"  ✓ Time: {elapsed:.2f}s")
except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)

# ============================================================================
# STEP 4: Save Scaler
# ============================================================================
print("\n[05/10] Saving scaler...")
step_start = time.time()

try:
    joblib.dump(scaler, scaler_path)
    print(f"  ✓ Scaler saved: {scaler_path}")
    print(f"  ✓ Time: {time.time() - step_start:.2f}s")
    print(f"  ⏳ Progress: 40% complete")
except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)

# ============================================================================
# STEP 5: Prepare Training Features
# ============================================================================
print("\n[06/10] Preparing features for training...")
step_start = time.time()

try:
    exclude_cols = ['TVT', 'well_id', 'TVT_input', 'Geology', 'GR_type']
    feature_cols = [col for col in train_processed.columns if col not in exclude_cols]
    X = train_featured[feature_cols]
    y = train_featured['TVT']
    groups = train_featured['well_id']
    
    print(f"  ✓ Features: {len(feature_cols)}")
    print(f"  ✓ Training samples: {len(X)}")
    elapsed = time.time() - step_start
    print(f"  ✓ Time: {elapsed:.2f}s")
    print(f"  ⏳ Progress: 50% complete")
except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)

# ============================================================================
# STEP 6: Train Model (FAST - No CV)
# ============================================================================
print(f"\n[07/10] Training {model_type.upper()} model (full data, GPU)...")
step_start = time.time()

try:
    model, cv_scores = train_model(X, y, groups, model_type=model_type)
    train_time = time.time() - step_start
    print(f"  ✓ Model trained successfully")
    print(f"  ✓ Time: {train_time:.2f}s")
    print(f"  ⏳ Progress: 65% complete")
except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)

# ============================================================================
# STEP 7: Save Model
# ============================================================================
print("\n[08/10] Saving trained model...")
step_start = time.time()

try:
    save_model(model, str(model_path))
    print(f"  ✓ Model saved: {model_path}")
    elapsed = time.time() - step_start
    print(f"  ✓ Time: {elapsed:.2f}s")
    print(f"  ⏳ Progress: 75% complete")
except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)

# ============================================================================
# STEP 8: Load Test Data
# ============================================================================
print("\n[09/10] Loading test data...")
step_start = time.time()

try:
    test_df = load_all_data(str(data_dir), 'test')
    print(f"  ✓ Test data shape: {test_df.shape}")
    elapsed = time.time() - step_start
    print(f"  ✓ Time: {elapsed:.2f}s")
    print(f"  ⏳ Progress: 85% complete")
except Exception as e:
    print(f"  ✗ Error: {e}")
    exit(1)

# ============================================================================
# STEP 9: Generate Predictions
# ============================================================================
print("\n[10/10] Generating predictions...")
step_start = time.time()

try:
    # ADD FEATURES FIRST (before preprocessing - this creates engineered columns)
    test_featured = add_features(test_df)
    print("  ✓ Features engineered")
    
    # THEN Preprocess test data (now it has same columns as training)
    test_processed = preprocess_data(test_featured, is_train=False, scaler=scaler, imputer=imputer)
    print("  ✓ Preprocessing complete")
    
    # Prepare features
    exclude_cols = ['well_id', 'Geology', 'GR_type', 'id']
    feature_cols_test = [col for col in test_processed.columns if col not in exclude_cols and col != 'id']
    X_test = test_processed[feature_cols_test]
    
    # Generate predictions
    predictions = model.predict(X_test)
    print("  ✓ Predictions generated")
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_featured['id'],
        'tvt': predictions
    })
    
    # Save submission
    submission.to_csv(submission_path, index=False)
    elapsed = time.time() - step_start
    print(f"  ✓ Submission saved: {submission_path}")
    print(f"  ✓ Shape: {submission.shape}")
    print(f"  ✓ Time: {elapsed:.2f}s")
    print(f"  ⏳ Progress: 95% complete")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# Final Summary
# ============================================================================
total_time = time.time() - start_total

print("\n" + "=" * 70)
print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 70)
print(f"⏱️  Total Time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
print(f"📊 Samples processed: {len(submission)}")
print(f"📁 Submission file: {submission_path}")
print(f"📈 Sample predictions:")
print(submission.head(10).to_string(index=False))
print("\n🚀 Ready for Kaggle submission!")
print("=" * 70)
