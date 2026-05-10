#!/usr/bin/env python3
"""
ROGII Wellbore Geology Prediction - Complete Pipeline
GPU-accelerated end-to-end pipeline from data to submission
"""

import time
import pandas as pd
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from src.data import load_all_data
from src.preprocess import preprocess_data, add_features
from src.model import train_model, save_model
from src.predict import generate_predictions

def main():
    start_time = time.time()
    print("🚀 Starting ROGII Pipeline with GPU acceleration")
    print("=" * 60)

    # Configuration
    data_dir = Path('.')
    model_type = 'lgb'  # LightGBM with GPU for fastest training
    model_path = data_dir / "models" / f"{model_type}_gpu_model.pkl"
    scaler_path = data_dir / "models" / "scaler.pkl"
    submission_path = data_dir / "submission.csv"

    # Create models directory
    model_path.parent.mkdir(exist_ok=True)

    print("📊 Step 1: Loading training data...")
    step_start = time.time()
    train_df = load_all_data(str(data_dir), 'train')
    print(f"   Shape: {train_df.shape}")
    print(f"   Wells: {train_df['well_id'].nunique()}")
    print(f"   Time: {time.time() - step_start:.2f}s")

    print("\n🔧 Step 2: Preprocessing and feature engineering...")
    step_start = time.time()

    # Preprocess
    train_processed, scaler, imputer = preprocess_data(train_df, is_train=True)
    print(f"   Preprocessing time: {time.time() - step_start:.2f}s")

    # Add features
    train_featured = add_features(train_processed)
    print(f"   Feature engineering time: {time.time() - step_start:.2f}s")
    print(f"   Features: {len(train_featured.columns)}")

    # Save scaler
    joblib.dump(scaler, scaler_path)
    print(f"   Scaler saved to: {scaler_path}")

    print("\n🤖 Step 3: Training model with GPU acceleration...")
    step_start = time.time()

    # Prepare features and target
    exclude_cols = ['TVT', 'well_id', 'TVT_input', 'Geology', 'GR_type']
    feature_cols = [col for col in train_featured.columns if col not in exclude_cols]
    X = train_featured[feature_cols]
    y = train_featured['TVT']
    groups = train_featured['well_id']

    print(f"   Training with {len(feature_cols)} features on {len(X)} samples")
    print(f"   Using {model_type.upper()} with GPU acceleration")

    # Train model
    model, cv_scores = train_model(X, y, groups, model_type=model_type)
    print(f"   Training time: {time.time() - step_start:.2f}s")
    print(f"   CV RMSE: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Save model
    save_model(model, str(model_path))
    print(f"   Model saved to: {model_path}")

    print("\n🎯 Step 4: Generating predictions...")
    step_start = time.time()

    # Load scaler for predictions
    scaler = joblib.load(scaler_path)

    # Generate predictions
    submission = generate_predictions(str(data_dir), str(model_path), str(scaler_path))
    submission.to_csv(submission_path, index=False)

    print(f"   Prediction time: {time.time() - step_start:.2f}s")
    print(f"   Submission shape: {submission.shape}")
    print(f"   Sample predictions: {submission['tvt'].head().values}")
    print(f"   Saved to: {submission_path}")

    # Final timing
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("✅ Pipeline completed successfully!")
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"📁 Submission file: {submission_path}")
    print("\n🚀 Ready for Kaggle submission!")

if __name__ == "__main__":
    main()