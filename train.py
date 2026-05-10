"""
Main training script for ROGII Wellbore Geology Prediction.
Orchestrates the full training pipeline with configuration.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data import load_all_data
from src.preprocess import preprocess_data, add_features
from src.model import train_model, save_model
from src.utils import (
    set_random_seed, 
    save_experiment_results, 
    calculate_feature_importance,
    memory_usage_report
)
from config import (
    DATA_DIR, 
    MODELS_DIR, 
    RANDOM_SEED, 
    MODEL_PARAMS,
    FEATURE_CONFIG,
    CV_CONFIG
)


def main(model_type='lgb', save_results=True):
    """
    Main training pipeline.
    
    Args:
        model_type: Type of model to train ('lgb', 'xgb', 'cat')
        save_results: Whether to save experiment results
    """
    print("=" * 80)
    print("ROGII Wellbore Geology Prediction - Training Pipeline")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model type: {model_type}")
    print(f"Random seed: {RANDOM_SEED}")
    print()
    
    # Set random seed
    set_random_seed(RANDOM_SEED)
    
    # Step 1: Load data
    print("[1/6] Loading training data...")
    train_df = load_all_data(str(DATA_DIR), 'train')
    print(f"  ✓ Loaded {len(train_df)} rows from {train_df['well_id'].nunique()} wells")
    print(f"  ✓ Shape: {train_df.shape}")
    print()
    
    # Memory usage report
    print("Memory usage before preprocessing:")
    mem_report = memory_usage_report(train_df)
    print(f"  Total: {mem_report['memory_mb'].sum():.2f} MB")
    print()
    
    # Step 2: Preprocess data
    print("[2/6] Preprocessing data...")
    train_processed, scaler, imputer = preprocess_data(train_df, is_train=True)
    print(f"  ✓ Preprocessing complete")
    print(f"  ✓ Shape after preprocessing: {train_processed.shape}")
    
    # Save scaler
    scaler_path = MODELS_DIR / "scaler.pkl"
    import joblib
    joblib.dump(scaler, scaler_path)
    print(f"  ✓ Scaler saved to {scaler_path}")
    print()
    
    # Step 3: Feature engineering
    print("[3/6] Engineering features...")
    train_featured = add_features(train_processed)
    print(f"  ✓ Feature engineering complete")
    print(f"  ✓ Shape after feature engineering: {train_featured.shape}")
    print(f"  ✓ Added {train_featured.shape[1] - train_processed.shape[1]} new features")
    print()
    
    # Step 4: Prepare features and target
    print("[4/6] Preparing features and target...")
    
    # Identify target and exclude columns
    target_col = 'TVT'
    exclude_cols = ['TVT', 'well_id', 'id']
    
    # Get feature columns
    feature_cols = [col for col in train_featured.columns if col not in exclude_cols]
    
    # Handle any remaining missing values
    train_featured[feature_cols] = train_featured[feature_cols].fillna(0)
    
    X = train_featured[feature_cols]
    y = train_featured[target_col]
    groups = train_featured['well_id']
    
    print(f"  ✓ Features: {len(feature_cols)} columns")
    print(f"  ✓ Target: {target_col}")
    print(f"  ✓ Groups: {groups.nunique()} unique wells")
    print(f"  ✓ Target range: [{y.min():.4f}, {y.max():.4f}]")
    print(f"  ✓ Target mean: {y.mean():.4f} ± {y.std():.4f}")
    print()
    
    # Step 5: Train model
    print(f"[5/6] Training {model_type.upper()} model with {CV_CONFIG['n_splits']}-fold CV...")
    model, cv_scores = train_model(X, y, groups, model_type=model_type)
    
    print(f"  ✓ Training complete!")
    print(f"  ✓ CV RMSE scores: {[f'{score:.4f}' for score in cv_scores]}")
    print(f"  ✓ Mean CV RMSE: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print()
    
    # Step 6: Save model and results
    print("[6/6] Saving model and results...")
    
    # Save model
    model_path = MODELS_DIR / f"{model_type}_model.pkl"
    save_model(model, str(model_path))
    print(f"  ✓ Model saved to {model_path}")
    
    # Calculate feature importance
    try:
        importance_df = calculate_feature_importance(model, feature_cols, top_n=20)
        print(f"\n  Top 10 Most Important Features:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"    {idx+1}. {row['feature']}: {row['importance_pct']:.2f}%")
        
        # Save feature importance
        importance_path = MODELS_DIR / f"{model_type}_feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        print(f"\n  ✓ Feature importance saved to {importance_path}")
    except Exception as e:
        print(f"  ⚠ Could not calculate feature importance: {e}")
    
    # Save experiment results
    if save_results:
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'random_seed': RANDOM_SEED,
            'n_samples': len(X),
            'n_features': len(feature_cols),
            'n_wells': groups.nunique(),
            'cv_scores': cv_scores.tolist(),
            'mean_cv_rmse': float(cv_scores.mean()),
            'std_cv_rmse': float(cv_scores.std()),
            'target_mean': float(y.mean()),
            'target_std': float(y.std()),
            'model_params': MODEL_PARAMS.get(model_type, {}),
            'feature_config': FEATURE_CONFIG,
            'cv_config': CV_CONFIG
        }
        
        results_path = MODELS_DIR / f"{model_type}_results.json"
        save_experiment_results(results, str(results_path))
        print(f"  ✓ Experiment results saved to {results_path}")
    
    print()
    print("=" * 80)
    print("Training Complete!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Final CV RMSE: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("=" * 80)
    
    return model, cv_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ROGII wellbore prediction model')
    parser.add_argument(
        '--model', 
        type=str, 
        default='lgb', 
        choices=['lgb', 'xgb', 'cat'],
        help='Model type to train (default: lgb)'
    )
    parser.add_argument(
        '--no-save', 
        action='store_true',
        help='Do not save experiment results'
    )
    
    args = parser.parse_args()
    
    try:
        model, cv_scores = main(
            model_type=args.model,
            save_results=not args.no_save
        )
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
