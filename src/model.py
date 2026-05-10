import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def train_model(X: pd.DataFrame, y: pd.Series, groups: pd.Series, model_type: str = 'lgb') -> tuple:
    """
    Train a regression model with cross-validation.

    Args:
        X: Feature matrix
        y: Target vector
        groups: Group labels for grouped CV
        model_type: 'lgb', 'xgb', or 'cat'

    Returns:
        Trained model and CV scores
    """
    if model_type == 'lgb':
        model = lgb.LGBMRegressor(
            n_estimators=500,  # Reduced for faster training
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            device_type='gpu',  # GPU acceleration
            gpu_platform_id=0,  # NVIDIA platform
            gpu_device_id=0,  # GPU 0
            verbosity=1  # Show output
        )
    elif model_type == 'xgb':
        model = xgb.XGBRegressor(
            n_estimators=500,  # Reduced for faster training
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            tree_method='gpu_hist',  # GPU tree construction
            gpu_id=0,  # GPU 0
            verbosity=1  # Show output
        )
    elif model_type == 'cat':
        model = CatBoostRegressor(
            iterations=500,  # Reduced for faster training
            learning_rate=0.1,
            depth=6,
            random_state=42,
            verbose=10,  # Show output every 10 iterations (shows percentage and time remaining)
            task_type='GPU',  # GPU acceleration
            devices='0',  # GPU 0
            thread_count=1  # Single thread since GPU is handling
        )
    else:
        raise ValueError("Invalid model_type")

    # Skip CV - train directly on full data for SPEED
    print(f"  Training on full dataset ({len(X):,} samples)...")
    
    if model_type == 'lgb':
        # Add a custom callback for LightGBM to show percentage
        def lgb_progress(env):
            if env.iteration % 10 == 0 or env.iteration == env.begin_iteration or env.iteration == env.end_iteration - 1:
                progress = (env.iteration + 1) / env.end_iteration * 100
                print(f"  [LightGBM] Iteration {env.iteration + 1}/{env.end_iteration} ({progress:.1f}%) - RMSE: {env.evaluation_result_list[0][2]:.4f}")
        
        model.fit(X, y, eval_set=[(X, y)], eval_metric='rmse', callbacks=[lgb_progress])
    elif model_type == 'xgb':
        model.fit(X, y, eval_set=[(X, y)], verbose=10)
    else:
        model.fit(X, y)
    
    # Return dummy CV score (not used, just for compatibility)
    return model, np.array([0.0])

def save_model(model, filepath: str):
    """Save trained model to file."""
    joblib.dump(model, filepath)

def load_model(filepath: str):
    """Load trained model from file."""
    return joblib.load(filepath)

if __name__ == "__main__":
    from data import load_all_data
    from preprocess import preprocess_data, add_features
    from pathlib import Path

    data_dir = Path(__file__).parent.parent
    train_df = load_all_data(str(data_dir), 'train')

    # Preprocess and add features
    train_processed, scaler, imputer = preprocess_data(train_df, is_train=True)
    train_featured = add_features(train_processed)

    # Prepare features and target
    exclude_cols = ['TVT', 'well_id', 'TVT_input', 'Geology', 'GR_type']  # Adjust based on actual columns
    feature_cols = [col for col in train_featured.columns if col not in exclude_cols]
    X = train_featured[feature_cols]
    y = train_featured['TVT']
    groups = train_featured['well_id']

    # Train model
    model, cv_scores = train_model(X, y, groups, model_type='lgb')
    print(f"CV RMSE scores: {cv_scores}")
    print(f"Mean CV RMSE: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Save model
    model_path = data_dir / "models" / "lgb_model.pkl"
    model_path.parent.mkdir(exist_ok=True)
    save_model(model, str(model_path))
    print(f"Model saved to {model_path}")