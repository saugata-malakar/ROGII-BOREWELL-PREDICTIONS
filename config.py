"""
Configuration file for ROGII Wellbore Geology Prediction project.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)

# Data configuration
RANDOM_SEED = 42
N_FOLDS = 5

# Model configuration
MODEL_PARAMS = {
    'lgb': {
        'n_estimators': 1000,
        'learning_rate': 0.1,
        'max_depth': 6,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        'verbose': -1
    },
    'xgb': {
        'n_estimators': 1000,
        'learning_rate': 0.1,
        'max_depth': 6,
        'min_child_weight': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        'tree_method': 'hist'
    },
    'cat': {
        'iterations': 1000,
        'learning_rate': 0.1,
        'depth': 6,
        'l2_leaf_reg': 3,
        'random_state': RANDOM_SEED,
        'verbose': 0,
        'thread_count': -1
    }
}

# Feature engineering configuration
FEATURE_CONFIG = {
    'rolling_windows': [3, 5, 10, 20],
    'sensor_columns': ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA', 'GR'],
    'trajectory_columns': ['X', 'Y', 'Z'],
    'create_differences': True,
    'create_rolling_stats': True,
    'create_well_aggregations': True,
    'create_lag_features': False,  # Can be slow
    'lag_periods': [1, 2, 3, 5]
}

# Preprocessing configuration
PREPROCESSING_CONFIG = {
    'imputation_strategy': 'median',
    'scaling_method': 'standard',  # 'standard', 'minmax', 'robust'
    'handle_outliers': False,
    'outlier_threshold': 3.0  # z-score threshold
}

# Cross-validation configuration
CV_CONFIG = {
    'method': 'groupkfold',  # 'groupkfold', 'kfold', 'stratifiedkfold'
    'n_splits': N_FOLDS,
    'shuffle': False,
    'random_state': RANDOM_SEED
}

# Submission configuration
SUBMISSION_CONFIG = {
    'filename': 'submission.csv',
    'id_column': 'id',
    'target_column': 'tvt'
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': PROJECT_ROOT / 'training.log'
}

# Kaggle constraints
KAGGLE_CONSTRAINTS = {
    'max_runtime_hours': 9,
    'internet_disabled': True,
    'submission_filename': 'submission.csv'
}

# Experiment tracking
EXPERIMENT_CONFIG = {
    'track_experiments': False,  # Set to True to enable MLflow
    'experiment_name': 'rogii-wellbore-prediction',
    'mlflow_uri': './mlruns'
}

if __name__ == "__main__":
    print("Configuration loaded successfully!")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Number of folds: {N_FOLDS}")
