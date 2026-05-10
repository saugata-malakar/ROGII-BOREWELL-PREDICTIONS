"""
Utility functions for the ROGII Wellbore Geology Prediction project.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from typing import Dict, List, Tuple, Any
import json


def set_random_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # Set seeds for ML libraries
    try:
        import lightgbm as lgb
        lgb.LGBMRegressor(random_state=seed)
    except ImportError:
        pass
    
    try:
        import xgboost as xgb
        xgb.XGBRegressor(random_state=seed)
    except ImportError:
        pass


def validate_submission(submission_path: str, sample_path: str = None) -> bool:
    """
    Validate submission file format.
    
    Args:
        submission_path: Path to submission CSV
        sample_path: Path to sample submission (optional)
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    df = pd.read_csv(submission_path)
    
    # Check columns
    if list(df.columns) != ['id', 'tvt']:
        raise ValueError(f"Invalid columns. Expected ['id', 'tvt'], got {list(df.columns)}")
    
    # Check for missing values
    if df.isnull().any().any():
        raise ValueError("Submission contains missing values")
    
    # Check for duplicate IDs
    if df['id'].duplicated().any():
        raise ValueError("Submission contains duplicate IDs")
    
    # Check data types
    if not pd.api.types.is_numeric_dtype(df['tvt']):
        raise ValueError("Target 'tvt' must be numeric")
    
    # Compare with sample if provided
    if sample_path:
        sample_df = pd.read_csv(sample_path)
        if len(df) != len(sample_df):
            raise ValueError(f"Row count mismatch. Expected {len(sample_df)}, got {len(df)}")
        
        # Check if IDs match
        if not df['id'].equals(sample_df['id']):
            missing_ids = set(sample_df['id']) - set(df['id'])
            extra_ids = set(df['id']) - set(sample_df['id'])
            if missing_ids:
                raise ValueError(f"Missing IDs: {list(missing_ids)[:5]}...")
            if extra_ids:
                raise ValueError(f"Extra IDs: {list(extra_ids)[:5]}...")
    
    print(f"✓ Submission validation passed!")
    print(f"  - Shape: {df.shape}")
    print(f"  - ID range: {df['id'].iloc[0]} to {df['id'].iloc[-1]}")
    print(f"  - TVT range: [{df['tvt'].min():.4f}, {df['tvt'].max():.4f}]")
    print(f"  - TVT mean: {df['tvt'].mean():.4f}")
    
    return True


def save_experiment_results(results: Dict[str, Any], filepath: str):
    """
    Save experiment results to JSON file.
    
    Args:
        results: Dictionary of experiment results
        filepath: Path to save JSON file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    results_converted = convert_types(results)
    
    with open(filepath, 'w') as f:
        json.dump(results_converted, f, indent=2)
    
    print(f"Experiment results saved to {filepath}")


def load_experiment_results(filepath: str) -> Dict[str, Any]:
    """
    Load experiment results from JSON file.
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        Dictionary of experiment results
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    return results


def calculate_feature_importance(model, feature_names: List[str], top_n: int = 20) -> pd.DataFrame:
    """
    Calculate and return feature importance.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to return
    
    Returns:
        DataFrame with feature names and importance scores
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    
    importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
    importance_df['importance_pct'] = 100 * importance_df['importance'] / importance_df['importance'].sum()
    
    return importance_df.reset_index(drop=True)


def memory_usage_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate memory usage report for DataFrame.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with memory usage by column
    """
    mem_usage = df.memory_usage(deep=True)
    mem_df = pd.DataFrame({
        'column': mem_usage.index,
        'memory_mb': mem_usage.values / 1024**2,
        'dtype': [df[col].dtype if col != 'Index' else 'Index' for col in mem_usage.index]
    })
    
    mem_df = mem_df.sort_values('memory_mb', ascending=False)
    total_mb = mem_df['memory_mb'].sum()
    
    print(f"Total memory usage: {total_mb:.2f} MB")
    return mem_df


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame dtypes to reduce memory usage.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with optimized dtypes
    """
    df = df.copy()
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Optimize object columns (convert to category if low cardinality)
    for col in df.select_dtypes(include=['object']).columns:
        num_unique = df[col].nunique()
        num_total = len(df[col])
        if num_unique / num_total < 0.5:  # Less than 50% unique values
            df[col] = df[col].astype('category')
    
    return df


def create_submission_template(test_ids: List[str], output_path: str):
    """
    Create submission template with zeros.
    
    Args:
        test_ids: List of test IDs
        output_path: Path to save submission CSV
    """
    submission = pd.DataFrame({
        'id': test_ids,
        'tvt': 0.0
    })
    
    submission.to_csv(output_path, index=False)
    print(f"Submission template created: {output_path}")
    print(f"Shape: {submission.shape}")


def compare_submissions(sub1_path: str, sub2_path: str):
    """
    Compare two submission files.
    
    Args:
        sub1_path: Path to first submission
        sub2_path: Path to second submission
    """
    sub1 = pd.read_csv(sub1_path)
    sub2 = pd.read_csv(sub2_path)
    
    # Check if IDs match
    if not sub1['id'].equals(sub2['id']):
        print("⚠ Warning: IDs do not match between submissions")
        return
    
    # Calculate differences
    diff = sub1['tvt'] - sub2['tvt']
    
    print(f"Submission Comparison:")
    print(f"  Mean difference: {diff.mean():.6f}")
    print(f"  Std difference: {diff.std():.6f}")
    print(f"  Max absolute difference: {diff.abs().max():.6f}")
    print(f"  Correlation: {sub1['tvt'].corr(sub2['tvt']):.6f}")
    
    # Show largest differences
    diff_df = pd.DataFrame({
        'id': sub1['id'],
        'sub1_tvt': sub1['tvt'],
        'sub2_tvt': sub2['tvt'],
        'difference': diff,
        'abs_difference': diff.abs()
    })
    
    print(f"\nTop 10 largest differences:")
    print(diff_df.nlargest(10, 'abs_difference')[['id', 'sub1_tvt', 'sub2_tvt', 'difference']])


if __name__ == "__main__":
    # Example usage
    print("Utility functions loaded successfully!")
    
    # Test random seed
    set_random_seed(42)
    print(f"Random number: {np.random.rand()}")
    
    # Test validation (if submission exists)
    try:
        validate_submission("submission.csv", "sample_submission.csv")
    except FileNotFoundError:
        print("Submission files not found (this is expected if not yet generated)")
