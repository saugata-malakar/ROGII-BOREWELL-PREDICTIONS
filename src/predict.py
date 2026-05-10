import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from .data import load_all_data
from .preprocess import preprocess_data, add_features
from .model import load_model

def generate_predictions(data_dir: str, model_path: str, scaler_path: str) -> pd.DataFrame:
    """
    Generate predictions for test data.

    Args:
        data_dir: Base data directory
        model_path: Path to trained model
        scaler_path: Path to fitted scaler

    Returns:
        DataFrame with id and tvt predictions
    """
    # Load test data
    test_df = load_all_data(data_dir, 'test')

    # Load scaler
    scaler = joblib.load(scaler_path)

    # Preprocess (use training scaler)
    test_processed = preprocess_data(test_df, is_train=False, scaler=scaler)
    test_featured = add_features(test_processed)

    # Prepare features (ensure exact match with training)
    exclude_cols = ['TVT', 'well_id', 'TVT_input', 'Geology', 'GR_type', 'id']
    feature_cols = [col for col in test_featured.columns if col not in exclude_cols]
    
    # Check if the model has feature names attribute and use it to select exact columns
    model = load_model(model_path)
    if hasattr(model, 'feature_name_'):
        feature_cols = model.feature_name_
    elif hasattr(model, 'feature_names_in_'):
        feature_cols = model.feature_names_in_
        
    X_test = test_featured[feature_cols]

    # Load model and predict
    model = load_model(model_path)
    predictions = model.predict(X_test)

    # Create submission DataFrame
    submission = pd.DataFrame({
        'id': test_featured['id'],
        'tvt': predictions
    })

    return submission

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent
    model_path = data_dir / "models" / "lgb_model.pkl"
    scaler_path = data_dir / "models" / "scaler.pkl"  # Assume scaler is saved

    # Load scaler (need to implement saving in preprocess)
    import joblib
    scaler = joblib.load(scaler_path)

    submission = generate_predictions(str(data_dir), str(model_path), scaler)
    submission_path = data_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(submission.head())