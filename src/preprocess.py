import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path

def preprocess_data(df: pd.DataFrame, is_train: bool = True, scaler=None, imputer=None) -> pd.DataFrame:
    """
    Preprocess the wellbore data.

    Args:
        df: Raw DataFrame
        is_train: Whether this is training data (for fitting scalers)
        scaler: Pre-fitted scaler (used when is_train=False)
        imputer: Pre-fitted imputer (used when is_train=False)

    Returns:
        Preprocessed DataFrame (and scaler, imputer if is_train=True)
    """
    df = df.copy()

    # Drop training-only columns to avoid feature mismatch in test set
    train_only_cols = ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']
    cols_to_drop = [col for col in train_only_cols if col in df.columns]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)

    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if is_train:
        imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    else:
        if imputer is None:
            # Try to load from disk
            imputer_path = Path('.') / "models" / "imputer.pkl"
            imputer = joblib.load(imputer_path)
        
        # Ensure exact column match for imputer
        imputer_cols = imputer.get_feature_names_out().tolist() if hasattr(imputer, 'get_feature_names_out') else getattr(imputer, 'feature_names_in_', numeric_cols)
        
        # Add any missing columns as NaN
        for col in imputer_cols:
            if col not in df.columns:
                df[col] = np.nan
                
        df[imputer_cols] = imputer.transform(df[imputer_cols])
        
        # For any numeric columns test has that training didn't: fill with median
        for col in numeric_cols:
            if col not in imputer_cols:
                if df[col].isna().any().any() if hasattr(df[col].isna().any(), 'any') else df[col].isna().any():
                    df[col] = df[col].fillna(df[col].median())

    # Encode categorical columns if any (Geology might be categorical)
    if 'Geology' in df.columns:
        le = LabelEncoder()
        if is_train:
            df['Geology'] = le.fit_transform(df['Geology'].astype(str))
            # Save the label encoder
            le_path = Path('.') / "models" / "label_encoder.pkl"
            joblib.dump(le, le_path)
        else:
            # Load saved label encoder
            try:
                le_path = Path('.') / "models" / "label_encoder.pkl"
                le = joblib.load(le_path)
                df['Geology'] = le.transform(df['Geology'].astype(str))
            except:
                # Fallback if encoder not found or has unseen values
                pass

    # Normalize numeric features (exclude target and IDs)
    exclude_cols = ['TVT', 'well_id', 'id'] if 'id' in df.columns else ['TVT', 'well_id']
    numeric_features = [col for col in numeric_cols if col not in exclude_cols]

    if is_train:
        scaler = StandardScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])
        # Save imputer
        imputer_path = Path('.') / "models" / "imputer.pkl"
        joblib.dump(imputer, imputer_path)
        return df, scaler, imputer
    else:
        # Use provided scaler
        if scaler is not None:
            # Get columns scaler was fit on
            scaler_cols = getattr(scaler, 'feature_names_in_', numeric_features).tolist() if hasattr(scaler, 'feature_names_in_') else numeric_features
            
            # Ensure exact column match for scaler
            for col in scaler_cols:
                if col not in df.columns:
                    df[col] = np.nan
                    
            df[scaler_cols] = scaler.transform(df[scaler_cols])
        return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to the DataFrame.

    Args:
        df: Preprocessed DataFrame

    Returns:
        DataFrame with additional features
    """
    df = df.copy()

    # Trajectory features (if columns exist)
    if 'Z' in df.columns:
        df['depth_diff'] = df['Z'].diff().fillna(0)
    if all(col in df.columns for col in ['X', 'Y', 'Z']):
        df['distance'] = np.sqrt(df['X'].diff()**2 + df['Y'].diff()**2 + df['Z'].diff()**2).fillna(0)

    # Sensor differences - only for columns that exist
    sensor_cols = ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA', 'GR_horiz']
    sensor_cols = [col for col in sensor_cols if col in df.columns]  # Filter to only existing columns
    
    for col in sensor_cols:
        df[f'{col}_diff'] = df[col].diff().fillna(0)
        df[f'{col}_rolling_mean'] = df[col].rolling(window=5, min_periods=1).mean()
        df[f'{col}_rolling_std'] = df[col].rolling(window=5, min_periods=1).std()

    # Well-specific aggregations (grouped by well_id)
    if 'well_id' in df.columns and len(sensor_cols) > 0:
        for col in sensor_cols:
            df[f'{col}_well_mean'] = df.groupby('well_id')[col].transform('mean')
            df[f'{col}_well_std'] = df.groupby('well_id')[col].transform('std')

    return df

if __name__ == "__main__":
    from data import load_all_data
    from pathlib import Path

    data_dir = Path(__file__).parent.parent
    train_df = load_all_data(str(data_dir), 'train')
    train_processed, scaler, imputer = preprocess_data(train_df, is_train=True)
    train_featured = add_features(train_processed)
    print(f"Processed train data shape: {train_featured.shape}")
    print(train_featured.head())