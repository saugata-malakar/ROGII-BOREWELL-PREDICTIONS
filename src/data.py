import pandas as pd
import os
from pathlib import Path

def load_well_data(data_dir: str, well_id: str, data_type: str = 'train') -> pd.DataFrame:
    """
    Load a single well's data from horizontal_well.csv and typewell.csv.

    Args:
        data_dir: Base data directory
        well_id: Well identifier (e.g., '000d7d20')
        data_type: 'train' or 'test'

    Returns:
        Merged DataFrame with horizontal and typewell data
    """
    base_path = Path(data_dir) / data_type

    # Load horizontal well data
    horiz_path = base_path / f"{well_id}__horizontal_well.csv"
    horiz_df = pd.read_csv(horiz_path)

    # Load typewell data
    type_path = base_path / f"{well_id}__typewell.csv"
    type_df = pd.read_csv(type_path)

    # Merge on TVT if it exists in both dataframes
    if 'TVT' in horiz_df.columns and 'TVT' in type_df.columns:
        merged_df = pd.merge(horiz_df, type_df, on='TVT', how='left', suffixes=('_horiz', '_type'))
    else:
        # For test data where TVT might not be in horizontal_well, use index-based merge
        # Assume both files have same number of rows in same order
        merged_df = pd.concat([horiz_df, type_df], axis=1)

    # Add well_id column
    merged_df['well_id'] = well_id

    return merged_df

def load_all_data(data_dir: str, data_type: str = 'train') -> pd.DataFrame:
    """
    Load all wells' data for train or test.

    Args:
        data_dir: Base data directory
        data_type: 'train' or 'test'

    Returns:
        Combined DataFrame from all wells
    """
    base_path = Path(data_dir) / data_type
    all_dfs = []

    # Get all well IDs from horizontal files
    horiz_files = list(base_path.glob("*__horizontal_well.csv"))
    well_ids = [f.stem.split('__')[0] for f in horiz_files]

    for well_id in well_ids:
        df = load_well_data(data_dir, well_id, data_type)
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Create unique ID for submission (test data)
    if data_type == 'test':
        combined_df['id'] = combined_df['well_id'] + '_' + combined_df.groupby('well_id').cumcount().astype(str)

    return combined_df

if __name__ == "__main__":
    # Example usage
    data_dir = Path(__file__).parent.parent / "data"  # Adjust if needed
    train_df = load_all_data(str(data_dir), 'train')
    print(f"Train data shape: {train_df.shape}")
    print(train_df.head())