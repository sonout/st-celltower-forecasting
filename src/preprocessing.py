import pandas as pd
import numpy as np
from pathlib import Path
import glob
import time
import yaml
import argparse
import json
from datetime import datetime, timezone
from uuid import uuid4
from typing import Tuple, List, Dict
from .datasets import file_sha256
from .configuration import load_config as read_config, DEFAULT_CONFIG

def load_config(config_path: str) -> dict:
    """Loads the YAML configuration file."""
    config = read_config(config_path)
    print(f"✅ Configuration loaded from: {config_path}")
    return config

def load_and_filter_data(data_dir: Path, columns_to_keep: list) -> pd.DataFrame:
    """Loads all CELL CSVs, concatenates, and filters for essential columns."""
    print(f"1. Loading and filtering data from {data_dir}...")
    csv_files = sorted(glob.glob(str(data_dir / "CELL_*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CELL_*.csv files found in {data_dir}")

    df = pd.concat((pd.read_csv(f, usecols=lambda c: c in columns_to_keep) for f in csv_files), ignore_index=True)

    # Basic cleaning and type conversion
    df['eventdate'] = pd.to_datetime(df['eventdate'])
    df.drop_duplicates(subset=['CELL_', 'eventdate'], inplace=True)

    print(f"   ... Loaded {len(df):,} records from {len(csv_files)} files.")
    return df

def create_complete_index(df: pd.DataFrame, granularity: int) -> pd.DataFrame:
    """Expands the DataFrame to a full time-series index for every cell."""
    print("2. Creating complete time-series index for all cells...")

    unique_cells = df['CELL_'].unique()
    full_date_range = pd.date_range(
        start=df['eventdate'].min(),
        end=df['eventdate'].max(),
        freq=f'{granularity}min'
    )

    multi_index = pd.MultiIndex.from_product(
        [unique_cells, full_date_range],
        names=['CELL_', 'eventdate']
    )

    df.set_index(['CELL_', 'eventdate'], inplace=True)
    df = df.reindex(multi_index)

    df.sort_index(inplace=True)

    print(f"   ... Expanded DataFrame to {len(df):,} records.")
    return df

def impute_missing_values(df: pd.DataFrame, kpi_cols: list) -> pd.DataFrame:
    """Fill only from past observations, so future targets cannot enter inputs."""
    print("3. Filling missing values from past observations...")
    df[kpi_cols] = df.groupby(level='CELL_')[kpi_cols].ffill()
    df = df.fillna(0)

    print("   ... Imputation complete.")
    return df

def engineer_features(df: pd.DataFrame, kpi_cols: list, config_dict: dict, granularity: int) -> pd.DataFrame:
    """Creates time-based, lag, and rolling average features."""
    print("4. Engineering new features...")

    # Time-based features - check configuration
    eventdate = df.index.get_level_values('eventdate')
    
    # Check if time features should be added at all
    if config_dict.get('add_timefeats', True):  # Default to True for backward compatibility
        if config_dict.get('cyclic_timefeats', True):  # Default to True for backward compatibility
            # Cyclic encoding
            df['hour_sin'] = np.sin(2 * np.pi * eventdate.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * eventdate.hour / 24)
            df['day_of_week_sin'] = np.sin(2 * np.pi * eventdate.dayofweek / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * eventdate.dayofweek / 7)
        else:
            # Non-cyclic encoding
            df['hour'] = eventdate.hour
            df['day_of_week'] = eventdate.dayofweek
        
        # is_weekend is added regardless of cyclic_timefeats setting
        df['is_weekend'] = (eventdate.dayofweek >= 5).astype(int)

    # Advanced Features (Lag & Rolling) - only if configured
    steps_per_hour = 60 // granularity
    grouped = df.groupby(level='CELL_')[kpi_cols]

    # Lag Features
    if 'lag_hours' in config_dict and config_dict['lag_hours']:
        for lag_hours in config_dict['lag_hours']:
            lag_steps = lag_hours * steps_per_hour
            df[[f'{col}_lag_{lag_hours}h' for col in kpi_cols]] = grouped.shift(lag_steps)

    # Rolling Average Features
    if 'rolling_avg_hours' in config_dict and config_dict['rolling_avg_hours']:
        window_steps = config_dict['rolling_avg_hours'] * steps_per_hour
        rolling_df = grouped.rolling(window=window_steps, min_periods=1).mean()
        df[[f'{col}_roll_avg_{config_dict["rolling_avg_hours"]}h' for col in kpi_cols]] = rolling_df.reset_index(level=0, drop=True)

    df.fillna(0, inplace=True)

    # Count features created
    num_time_features = 0
    if config_dict.get('add_timefeats', True):
        if config_dict.get('cyclic_timefeats', True):
            num_time_features = 5  # hour_sin, hour_cos, day_of_week_sin, day_of_week_cos, is_weekend
        else:
            num_time_features = 3  # hour, day_of_week, is_weekend
    
    num_lag_features = len(config_dict.get('lag_hours', [])) * len(kpi_cols) if config_dict.get('lag_hours') else 0
    num_roll_features = len(kpi_cols) if config_dict.get('rolling_avg_hours') else 0
    total_features = num_time_features + num_lag_features + num_roll_features
    print(f"   ... Created {total_features} new features.")
    return df

def create_static_features(unique_cells: list) -> pd.DataFrame:
    """
    Creates a DataFrame of static features for each unique cell in a specific order.
    Does NOT include graph-based features.
    """
    print("... Engineering static features...")
    # Use the provided list of unique_cells to enforce order
    static_df = pd.DataFrame(index=unique_cells)
    static_df.index.name = 'CELL_'

    # 1. Parse CELL_ ID to get Sector and Band
    # Assumes format {SITE}-{SECTOR}{BAND}, e.g., 'SYN0001-10'.
    static_df['SECTOR'] = static_df.index.str.split('-').str[1].str[0].astype(int)
    static_df['BAND'] = static_df.index.str.split('-').str[1].str[1].astype(int)

    # Unknown bands have capacity 0 instead of an undefined numeric feature.
    band_to_mhz = {0: 20, 1: 10, 2: 20, 3: 10}
    static_df['BANDWIDTH_MHZ'] = static_df['BAND'].map(band_to_mhz).fillna(0)
    static_df['IS_HIGH_CAPACITY'] = (static_df['BANDWIDTH_MHZ'] == 20).astype(int)
    
    # 3. One-hot encode categorical features to make them numerical
    static_df = pd.get_dummies(static_df, columns=['SECTOR', 'BAND'], prefix_sep='_')
    
    print(f"   ... Created {static_df.shape[1]} static features for {len(static_df)} cells.")
    return static_df



def create_sequences(
    df: pd.DataFrame,
    input_steps: int,
    output_steps: int,
    feature_cols: List[str],
    target_cols: List[str],
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]], pd.Index]:
    """
    Create overlapping sequences for training/testing with multi-horizon optimization.
    
    Args:
        df: DataFrame with MultiIndex (CELL_, eventdate)
        input_steps: Number of historical timesteps (e.g., 48 = 12 hours at 15min granularity)
        output_steps: Number of future timesteps to predict (e.g., 48 = 12 hours)
        feature_cols: List of feature columns to use as input
        target_cols: List of target columns to predict
        stride: Step size for sliding window (default=1)
    
    Returns:
        X: (N_sequences, input_steps, n_cells, n_features)
        Y: (N_sequences, output_steps, n_cells, n_targets)
        timestamps: List of (input_start, input_end, output_end) tuples
    """
    print(f"5. Creating sequences with input_steps={input_steps}, output_steps={output_steps}, stride={stride}...")
    if input_steps < 1 or output_steps < 1 or stride < 1:
        raise ValueError('input_steps, output_steps, and stride must be positive.')
    
    # Sort by timestamp first (important for sequence ordering)
    df_sorted = df.sort_index(level=['eventdate', 'CELL_'])
    
    # Get unique values
    unique_cells = df_sorted.index.get_level_values('CELL_').unique()
    unique_timestamps = df_sorted.index.get_level_values('eventdate').unique()
    n_cells = len(unique_cells)
    n_timestamps = len(unique_timestamps)
    
    print(f"   ... Processing {n_cells} cells across {n_timestamps} timestamps")
    
    # Unstack to get cells as columns (much faster than pivot for large data)
    df_unstacked_features = df_sorted[feature_cols].unstack(level='CELL_').fillna(0)
    df_unstacked_targets = df_sorted[target_cols].unstack(level='CELL_').fillna(0)
    
    # Reindex to ensure all timestamps are present
    df_unstacked_features = df_unstacked_features.reindex(unique_timestamps, fill_value=0)
    df_unstacked_targets = df_unstacked_targets.reindex(unique_timestamps, fill_value=0)
    
    # Convert to numpy arrays and reshape
    # Shape will be (timestamps, cells * features) - need to reshape
    feature_columns = pd.MultiIndex.from_product([feature_cols, unique_cells])
    target_columns = pd.MultiIndex.from_product([target_cols, unique_cells])
    feature_array = df_unstacked_features.reindex(columns=feature_columns).to_numpy(dtype=np.float32).reshape(
        n_timestamps, len(feature_cols), n_cells).transpose(0, 2, 1)
    target_array = df_unstacked_targets.reindex(columns=target_columns).to_numpy(dtype=np.float32).reshape(
        n_timestamps, len(target_cols), n_cells).transpose(0, 2, 1)
    
    # Calculate valid range
    max_start_idx = n_timestamps - input_steps - output_steps
    if max_start_idx < 0:
        raise ValueError(f"Not enough timestamps ({n_timestamps}) for the requested "
                        f"input_steps ({input_steps}) + output_steps ({output_steps})")
    
    # Calculate number of sequences
    n_sequences = (max_start_idx // stride) + 1
    
    # Use numpy's advanced indexing for efficient sequence creation
    seq_indices = np.arange(0, max_start_idx + 1, stride)
    
    # Create index arrays for efficient slicing
    input_indices = seq_indices[:, None] + np.arange(input_steps)
    output_indices = seq_indices[:, None] + np.arange(input_steps, input_steps + output_steps)
    
    # Extract sequences using advanced indexing
    X = feature_array[input_indices]
    Y = target_array[output_indices]
    
    # Create timestamps list
    timestamps = [
        (unique_timestamps[i], 
        unique_timestamps[i + input_steps - 1],
        unique_timestamps[i + input_steps + output_steps - 1])
        for i in seq_indices
    ]
    
    print(f"   ... Created {n_sequences} sequences")
    print(f"   ... X shape: {X.shape} (sequences, input_steps, cells, features)")
    print(f"   ... Y shape: {Y.shape} (sequences, output_steps, cells, targets)")
    
    return X, Y, timestamps, unique_cells

def split_sequences(
    X: np.ndarray,
    Y: np.ndarray,
    timestamps: List,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    data_split: Dict = None,
    granularity_min: int = 15,
) -> Dict:
    """
    Split by forecast timestamps; drop windows whose targets cross a boundary.
    
    Args:
        X: Input sequences
        Y: Output sequences
        timestamps: List of timestamp tuples
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
    
    Returns:
        Dictionary containing train/val/test splits
    """
    print("6. Splitting sequences into train/val/test sets...")
    
    if len(X) != len(Y) or len(X) != len(timestamps) or not len(X):
        raise ValueError('X, Y, and timestamps must have equal, nonzero lengths.')
    target_start = pd.DatetimeIndex([t[1] for t in timestamps]) + pd.Timedelta(minutes=granularity_min)
    target_end = pd.DatetimeIndex([t[2] for t in timestamps])
    if not target_start.is_monotonic_increasing:
        raise ValueError('Sequence timestamps must be chronological.')
    if data_split:
        boundaries = [(pd.Timestamp(data_split[f'{s}_start']), pd.Timestamp(data_split[f'{s}_end']))
                      for s in ('train', 'val', 'test')]
        if any(a >= b for a, b in boundaries) or any(boundaries[i][1] > boundaries[i+1][0] for i in (0, 1)):
            raise ValueError('Date splits must be ordered, nonoverlapping, nonempty intervals.')
    else:
        if not (0 < train_ratio < 1 and 0 < val_ratio and train_ratio + val_ratio < 1):
            raise ValueError('Ratios must leave nonempty train, validation, and test partitions.')
        n_train, n_val = int(len(X) * train_ratio), int(len(X) * val_ratio)
        if n_train < 1 or n_val < 1:
            raise ValueError('Too few sequences for the requested split ratios.')
        cut1, cut2 = target_start[n_train], target_start[n_train+n_val]
        boundaries = [(target_start[0], cut1), (cut1, cut2),
                      (cut2, target_end[-1] + pd.Timedelta(minutes=granularity_min))]
    splits = {}
    for split, (start, end) in zip(('train', 'val', 'test'), boundaries):
        indices = np.flatnonzero((target_start >= start) & (target_end < end))
        if not len(indices):
            raise ValueError(f'No complete forecasts in the {split} interval {start} to {end}.')
        splits[f'X_{split}'], splits[f'Y_{split}'] = X[indices], Y[indices]
        splits[f'timestamps_{split}'] = [timestamps[i] for i in indices]
        print(f'   ... {split}: {len(indices)} sequences; targets in [{start}, {end})')
    return splits

def save_sequences(splits: Dict, output_dir: Path, config: Dict):
    """Save sequences and metadata to disk."""
    print(f"7. Saving sequences to {output_dir}...")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save numpy arrays
    for key in ['X_train', 'Y_train', 'X_val', 'Y_val', 'X_test', 'Y_test']:
        np.save(output_dir / f'{key}.npy', splits[key])
    
    # Save timestamps as pickle (preserves datetime objects)
    import pickle
    for key in ['timestamps_train', 'timestamps_val', 'timestamps_test']:
        with open(output_dir / f'{key}.pkl', 'wb') as f:
            pickle.dump(splits[key], f)
    
    # Save metadata
    metadata = {
        'schema_version': 2,
        'dataset_id': output_dir.name,
        'split_strategy': 'target_dates' if config.get('data_split') else 'purged_ratios',
        'data_split': config.get('data_split'),
        'imputation': 'causal_forward_fill',
        'preprocessing_config': config['data'],
        'input_steps': config['data']['input_steps'],
        'output_steps': config['data']['output_steps'],
        'granularity_min': config['data']['granularity_min'],
        'n_cells': splits['X_train'].shape[2],
        'n_features': splits['X_train'].shape[3],
        'n_targets': splits['Y_train'].shape[3],
        'feature_columns': config['feature_columns'],
        'target_columns': config['target_columns'],
        'train_sequences': len(splits['X_train']),
        'val_sequences': len(splits['X_val']),
        'test_sequences': len(splits['X_test']),
        'cell_order': config['cell_order'],
    }
    
    with open(output_dir / 'metadata.yaml', 'w') as f:
        yaml.dump(metadata, f)
    
    print(f"   ... Saved all sequences and metadata")

def main(config_path: str):
    """Main function to run the entire preprocessing pipeline with sequence creation."""
    start_time = time.time()

    config = load_config(config_path)
    cfg = config['data']

    data_dir = Path(cfg['data_dir'])
    output_root = Path(cfg['output_dir'])
    dataset_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8]
    output_dir = output_root / 'datasets' / dataset_id
    
    # Combine essential and kpi columns for loading
    columns_to_load = cfg['essential_columns'] + cfg['kpi_columns']

    # Execute the preprocessing pipeline
    df = load_and_filter_data(data_dir, columns_to_load)
    df = create_complete_index(df, cfg['granularity_min'])
    df = impute_missing_values(df, cfg['kpi_columns'])
    df = engineer_features(df, cfg['kpi_columns'], cfg, cfg['granularity_min'])

    # Define feature and target columns
    # Features include all KPI columns plus engineered features
    feature_cols = cfg['kpi_columns'].copy()
    
    # Add time features based on configuration
    if cfg.get('add_timefeats', True):
        if cfg.get('cyclic_timefeats', True):
            feature_cols.extend(['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'is_weekend'])
        else:
            feature_cols.extend(['hour', 'day_of_week', 'is_weekend'])

    # Add lag and rolling features if they exist
    if cfg.get('lag_hours'):
        for lag_hours in cfg['lag_hours']: 
            feature_cols.extend([f'{col}_lag_{lag_hours}h' for col in cfg['kpi_columns']])
    
    if cfg.get('rolling_avg_hours'):
        feature_cols.extend([f'{col}_roll_avg_{cfg["rolling_avg_hours"]}h' for col in cfg['kpi_columns']])
    
    # Target columns are the KPIs we want to predict
    target_cols = config['target_kpis'] #cfg['kpi_columns']
    
    # Store in config for metadata
    config['feature_columns'] = feature_cols
    config['target_columns'] = target_cols
    
    # Create sequences
    X, Y, timestamps, cell_order = create_sequences(
        df=df,
        input_steps=cfg['input_steps'],
        output_steps=cfg['output_steps'],
        feature_cols=feature_cols,
        target_cols=target_cols,
        stride=cfg.get('stride', 1)  # Default stride of 1 if not specified
    )

    # 5. Create and save static features using the retrieved cell order
    static_df = create_static_features(cell_order)
    static_features_array = static_df.to_numpy(dtype=np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'static_features.npy', static_features_array)
    print(f"   ... Saved static features with shape {static_features_array.shape} to {output_dir / 'static_features.npy'}")
    config['static_feature_columns'] = static_df.columns.tolist()
    config['cell_order'] = cell_order.tolist()
    
    # Split into train/val/test
    splits = split_sequences(
        X, Y, timestamps,
        train_ratio=cfg.get('train_ratio', 0.7),
        val_ratio=cfg.get('val_ratio', 0.15),
        data_split=config.get('data_split'),
        granularity_min=cfg['granularity_min'],
    )
    
    # Save everything
    save_sequences(splits, output_dir, config)
    if cfg.get('save_preprocessed_df', False):
        df.reset_index().to_parquet(output_dir / cfg['preprocessed_filename'], index=False)
        print(f"   ... Also saved preprocessed DataFrame to {output_dir / cfg['preprocessed_filename']}")
    manifest = {'dataset_id': dataset_id, 'sha256': {
        p.name: file_sha256(p) for p in sorted(output_dir.iterdir()) if p.is_file()
    }}
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    output_root.mkdir(parents=True, exist_ok=True)
    pointer = output_root / f'.latest-{dataset_id}.json'
    pointer.write_text(json.dumps({'dataset_id': dataset_id, 'path': f'datasets/{dataset_id}'}, indent=2) + '\n')
    pointer.replace(output_root / 'latest.json')

    end_time = time.time()
    print("-" * 50)
    print("✅ Preprocessing and sequence creation complete!")
    print(f"   Final shapes:")
    print(f"     - Train: X={splits['X_train'].shape}, Y={splits['Y_train'].shape}")
    print(f"     - Val:   X={splits['X_val'].shape}, Y={splits['Y_val'].shape}")
    print(f"     - Test:  X={splits['X_test'].shape}, Y={splits['Y_test'].shape}")
    print(f"   Output directory: {output_dir}")
    print(f"   Total time: {end_time - start_time:.2f} seconds")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a new versioned telecom dataset.')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    main(config_path=parser.parse_args().config)
