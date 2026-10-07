import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yaml
import os


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAPE as percentage
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_metrics(y_true: np.ndarray, 
                     y_pred: np.ndarray, 
                     metrics: List[str] = None) -> Dict[str, float]:
    """
    Calculate multiple evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values  
        metrics: List of metrics to calculate ('mae', 'rmse', 'mape')
        
    Returns:
        Dictionary mapping metric names to values
    """
    if metrics is None:
        metrics = ['mae', 'rmse', 'mape']
    
    results = {}
    
    for metric in metrics:
        if metric == 'mae':
            results['mae'] = mean_absolute_error(y_true, y_pred)
        elif metric == 'rmse':
            results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric == 'mape':
            results['mape'] = calculate_mape(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    return results


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)



def split_data_by_dates(df: pd.DataFrame, 
                       train_start: str, train_end: str,
                       val_start: str, val_end: str, 
                       test_start: str, test_end: str,
                       verbose: bool = True) -> tuple:
    """
    Split DataFrame by date ranges.
    
    Args:
        df: Input DataFrame with 'eventdate' column
        train_start, train_end: Training period dates
        val_start, val_end: Validation period dates  
        test_start, test_end: Test period dates
        verbose: Whether to print split information
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    df['eventdate'] = pd.to_datetime(df['eventdate'])
    
    # Convert date strings to pandas datetime
    train_start = pd.to_datetime(train_start)
    train_end = pd.to_datetime(train_end)
    val_start = pd.to_datetime(val_start)
    val_end = pd.to_datetime(val_end)
    test_start = pd.to_datetime(test_start)
    test_end = pd.to_datetime(test_end)

    # Create masks for each period
    train_mask = (df['eventdate'] >= train_start) & (df['eventdate'] < train_end)
    val_mask = (df['eventdate'] >= val_start) & (df['eventdate'] < val_end)
    test_mask = (df['eventdate'] >= test_start) & (df['eventdate'] <= test_end)

    train_df = df[train_mask].copy().reset_index(drop=True)
    val_df = df[val_mask].copy().reset_index(drop=True)
    test_df = df[test_mask].copy().reset_index(drop=True)

    if verbose:
        print("Data Split Information:")
        print(f"Train: {len(train_df)} samples ({train_start} to {train_end})")
        print(f"Val: {len(val_df)} samples ({val_start} to {val_end})")
        print(f"Test: {len(test_df)} samples ({test_start} to {test_end})")

    return train_df, val_df, test_df




