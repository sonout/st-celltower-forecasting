"""Optional plotting helpers for verified predictions and their dataset snapshots."""

from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_forecast_vs_actual(data: Dict[str, Any], cell_id: str, kpi_name: str, sequence_idx: int):
    """
    Plots the historical input, ground truth, and prediction for a single sequence.
    """
    # Extract data from the dictionary
    preds, y_test, x_test = data['predictions'], data['y_test'], data['x_test']
    metadata = data['metadata']

    # Find the integer indices for the given cell and KPI names
    try:
        cell_idx = metadata['cell_order'].index(cell_id)
    except ValueError:
        raise ValueError(f"Cell ID '{cell_id}' not found. Please check the ID.")
    
    try:
        # The target KPI is what we want to plot from Y and predictions
        target_kpi_idx = metadata['target_columns'].index(kpi_name)
        # The same KPI also exists as an input feature in X
        feature_kpi_idx = metadata['feature_columns'].index(kpi_name)
    except ValueError:
        raise ValueError(f"KPI '{kpi_name}' not found in target/feature columns.")

    if sequence_idx >= len(preds):
        raise IndexError(f"Sequence index {sequence_idx} is out of bounds. Max index is {len(preds)-1}.")

    print(f"\n📈 Generating forecast plot for:")
    print(f"   - Cell: {cell_id} (index {cell_idx})")
    print(f"   - KPI: {kpi_name} (index {target_kpi_idx})")
    print(f"   - Test Sequence: {sequence_idx}")

    # --- Data Slicing ---
    # Shape: (sequences, timesteps, cells, features)
    history = x_test[sequence_idx, :, cell_idx, feature_kpi_idx]
    truth = y_test[sequence_idx, :, cell_idx, target_kpi_idx]
    prediction = preds[sequence_idx, :, cell_idx, target_kpi_idx]

    # --- Time Axes ---
    input_steps = len(history)
    output_steps = len(truth)
    history_axis = np.arange(input_steps)
    future_axis = np.arange(input_steps, input_steps + output_steps)

    # --- Plotting ---
    plt.figure()
    plt.plot(history_axis, history, color='gray', label='Historical Input (X_test)')
    plt.plot(future_axis, truth, 'o-', color='blue', label='Ground Truth (Y_test)')
    plt.plot(future_axis, prediction, 'o--', color='red', label='Model Prediction')
    
    # Add a vertical line to separate history from forecast
    plt.axvline(x=input_steps - 0.5, color='black', linestyle='--', linewidth=2, label='Prediction Start')

    plt.title(f"Forecast vs. Actuals for {kpi_name}\nCell: {cell_id} | Sequence: {sequence_idx}")
    plt.xlabel("Time Steps (1 step = 15 minutes)")
    plt.ylabel(kpi_name)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_error_by_horizon(data: Dict[str, Any], kpi_name: str):
    """
    Calculates and plots the Mean Absolute Error (MAE) for each prediction step in the horizon.
    """
    preds, y_test = data['predictions'], data['y_test']
    metadata = data['metadata']
    
    try:
        kpi_idx = metadata['target_columns'].index(kpi_name)
    except ValueError:
        raise ValueError(f"KPI '{kpi_name}' not found in target columns.")
        
    print(f"\n📊 Generating error by horizon plot for KPI: {kpi_name}")

    # Calculate absolute error for the specific KPI
    # Shape of preds/y_test: (sequences, output_steps, cells, kpis)
    absolute_errors = np.abs(preds[:, :, :, kpi_idx] - y_test[:, :, :, kpi_idx])
    
    # Average the error across all sequences and all cells for each time step
    # We average over axis 0 (sequences) and axis 2 (cells)
    mae_per_horizon = np.mean(absolute_errors, axis=(0, 2))
    
    horizon_steps = np.arange(1, len(mae_per_horizon) + 1)

    plt.figure()
    sns.barplot(x=horizon_steps, y=mae_per_horizon, color='skyblue')
    plt.title(f'Mean Absolute Error (MAE) by Prediction Horizon for {kpi_name}')
    plt.xlabel('Prediction Horizon (15-min Steps Ahead)')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.tight_layout()
    plt.show()

def plot_error_distribution(data: Dict[str, Any], kpi_name: str):
    """
    Plots a histogram of the prediction residuals (errors).
    """
    preds, y_test = data['predictions'], data['y_test']
    metadata = data['metadata']

    try:
        kpi_idx = metadata['target_columns'].index(kpi_name)
    except ValueError:
        raise ValueError(f"KPI '{kpi_name}' not found in target columns.")
        
    print(f"\n📉 Generating error distribution plot for KPI: {kpi_name}")
    
    # Calculate residuals (prediction - truth) and flatten into a 1D array
    residuals = (preds[:, :, :, kpi_idx] - y_test[:, :, :, kpi_idx]).flatten()
    
    mean_error = np.mean(residuals)
    std_error = np.std(residuals)
    
    plt.figure()
    sns.histplot(residuals, kde=True, bins=100)
    plt.axvline(x=0, color='red', linestyle='--', label='Zero Error')
    plt.axvline(x=mean_error, color='black', linestyle='-', label=f'Mean Error: {mean_error:.2f}')
    
    plt.title(f'Distribution of Prediction Errors (Residuals) for {kpi_name}\nMean={mean_error:.2f}, Std Dev={std_error:.2f}')
    plt.xlabel('Prediction Error (Prediction - Truth)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_top_n_error_cells(data: Dict[str, Any], kpi_name: str, n_cells: int):
    """
    Identifies and plots the top N cells with the highest average MAE.
    """
    preds, y_test = data['predictions'], data['y_test']
    metadata = data['metadata']
    
    try:
        kpi_idx = metadata['target_columns'].index(kpi_name)
    except ValueError:
        raise ValueError(f"KPI '{kpi_name}' not found in target columns.")
        
    print(f"\n🗼 Finding top {n_cells} worst-performing cells for KPI: {kpi_name}")
    
    # Calculate absolute error for the specific KPI
    absolute_errors = np.abs(preds[:, :, :, kpi_idx] - y_test[:, :, :, kpi_idx])
    
    # Average error across sequences (axis 0) and time horizons (axis 1) to get a single error value per cell
    mae_per_cell = np.mean(absolute_errors, axis=(0, 1))
    
    # Create a DataFrame for easy sorting and plotting
    cell_errors_df = pd.DataFrame({
        'CELL_ID': metadata['cell_order'],
        'MAE': mae_per_cell
    })
    
    top_cells = cell_errors_df.sort_values('MAE', ascending=False).head(n_cells)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='MAE', y='CELL_ID', data=top_cells, palette='viridis')
    plt.title(f'Top {n_cells} Cells with Highest MAE for {kpi_name}')
    plt.xlabel('Mean Absolute Error (MAE) - Averaged over all predictions')
    plt.ylabel('Cell ID')
    plt.tight_layout()
    plt.show()
