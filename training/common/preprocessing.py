"""
Functions for preprocessing EEG data and engineering features.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

from .utils.logging import logger
from .utils.common import calculate_samples_from_seconds

def apply_moving_average(df: pd.DataFrame, window_seconds: float = 0, min_periods: int = 1) -> pd.DataFrame:
    """
    Apply moving average smoothing to numeric columns.
    
    Args:
        df: Input DataFrame with time series data
        window_seconds: Size of the moving window in seconds (0 to skip smoothing)
        min_periods: Minimum number of observations required for valid window
        
    Returns:
        DataFrame with smoothed values
    """
    if window_seconds <= 0:
        return df
    
    # Make a copy of the dataframe
    smoothed_df = df.copy()
    
    # Convert window_seconds to number of samples
    window_samples = calculate_samples_from_seconds(df, window_seconds)
    
    # Apply rolling window only to numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    smoothed_df[numeric_cols] = df[numeric_cols].rolling(
        window=window_samples, 
        min_periods=min_periods,
        center=True
    ).mean()
    
    # Fill any NaN values introduced at the edges
    smoothed_df = smoothed_df.fillna(df)
    
    return smoothed_df

def preprocess_eeg_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw EEG data.
    
    Args:
        df: Raw EEG data
        
    Returns:
        Preprocessed EEG data
    """
    logger.info(f"Preprocessing data with shape {df.shape}")
    
    try:
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Handle timestamps
        if 'TimeStamp' in df_clean.columns:
            try:
                # Handle timestamp format variations
                df_clean['TimeStamp'] = pd.to_datetime(df_clean['TimeStamp'], errors='coerce')
                
                # Drop rows with invalid timestamps
                invalid_timestamps = df_clean['TimeStamp'].isna()
                if invalid_timestamps.any():
                    num_invalid = invalid_timestamps.sum()
                    logger.warning(f"Dropping {num_invalid} rows with invalid timestamps")
                    df_clean = df_clean.dropna(subset=['TimeStamp'])
                    
                logger.info(f"Successfully converted timestamps. Remaining rows: {len(df_clean)}")
                
                # Sort by timestamp
                df_clean = df_clean.sort_values('TimeStamp')
                
            except Exception as e:
                logger.warning(f"Error processing timestamps: {str(e)}. Continuing without timestamp conversion.")
        
        # Drop rows with missing values
        df_clean = df_clean.dropna()

        print(f"NaNs in numeric columns: {df_clean.isna().sum()}")

        # Remove outliers (optional)
        # Here you could add code to identify and remove outliers

        # Sample with sliding window
        df_clean = sample_by_sliding_window(
            df_clean,
            window_size_seconds=5.0,
            samples_per_window=50,
            timestamp_col='TimeStamp'
        )
        
        logger.info(f"Preprocessing complete. Output shape: {df_clean.shape}")
        return df_clean
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

def engineer_eeg_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from preprocessed EEG data.
    
    Args:
        df: Preprocessed EEG data
        
    Returns:
        DataFrame with engineered features
    """
    logger.info(f"Engineering features from data with shape {df.shape}")
    
    try:
        # Start with preprocessed data
        df_features = df.copy()
        
        # Select only numeric columns for feature engineering
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Exclude any non-EEG data columns
        exclude_patterns = [
            "TimeStamp",
            "Accelerometer_X",
            "Accelerometer_Y",
            "Accelerometer_Z",
            "Gyro_X",
            "Gyro_Y",
            "Gyro_Z",
            "Elements"
        ]
        
        eeg_cols = [col for col in numeric_cols if not any(pattern in col for pattern in exclude_patterns)]
        
        if not eeg_cols:
            logger.warning("No EEG data columns found for feature engineering")
            return df_features
            
        # Basic statistical features
        logger.debug("Calculating statistical features")
        for col in eeg_cols:
            # Rolling window statistics (with a 1-second window)
            window_size = calculate_samples_from_seconds(df, 1.0)
            
            # Only calculate if we have enough data
            if len(df) >= window_size:
                # Rolling mean
                df_features[f"{col}_mean"] = df[col].rolling(window=window_size, center=True).mean()
                
                # Rolling standard deviation
                df_features[f"{col}_std"] = df[col].rolling(window=window_size, center=True).std()
                
                # Rolling min/max range
                df_features[f"{col}_range"] = (
                    df[col].rolling(window=window_size, center=True).max() - 
                    df[col].rolling(window=window_size, center=True).min()
                )
        
        # Fill any NaN values created by the rolling windows
        df_features = df_features.ffill().bfill()
        
        # Add more advanced features here as needed
        # ...
        
        logger.info(f"Feature engineering complete. Output shape: {df_features.shape}")
        return df_features
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise

def select_features(X: np.ndarray, y: np.ndarray, method: str = 'anova', k: int = 20) -> np.ndarray:
    """
    Select most important features based on relationship with target variable.
    
    Args:
        X: Feature matrix
        y: Target variable
        method: Feature selection method ('anova' or 'mutual_info')
        k: Number of top features to select
        
    Returns:
        Selected feature matrix
    """
    logger.info(f"Selecting {k} features using {method} method")
    
    try:
        if method == 'anova':
            selector = SelectKBest(f_classif, k=k)
        elif method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=k)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
            
        X_selected = selector.fit_transform(X, y)
        
        # Get indices of selected features
        selected_indices = selector.get_support(indices=True)
        
        logger.info(f"Feature selection complete. Selected {X_selected.shape[1]} features")
        return X_selected, selected_indices
        
    except Exception as e:
        logger.error(f"Error in feature selection: {str(e)}")
        raise

def process_features(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Process features with forward and backward fill for missing values.
    
    Args:
        df_features: DataFrame with features
        
    Returns:
        Processed DataFrame with no missing values
    """
    # Replace deprecated method with recommended alternatives
    return df_features.ffill().bfill()

def sample_by_sliding_window(df, window_size_seconds=5.0, samples_per_window=10, timestamp_col='TimeStamp'):
    """
    Sample data using a time-based sliding window approach with interquartile sampling.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data to sample
    window_size_seconds : float, default=5.0
        Size of the sliding window in seconds
    samples_per_window : int, default=10
        Number of samples to select from each window
    timestamp_col : str, default='TimeStamp'
        Name of the column containing timestamps
        
    Returns:
    --------
    pandas.DataFrame
        Sampled DataFrame
    
    Notes:
    ------
    This function:
    1. Creates non-overlapping windows of specified size in seconds
    2. For each window, selects evenly distributed samples based on interquartile range
    3. Maintains temporal coherence within each window
    """
    if not isinstance(df, pd.DataFrame):
        logger.error("Input must be a pandas DataFrame")
        raise TypeError("Input must be a pandas DataFrame")
    
    if timestamp_col not in df.columns:
        logger.error(f"DataFrame must contain a '{timestamp_col}' column")
        raise ValueError(f"DataFrame must contain a '{timestamp_col}' column")
    
    # Check if DataFrame is empty
    if df.empty:
        logger.warning("Empty DataFrame provided to sample_by_sliding_window")
        return df.copy()
    
    # Check for missing values in timestamp column
    if df[timestamp_col].isna().any():
        missing_timestamps = df[timestamp_col].isna().sum()
        logger.warning(f"Found {missing_timestamps} missing values in timestamp column. Dropping these rows.")
        df = df.dropna(subset=[timestamp_col])
    
    # Check if DataFrame is empty after dropping NaNs
    if df.empty:
        logger.warning("DataFrame is empty after dropping NaN timestamps")
        return pd.DataFrame(columns=df.columns)
    
    # Ensure timestamp column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        try:
            logger.info(f"Converting '{timestamp_col}' to datetime format")
            df = df.copy()
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
            
            # Check for conversion failures
            if df[timestamp_col].isna().any():
                na_count = df[timestamp_col].isna().sum()
                logger.warning(f"{na_count} timestamps could not be converted to datetime. Dropping these rows.")
                df = df.dropna(subset=[timestamp_col])
        except Exception as e:
            logger.error(f"Error converting timestamps: {str(e)}")
            raise ValueError(f"Could not convert '{timestamp_col}' column to datetime: {str(e)}")
    
    # Check if DataFrame is still valid
    if df.empty:
        logger.warning("DataFrame is empty after timestamp conversion")
        return pd.DataFrame(columns=df.columns)
    
    # Sort by timestamp to ensure correct order
    df = df.sort_values(by=timestamp_col).reset_index(drop=True)
    
    # Get start and end timestamps
    start_time = df[timestamp_col].min()
    end_time = df[timestamp_col].max()
    
    # Calculate timespan
    time_span = (end_time - start_time).total_seconds()
    logger.info(f"Data spans {time_span:.2f} seconds from {start_time} to {end_time}")
    
    # Create list to store sampled data
    sampled_data = []
    
    # Calculate window duration
    window_duration = pd.Timedelta(seconds=window_size_seconds)
    
    # Create sliding windows
    current_time = start_time
    window_count = 0
    
    while current_time < end_time:
        # Define window bounds
        window_end = current_time + window_duration
        
        # Extract data for this window
        window_data = df[(df[timestamp_col] >= current_time) & 
                         (df[timestamp_col] < window_end)]
        
        # If we have data in this window
        if len(window_data) > 0:
            window_count += 1
            logger.debug(f"Window {window_count}: {len(window_data)} data points")
            
            if len(window_data) <= samples_per_window:
                # If we have fewer data points than requested samples, keep all points
                sampled_data.append(window_data)
                logger.debug(f"Window {window_count}: Keeping all {len(window_data)} points (fewer than requested)")
            else:
                try:
                    # Select samples based on interquartile ranges
                    # Calculate indices to select
                    indices = np.linspace(0, len(window_data) - 1, samples_per_window, dtype=int)
                    sampled_window = window_data.iloc[indices]
                    sampled_data.append(sampled_window)
                    logger.debug(f"Window {window_count}: Selected {len(sampled_window)} points")
                except Exception as e:
                    logger.error(f"Error sampling window {window_count}: {str(e)}")
                    # Fallback to random sampling if there's an error
                    try:
                        sampled_window = window_data.sample(min(samples_per_window, len(window_data)))
                        sampled_data.append(sampled_window)
                        logger.warning(f"Window {window_count}: Fell back to random sampling, selected {len(sampled_window)} points")
                    except Exception as e2:
                        logger.error(f"Fallback sampling also failed for window {window_count}: {str(e2)}")
                        # Last resort: keep all data in this window
                        sampled_data.append(window_data)
                        logger.warning(f"Window {window_count}: Keeping all {len(window_data)} points (sampling failed)")
        else:
            logger.debug(f"Empty window at {current_time}")
        
        # Slide the window by its full width (non-overlapping)
        current_time = window_end
    
    # Combine all sampled windows
    if sampled_data:
        result = pd.concat(sampled_data, ignore_index=True)
        logger.info(f"Sampling complete: {len(df)} -> {len(result)} data points ({len(result)/len(df)*100:.1f}%)")
        
        # Check for NaN values in result
        nan_counts = result.isna().sum()
        total_nans = nan_counts.sum()
        if total_nans > 0:
            logger.warning(f"Sampled data contains {total_nans} NaN values")
            top_nan_cols = nan_counts[nan_counts > 0].sort_values(ascending=False).head(5)
            for col, count in top_nan_cols.items():
                logger.warning(f"  Column '{col}': {count} NaNs ({count/len(result)*100:.1f}%)")
        
        return result
    else:
        logger.warning("No data sampled")
        return pd.DataFrame(columns=df.columns)

def process_eeg_files_with_sampling(file_dfs, window_size_seconds=5.0, samples_per_window=10, 
                                    timestamp_col='TimeStamp'):
    """
    Apply sliding window sampling to multiple EEG files.
    
    Parameters:
    -----------
    file_dfs : dict
        Dictionary of DataFrames, where keys are file IDs and values are DataFrames
    window_size_seconds : float, default=5.0
        Size of the sliding window in seconds
    samples_per_window : int, default=10
        Number of samples to select from each window
    timestamp_col : str, default='TimeStamp'
        Name of the column containing timestamps
        
    Returns:
    --------
    dict
        Dictionary of sampled DataFrames with the same keys as input
    """
    sampled_dfs = {}
    
    for file_id, df in file_dfs.items():
        # Apply sliding window sampling to each file
        sampled_df = sample_by_sliding_window(
            df, 
            window_size_seconds=window_size_seconds, 
            samples_per_window=samples_per_window,
            timestamp_col=timestamp_col
        )
        
        # Store the sampled DataFrame
        sampled_dfs[file_id] = sampled_df
    
    return sampled_dfs

def analyze_data_quality(df, show_plots=True, nan_threshold=0.5):
    """
    Analyze data quality, particularly focusing on missing values.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to analyze
    show_plots : bool, default=True
        Whether to generate and return visualization plots
    nan_threshold : float, default=0.5
        Threshold for highlighting high-NaN columns (as a fraction)

    Returns:
    --------
    dict
        Dictionary containing analysis results:
        - 'summary': Overall summary statistics
        - 'nan_columns': DataFrame with NaN statistics per column
        - 'high_nan_columns': List of columns with high NaN percentage
        - 'clean_columns': List of columns with no NaNs
        - 'plot': Matplotlib figure object (if show_plots=True)
    """
    from .utils.logging import logger
    
    # Create result dictionary
    results = {}
    
    # Get basic dataframe info
    n_rows, n_cols = df.shape
    dtypes = df.dtypes.value_counts()
    
    # Count total NaNs
    total_nans = df.isna().sum().sum()
    nan_percentage = (total_nans / (n_rows * n_cols)) * 100
    
    # Create summary statistics
    summary = {
        'total_rows': n_rows,
        'total_columns': n_cols,
        'dtype_counts': dtypes.to_dict(),
        'total_cells': n_rows * n_cols,
        'total_missing': int(total_nans),
        'missing_percentage': nan_percentage
    }
    results['summary'] = summary
    
    logger.info(f"Data shape: {n_rows} rows × {n_cols} columns")
    logger.info(f"Total missing values: {total_nans} ({nan_percentage:.2f}%)")
    
    # Analyze NaNs by column
    nan_counts = df.isna().sum()
    nan_percentages = (nan_counts / n_rows) * 100
    
    # Create a DataFrame with NaN statistics
    nan_df = pd.DataFrame({
        'column': nan_counts.index,
        'nan_count': nan_counts.values,
        'nan_percentage': nan_percentages.values,
        'dtype': df.dtypes.values
    })
    nan_df = nan_df.sort_values('nan_count', ascending=False).reset_index(drop=True)
    
    # Filter columns with high NaN percentages
    high_nan_threshold = nan_threshold * 100
    high_nan_cols = nan_df[nan_df['nan_percentage'] > high_nan_threshold]['column'].tolist()
    clean_cols = nan_df[nan_df['nan_count'] == 0]['column'].tolist()
    
    results['nan_columns'] = nan_df
    results['high_nan_columns'] = high_nan_cols
    results['clean_columns'] = clean_cols
    
    # Log findings
    if high_nan_cols:
        logger.warning(f"Found {len(high_nan_cols)} columns with >{high_nan_threshold}% NaN values:")
        for col in high_nan_cols[:10]:  # Show up to 10 columns
            col_idx = nan_df[nan_df['column'] == col].index[0]
            logger.warning(f"  - {col}: {nan_df.loc[col_idx, 'nan_percentage']:.1f}% NaNs ({nan_df.loc[col_idx, 'nan_count']} values)")
        if len(high_nan_cols) > 10:
            logger.warning(f"  ... and {len(high_nan_cols) - 10} more columns")
    
    logger.info(f"Found {len(clean_cols)} columns with no NaN values")
    
    # Generate plots if requested
    if show_plots:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Set up the figure
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot 1: Top 20 columns with most NaNs
            top_n = min(20, len(nan_df))
            top_cols = nan_df.head(top_n)
            
            ax1 = axes[0]
            bars = ax1.barh(top_cols['column'], top_cols['nan_percentage'], color='skyblue')
            ax1.set_xlabel('NaN Percentage (%)')
            ax1.set_ylabel('Column')
            ax1.set_title(f'Top {top_n} Columns with Most NaNs')
            ax1.grid(axis='x', linestyle='--', alpha=0.7)
            ax1.set_xlim(0, max(top_cols['nan_percentage']) * 1.1)
            
            # Add percentage labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                label = f"{width:.1f}%"
                ax1.text(width + max(top_cols['nan_percentage']) * 0.02, 
                        bar.get_y() + bar.get_height()/2, 
                        label, 
                        va='center')
            
            # Plot 2: NaN distribution across all columns (histogram)
            ax2 = axes[1]
            bins = np.linspace(0, 100, 21)  # 5% intervals
            ax2.hist(nan_percentages, bins=bins, color='lightgreen', edgecolor='black')
            ax2.set_xlabel('NaN Percentage (%)')
            ax2.set_ylabel('Number of Columns')
            ax2.set_title('Distribution of NaNs Across All Columns')
            ax2.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add vertical line for threshold
            ax2.axvline(high_nan_threshold, color='red', linestyle='--', 
                        label=f'Threshold ({high_nan_threshold}%)')
            ax2.legend()
            
            plt.tight_layout()
            results['plot'] = fig
        except Exception as e:
            logger.warning(f"Could not generate data quality plots: {str(e)}")
    
    # Provide recommendations
    recommendations = []
    
    if high_nan_cols:
        if len(high_nan_cols) / n_cols > 0.5:
            recommendations.append("Many columns have high NaN percentages. Consider more aggressive preprocessing or using models that handle missing values well.")
        else:
            recommendations.append(f"Consider dropping columns with high NaN percentages: {', '.join(high_nan_cols[:5])}" + 
                               (f"... and {len(high_nan_cols) - 5} more" if len(high_nan_cols) > 5 else ""))
    
    if nan_percentage > 20:
        recommendations.append("High overall missing data rate. Consider using imputation techniques or models robust to missing values.")
    
    results['recommendations'] = recommendations
    
    # Log recommendations
    if recommendations:
        logger.info("Recommendations based on data quality analysis:")
        for i, rec in enumerate(recommendations):
            logger.info(f"{i+1}. {rec}")
    
    return results 