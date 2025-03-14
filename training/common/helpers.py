import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from scipy import stats
import gc
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('eeg_classification')

def apply_moving_average(df, window_seconds=0, min_periods=1):
    """
    Apply moving average smoothing to a DataFrame with time-indexed data
    
    Parameters:
    df (DataFrame): DataFrame with datetime index or TimeStamp column
    window_seconds (float): Size of the moving average window in seconds (0 for no smoothing)
    min_periods (int): Minimum number of observations required to have a value
    
    Returns:
    DataFrame: Smoothed DataFrame
    """
    # If no smoothing requested, return original DataFrame
    if window_seconds <= 0:
        return df
    
    # Check if DataFrame has datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try to use TimeStamp column if it exists
        if 'TimeStamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['TimeStamp']):
            # Set TimeStamp as index temporarily for smoothing
            df = df.set_index('TimeStamp')
            reset_index = True
        else:
            print("Warning: Cannot apply time-based smoothing without datetime index")
            return df
    else:
        reset_index = False
    
    # Get only numeric columns for smoothing
    numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    
    # Apply rolling window with time-based window
    smoothed_df = df.copy()
    smoothed_df[numeric_cols] = df[numeric_cols].rolling(
        window=f'{window_seconds}s', 
        min_periods=min_periods
    ).mean()
    
    # Fill NaN values that occur at the start of the window
    smoothed_df = smoothed_df.fillna(method='bfill').fillna(method='ffill')
    
    # Reset index if needed
    if reset_index:
        # Keep the TimeStamp column
        smoothed_df = smoothed_df.reset_index()
    
    return smoothed_df

def calculate_samples_from_seconds(df, seconds):
    """
    Calculate the number of samples that correspond to a time duration in seconds
    
    Parameters:
    df (DataFrame): DataFrame with datetime index or TimeStamp column
    seconds (float): Time duration in seconds
    
    Returns:
    int: Equivalent number of samples
    """
    # Check if DataFrame has datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try to use TimeStamp column if it exists
        if 'TimeStamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['TimeStamp']):
            # Get the first few timestamps to calculate sampling rate
            time_diff = df['TimeStamp'].diff().dropna()
        else:
            # Can't determine sampling rate, return None
            print("Warning: Cannot determine sampling rate without datetime information")
            return None
    else:
        # Use the index
        time_diff = df.index.to_series().diff().dropna()
    
    # Calculate median time difference between samples
    if len(time_diff) > 0:
        # Convert to seconds
        median_diff_seconds = time_diff.median().total_seconds()
        
        # Calculate number of samples for the requested duration
        if median_diff_seconds > 0:
            return int(seconds / median_diff_seconds)
        else:
            print("Warning: Zero or negative time difference between samples")
            return None
    else:
        print("Warning: Not enough samples to calculate sampling rate")
        return None

def create_coherent_time_series_bundles_disk(file_dfs=None, combined_df=None, 
                                           bundle_size=30, step_size=15,
                                           bundle_time_seconds=None, step_time_seconds=None,
                                           evenly_spaced_samples=None, time_window_seconds=None,
                                           smoothing_seconds=0, feature_columns=None, 
                                           max_files_per_batch=5, max_bundles_per_file=None,
                                           max_memory_mb=1000, chunk_size=1000,
                                           output_dir='./eeg_bundles', sampling_rate=256):
    """
    Memory-efficient version that creates time series bundles in batches and saves to disk
    
    Parameters:
    file_dfs (dict): Dictionary with DataFrames for each file
    combined_df (DataFrame): Combined DataFrame with file identifiers
    bundle_size (int): Number of consecutive rows to include in each bundle (ignored if bundle_time_seconds or evenly_spaced_samples is provided)
    step_size (int): Step size for sliding window when creating bundles (ignored if step_time_seconds is provided)
    bundle_time_seconds (float): Duration in seconds for each bundle (overrides bundle_size if provided)
    step_time_seconds (float): Time step in seconds for sliding window (overrides step_size if provided)
    evenly_spaced_samples (int): If provided, select this many evenly spaced samples within each time window (overrides bundle_size)
    time_window_seconds (float): Time window in seconds to sample evenly spaced points from (required if evenly_spaced_samples is provided)
    smoothing_seconds (float): Window size in seconds for moving average smoothing (0 for no smoothing)
    feature_columns (list): Specific columns to include in bundles (None for all)
    max_files_per_batch (int): Maximum number of files to process at once
    max_bundles_per_file (int): Maximum number of bundles to extract per file (None for all)
    max_memory_mb (int): Maximum memory usage target in MB (approximate)
    chunk_size (int): Number of time windows to process at once for evenly spaced sampling
    output_dir (str): Directory to save bundles
    sampling_rate (int): Sampling rate in Hz (used when converting between samples and time for non-datetime indexed data)
    
    Returns:
    tuple: metadata_df (DataFrame with metadata for each bundle),
           bundle_info (dict with info about saved bundles)
    """
    # Ensure we have either file_dfs or combined_df
    if file_dfs is None and combined_df is None:
        raise ValueError("Either file_dfs or combined_df must be provided")
    
    # Validate evenly_spaced_samples and time_window_seconds parameters
    if evenly_spaced_samples is not None and time_window_seconds is None:
        raise ValueError("time_window_seconds must be provided when evenly_spaced_samples is specified")
    
    # If combined_df is provided but file_dfs is not, split it by file_id
    if file_dfs is None and combined_df is not None:
        if 'file_id' not in combined_df.columns:
            raise ValueError("combined_df must have a 'file_id' column")
        
        file_dfs = {file_id: group for file_id, group in combined_df.groupby('file_id')}
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize metadata storage
    all_metadata = []
    bundle_count = 0
    file_batch_count = 0
    
    # Process files in batches to save memory
    file_ids = list(file_dfs.keys())
    
    for batch_start in range(0, len(file_ids), max_files_per_batch):
        batch_end = min(batch_start + max_files_per_batch, len(file_ids))
        batch_file_ids = file_ids[batch_start:batch_end]
        
        print(f"\nProcessing files {batch_start+1}-{batch_end} of {len(file_ids)}")
        
        # Process this batch of files
        batch_bundles = []
        batch_metadata = []
        
        for file_id in batch_file_ids:
            print(f"  Processing file: {file_id}")
            df = file_dfs[file_id]  # We'll make a copy of only what we need
            
            if evenly_spaced_samples:
                # Evenly spaced samples method
                # Check if we need to set datetime index
                has_datetime_index = df.index.dtype.kind == 'M'  # M is for datetime
                
                if not has_datetime_index and 'TimeStamp' in df.columns:
                    working_df = df.set_index('TimeStamp')
                else:
                    working_df = df.copy()
                
                # Create a minimal working copy with just the columns we need
                if feature_columns is not None:
                    available_columns = [col for col in feature_columns if col in working_df.columns]
                    if len(available_columns) < len(feature_columns):
                        print(f"    Warning: Some requested features missing in {file_id}")
                    
                    if not available_columns:
                        print(f"    Skipping {file_id} - no requested features available")
                        continue
                        
                    working_df = working_df[available_columns].copy()
                else:
                    # Use all numeric columns except metadata
                    exclude_cols = ['file_id', 'TimeStamp', 'original_index']
                    numeric_cols = working_df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
                    available_columns = [col for col in numeric_cols if col not in exclude_cols]
                    working_df = working_df[available_columns].copy()
                
                # Apply smoothing if requested
                if smoothing_seconds > 0:
                    working_df = apply_moving_average(working_df, smoothing_seconds)
                
                # Calculate time range
                if has_datetime_index:
                    start_time = working_df.index.min()
                    end_time = working_df.index.max()
                    total_seconds = (end_time - start_time).total_seconds()
                else:
                    # Assume uniform sampling
                    total_seconds = len(working_df) / sampling_rate
                    
                # Calculate number of windows
                num_windows = max(1, int((total_seconds - time_window_seconds) / step_time_seconds) + 1)
                
                print(f"    Creating {num_windows} evenly spaced sample bundles from {file_id}")
                print(f"    Time range: {total_seconds:.1f} seconds, Window: {time_window_seconds}s, Step: {step_time_seconds}s")
                
                # Process in file batches to limit memory usage
                file_batch_count = 0
                file_bundle_count = 0
                batch_bundles = []
                batch_metadata = []
                
                for i in range(num_windows):
                    # Calculate start and end time for this window
                    window_start_second = i * step_time_seconds
                    window_end_second = window_start_second + time_window_seconds
                    
                    if has_datetime_index:
                        window_start_time = start_time + pd.Timedelta(seconds=window_start_second)
                        window_end_time = start_time + pd.Timedelta(seconds=window_end_second)
                        window_data = working_df.loc[window_start_time:window_end_time]
                    else:
                        window_start_idx = int(window_start_second * sampling_rate)
                        window_end_idx = min(len(working_df), int(window_end_second * sampling_rate))
                        window_data = working_df.iloc[window_start_idx:window_end_idx]
                    
                    # Skip if window is too small
                    if len(window_data) < evenly_spaced_samples:
                        continue
                    
                    # Sample the window data evenly
                    samples = get_evenly_spaced_samples(window_data, time_window_seconds, evenly_spaced_samples)

                    # Skip if we couldn't get the required samples
                    if samples is None or (isinstance(samples, tuple) and len(samples) < evenly_spaced_samples) or (hasattr(samples, 'shape') and samples.shape[0] < evenly_spaced_samples):
                        continue
                    
                    # Convert to numpy array (ensure we're working with the selected features only)
                    sample_array = samples.values
                    
                    # Print shape information for debugging
                    print(f"    Sample array shape: {sample_array.shape}, features: {sample_array.shape[1]}")
                    
                    # Create bundle (shape: time_steps x features)
                    bundle = sample_array
                    
                    # Add to batch
                    batch_bundles.append(bundle)
                    
                    # Create metadata for this bundle
                    metadata_item = {
                        'bundle_idx': bundle_count + len(batch_bundles) - 1,
                        'file_id': file_id,
                        'start_time': window_start_second,
                        'end_time': window_end_second,
                        'num_samples': len(bundle)
                    }
                    batch_metadata.append(metadata_item)
                    
                    file_bundle_count += 1
                    
                    # Save batch to disk if we've reached the batch size limit
                    if len(batch_bundles) >= chunk_size:
                        # Convert batch to 3D array (bundles x time_steps x features)
                        batch_array = np.array(batch_bundles)
                        
                        # Save to file
                        batch_filename = os.path.join(output_dir, f'batch_{file_batch_count}.npy')
                        np.save(batch_filename, batch_array)
                        file_batch_count += 1
            else:
                # Standard method: fixed number of consecutive samples
                # Create a minimal working copy with just the columns we need
                if feature_columns is not None:
                    available_columns = [col for col in feature_columns if col in df.columns]
                    if len(available_columns) < len(feature_columns):
                        print(f"    Warning: Some requested features missing in {file_id}")
                    
                    if not available_columns:
                        print(f"    Skipping {file_id} - no requested features available")
                        continue
                        
                    feature_df = df[available_columns].copy()
                else:
                    # Use all numeric columns except metadata
                    exclude_cols = ['file_id', 'TimeStamp', 'original_index']
                    numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
                    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
                    
                    if not numeric_cols:
                        print(f"    Skipping {file_id} - no numeric features available")
                        continue
                        
                    feature_df = df[numeric_cols].copy()
                
                # Apply smoothing if requested
                if smoothing_seconds > 0:
                    # Need timestamp for smoothing
                    if isinstance(df.index, pd.DatetimeIndex):
                        feature_df.index = df.index
                        feature_df = apply_moving_average(feature_df, window_seconds=smoothing_seconds)
                    elif 'TimeStamp' in df.columns:
                        # Set TimeStamp as index temporarily for smoothing
                        if pd.api.types.is_datetime64_any_dtype(df['TimeStamp']):
                            feature_df['TimeStamp'] = df['TimeStamp']
                            feature_df = feature_df.set_index('TimeStamp')
                            feature_df = apply_moving_average(feature_df, window_seconds=smoothing_seconds)
                            feature_df = feature_df.reset_index(drop=True)
                
                # Convert time-based parameters to sample counts if needed
                if bundle_time_seconds is not None:
                    calculated_bundle_size = calculate_samples_from_seconds(df, bundle_time_seconds)
                    if calculated_bundle_size is not None:
                        bundle_size = calculated_bundle_size
                        print(f"    Using bundle_size={bundle_size} samples for {bundle_time_seconds} seconds")
                    else:
                        print(f"    Cannot convert {bundle_time_seconds} seconds to samples, using default bundle_size={bundle_size}")
                
                if step_time_seconds is not None:
                    calculated_step_size = calculate_samples_from_seconds(df, step_time_seconds)
                    if calculated_step_size is not None:
                        step_size = calculated_step_size
                        print(f"    Using step_size={step_size} samples for {step_time_seconds} seconds")
                    else:
                        print(f"    Cannot convert {step_time_seconds} seconds to samples, using default step_size={step_size}")
                
                X = feature_df.values
                
                # Skip files that are too short
                if len(X) < bundle_size:
                    print(f"    Skipping {file_id} - too short ({len(X)} < {bundle_size})")
                    # Clean up to free memory
                    del feature_df
                    gc.collect()
                    continue
                
                # Calculate how many bundles to create
                total_possible = (len(X) - bundle_size) // step_size + 1
                bundles_to_create = total_possible
                if max_bundles_per_file is not None:
                    bundles_to_create = min(total_possible, max_bundles_per_file)
                    
                print(f"    Creating {bundles_to_create} bundles from {file_id}")
                
                # Process in chunks to limit memory usage
                file_bundle_count = 0
                
                for chunk_start in range(0, bundles_to_create, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, bundles_to_create)
                    
                    # Process this chunk of bundles
                    for bundle_idx in range(chunk_start, chunk_end):
                        # Calculate the start index in the original data
                        i = bundle_idx * step_size
                        
                        # Extract bundle
                        bundle = X[i:i+bundle_size]
                        batch_bundles.append(bundle)
                        
                        # Store metadata
                        batch_metadata.append({
                            'file_id': file_id,
                            'start_index': i,
                            'end_index': i + bundle_size - 1,
                            'bundle_index': bundle_count + len(batch_bundles) - 1,
                            'batch_index': file_batch_count,
                            'time_length_seconds': bundle_time_seconds if bundle_time_seconds is not None else None
                        })
                        
                        file_bundle_count += 1
                    
                    # Check if batch is large enough to save
                    if len(batch_bundles) >= 1000:
                        # Calculate approximate memory usage
                        mem_usage = len(batch_bundles) * batch_bundles[0].nbytes / (1024 * 1024)  # MB
                        
                        if mem_usage > max_memory_mb * 0.75:  # If using more than 75% of target
                            # Save current batch
                            batch_path = os.path.join(output_dir, f'bundle_batch_{file_batch_count}.npy')
                            np.save(batch_path, np.array(batch_bundles))
                            
                            # Update metadata and counters
                            all_metadata.extend(batch_metadata)
                            bundle_count += len(batch_bundles)
                            file_batch_count += 1
                            
                            # Reset batch storage
                            batch_bundles = []
                            batch_metadata = []
                            
                            # Force garbage collection
                            gc.collect()
                
                # Clean up to free memory
                del feature_df
                del X
                gc.collect()
            
            # If we've accumulated enough bundles, save to disk
            if len(batch_bundles) >= 1000:
                # Save current batch
                batch_path = os.path.join(output_dir, f'bundle_batch_{file_batch_count}.npy')
                np.save(batch_path, np.array(batch_bundles))
                
                # Update metadata and counters
                all_metadata.extend(batch_metadata)
                bundle_count += len(batch_bundles)
                file_batch_count += 1
                
                # Reset batch storage
                batch_bundles = []
                batch_metadata = []
                
                # Force garbage collection
                gc.collect()
        
        # Save any remaining bundles in this file batch
        if batch_bundles:
            batch_path = os.path.join(output_dir, f'bundle_batch_{file_batch_count}.npy')
            np.save(batch_path, np.array(batch_bundles))
            
            all_metadata.extend(batch_metadata)
            bundle_count += len(batch_bundles)
            file_batch_count += 1
            
            # Reset for next batch
            batch_bundles = []
            batch_metadata = []
            
            # Force garbage collection
            gc.collect()
    
    # Create metadata DataFrame
    metadata_df = pd.DataFrame(all_metadata)
    
    # Save metadata
    metadata_path = os.path.join(output_dir, 'bundle_metadata.csv')
    metadata_df.to_csv(metadata_path, index=False)
    
    # Create info dict
    bundle_info = {
        'output_dir': output_dir,
        'metadata_path': metadata_path,
        'total_bundles': bundle_count,
        'total_batches': file_batch_count,
        'bundle_size': bundle_size,
        'step_size': step_size,
        'bundle_time_seconds': bundle_time_seconds,
        'step_time_seconds': step_time_seconds,
        'evenly_spaced_samples': evenly_spaced_samples,
        'time_window_seconds': time_window_seconds,
        'smoothing_seconds': smoothing_seconds
    }
    
    print(f"\nCreated {bundle_count} bundles in {file_batch_count} batch files")
    print(f"Metadata saved to {metadata_path}")
    
    return metadata_df, bundle_info

def normalize_bundles_disk(bundle_info, normalization='per_feature', batch_size=100):
    """
    Normalize bundles that are stored on disk.
    
    Args:
        bundle_info: Information about the bundles on disk
        normalization: Type of normalization to apply
        batch_size: Number of bundles to process at once
        
    Returns:
        Updated bundle_info
    """
    logger.info(f"Normalizing bundles on disk using '{normalization}' method")
    
    # Create output directory for normalized bundles
    output_dir = os.path.join(bundle_info['output_dir'], 'normalized_bundles')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get total number of bundles
    total_bundles = bundle_info['total_bundles']
    logger.info(f"Total bundles to normalize: {total_bundles}")
    
    # Track normalization statistics
    stats = {
        'input_nans': 0,
        'output_nans': 0,
        'total_values': 0,
        'infinite_values': 0,
        'bundles_with_nans': 0,
        'feature_nan_counts': None,
        'min_before': float('inf'),
        'max_before': float('-inf'),
        'min_after': float('inf'),
        'max_after': float('-inf')
    }
    
    # Process in batches
    for start_idx in range(0, total_bundles, batch_size):
        end_idx = min(start_idx + batch_size, total_bundles)
        logger.debug(f"Processing bundles {start_idx} to {end_idx}")
        
        # Load batch of bundles
        batch_data = []
        for i in range(start_idx, end_idx):
            bundle_path = os.path.join(bundle_info['output_dir'], 'bundles', f"bundle_{i}.npy")
            try:
                bundle = np.load(bundle_path)
                batch_data.append(bundle)
                
                # Track input statistics
                nan_count = np.isnan(bundle).sum()
                if nan_count > 0:
                    stats['input_nans'] += nan_count
                    stats['bundles_with_nans'] += 1
                    logger.debug(f"Bundle {i} has {nan_count} NaN values before normalization")
                
                # Track min/max
                if not np.isnan(bundle).all():  # Only calculate if not all values are NaN
                    stats['min_before'] = min(stats['min_before'], np.nanmin(bundle))
                    stats['max_before'] = max(stats['max_before'], np.nanmax(bundle))
                
                # Track infinite values
                inf_count = np.isinf(bundle).sum()
                if inf_count > 0:
                    stats['infinite_values'] += inf_count
                    logger.warning(f"Bundle {i} has {inf_count} infinite values before normalization")
                
                # Update total values count
                stats['total_values'] += bundle.size
                
            except Exception as e:
                logger.error(f"Error loading bundle {i}: {str(e)}")
                continue
        
        if not batch_data:
            logger.warning(f"No valid bundles in batch {start_idx} to {end_idx}")
            continue
            
        # Stack all bundles into a batch
        X = np.stack(batch_data, axis=0)
        logger.debug(f"Loaded batch with shape {X.shape}")
        
        # Check for NaN values before normalization
        pre_nan_count = np.isnan(X).sum()
        if pre_nan_count > 0:
            logger.warning(f"Batch contains {pre_nan_count} NaN values before normalization")
            
            # Identify features with NaNs
            if X.ndim == 3:  # 3D tensor (samples, time, features)
                feature_nan_counts = np.isnan(X).sum(axis=(0, 1))
                if stats['feature_nan_counts'] is None:
                    stats['feature_nan_counts'] = feature_nan_counts
                else:
                    stats['feature_nan_counts'] += feature_nan_counts
                
                # Log features with high NaN counts
                for i, count in enumerate(feature_nan_counts):
                    if count > 0:
                        logger.debug(f"Feature {i} has {count} NaN values")
        
        # Apply normalization
        try:
            if normalization == 'per_bundle':
                # Normalize each bundle independently
                for i in range(X.shape[0]):
                    # Skip bundles with all NaN
                    if np.isnan(X[i]).all():
                        logger.warning(f"Bundle {start_idx + i} contains all NaN values, skipping normalization")
                        continue
                    
                    # For 3D tensors (bundles)
                    if X.ndim == 3:
                        # Normalize each feature across time steps
                        for j in range(X.shape[2]):
                            feature_data = X[i, :, j]
                            # Skip features with all NaN
                            if np.isnan(feature_data).all():
                                logger.debug(f"Bundle {start_idx + i}, feature {j} contains all NaN values")
                                continue
                            
                            mean = np.nanmean(feature_data)
                            std = np.nanstd(feature_data)
                            
                            # Check for zero standard deviation (constant feature)
                            if std == 0:
                                logger.debug(f"Bundle {start_idx + i}, feature {j} has zero standard deviation")
                                # Set to zero instead of dividing by zero
                                X[i, :, j] = 0
                            else:
                                X[i, :, j] = (feature_data - mean) / std
                    
                    # For 2D tensors (regular samples)
                    else:
                        for j in range(X.shape[1]):
                            feature_data = X[i, j]
                            if np.isnan(feature_data):
                                continue
                            
                            mean = np.nanmean(X[i])
                            std = np.nanstd(X[i])
                            
                            if std == 0:
                                X[i] = 0
                            else:
                                X[i] = (X[i] - mean) / std
            
            elif normalization == 'per_feature':
                # For 3D tensors (bundles)
                if X.ndim == 3:
                    # Normalize each feature across all bundles and time steps
                    for j in range(X.shape[2]):
                        feature_data = X[:, :, j]
                        # Skip features with all NaN
                        if np.isnan(feature_data).all():
                            logger.warning(f"Feature {j} contains all NaN values across all bundles")
                            continue
                        
                        mean = np.nanmean(feature_data)
                        std = np.nanstd(feature_data)
                        
                        # Log means and stds for debugging
                        logger.debug(f"Feature {j} - mean: {mean}, std: {std}")
                        
                        # Check for zero standard deviation (constant feature)
                        if std == 0 or np.isnan(std):
                            logger.warning(f"Feature {j} has {'zero' if std == 0 else 'NaN'} standard deviation")
                            # Set to zero instead of dividing by zero
                            X[:, :, j] = 0
                        else:
                            # Perform normalization with explicit NaN handling
                            normalized = (feature_data - mean) / std
                            X[:, :, j] = normalized
                            
                            # Check for NaNs introduced by normalization
                            new_nans = np.isnan(normalized).sum()
                            if new_nans > 0:
                                logger.warning(f"Normalization introduced {new_nans} NaN values in feature {j}")
                
                # For 2D tensors (regular samples)
                else:
                    for j in range(X.shape[1]):
                        feature_data = X[:, j]
                        # Skip features with all NaN
                        if np.isnan(feature_data).all():
                            logger.warning(f"Feature {j} contains all NaN values across all samples")
                            continue
                        
                        mean = np.nanmean(feature_data)
                        std = np.nanstd(feature_data)
                        
                        if std == 0 or np.isnan(std):
                            logger.warning(f"Feature {j} has {'zero' if std == 0 else 'NaN'} standard deviation")
                            X[:, j] = 0
                        else:
                            normalized = (feature_data - mean) / std
                            X[:, j] = normalized
                            
                            # Check for NaNs
                            new_nans = np.isnan(normalized).sum()
                            if new_nans > 0:
                                logger.warning(f"Normalization introduced {new_nans} NaN values in feature {j}")
                                
            elif normalization == 'global':
                # Skip bundles with all NaN
                if np.isnan(X).all():
                    logger.warning(f"All data contains NaN values, skipping global normalization")
                else:
                    # Normalize all data together
                    mean = np.nanmean(X)
                    std = np.nanstd(X)
                    
                    if std == 0 or np.isnan(std):
                        logger.warning(f"Data has {'zero' if std == 0 else 'NaN'} global standard deviation")
                        X = np.zeros_like(X)
                    else:
                        X = (X - mean) / std
            
            else:
                logger.error(f"Unknown normalization method: {normalization}")
                return bundle_info
                
        except Exception as e:
            logger.error(f"Error during normalization: {str(e)}")
            logger.error(traceback.format_exc())
            continue
        
        # Check for NaN values after normalization
        post_nan_count = np.isnan(X).sum()
        stats['output_nans'] += post_nan_count
        
        if post_nan_count > 0:
            logger.warning(f"Batch contains {post_nan_count} NaN values after normalization")
            
            # Calculate NaNs introduced by normalization
            nans_introduced = post_nan_count - pre_nan_count
            if nans_introduced > 0:
                logger.warning(f"Normalization introduced {nans_introduced} new NaN values")
                
            # Try to fix NaNs from normalization
            logger.info("Attempting to fix NaN values...")
            X = np.nan_to_num(X, nan=0.0)
            
            # Verify fix
            remaining_nans = np.isnan(X).sum()
            if remaining_nans > 0:
                logger.error(f"Still have {remaining_nans} NaN values after attempted fix")
            else:
                logger.info("Successfully fixed NaN values")
        
        # Check for infinite values
        inf_count = np.isinf(X).sum()
        if inf_count > 0:
            logger.warning(f"Batch contains {inf_count} infinite values after normalization")
            logger.info("Replacing infinite values with 0...")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Track min/max after normalization
        if not np.isnan(X).all():  # Only calculate if not all values are NaN
            stats['min_after'] = min(stats['min_after'], np.nanmin(X))
            stats['max_after'] = max(stats['max_after'], np.nanmax(X))
        
        # Save normalized bundles
        for i in range(X.shape[0]):
            bundle_idx = start_idx + i
            if bundle_idx >= total_bundles:
                break
                
            output_path = os.path.join(output_dir, f"bundle_{bundle_idx}.npy")
            try:
                # Check this specific bundle for NaNs
                bundle_nan_count = np.isnan(X[i]).sum()
                if bundle_nan_count > 0:
                    logger.warning(f"Bundle {bundle_idx} has {bundle_nan_count} NaN values before saving")
                    # Fix NaNs before saving
                    X[i] = np.nan_to_num(X[i], nan=0.0, posinf=0.0, neginf=0.0)
                
                np.save(output_path, X[i])
                
            except Exception as e:
                logger.error(f"Error saving normalized bundle {bundle_idx}: {str(e)}")
        
        # Log progress
        if (end_idx) % 500 == 0 or end_idx == total_bundles:
            logger.info(f"Normalized {end_idx}/{total_bundles} bundles")
    
    # Log final statistics
    logger.info("=== Normalization Statistics ===")
    logger.info(f"Total bundles processed: {total_bundles}")
    logger.info(f"Total values: {stats['total_values']}")
    logger.info(f"Input NaN values: {stats['input_nans']} ({stats['input_nans']/stats['total_values']*100:.4f}%)")
    logger.info(f"Output NaN values: {stats['output_nans']} ({stats['output_nans']/stats['total_values']*100:.4f}% - before final cleaning)")
    logger.info(f"Bundles with NaNs: {stats['bundles_with_nans']} ({stats['bundles_with_nans']/total_bundles*100:.2f}%)")
    logger.info(f"Infinite values: {stats['infinite_values']}")
    logger.info(f"Value range before: [{stats['min_before']}, {stats['max_before']}]")
    logger.info(f"Value range after: [{stats['min_after']}, {stats['max_after']}]")
    
    if stats['feature_nan_counts'] is not None:
        # Print top 10 features with most NaNs
        nan_features = [(i, count) for i, count in enumerate(stats['feature_nan_counts'])]
        nan_features.sort(key=lambda x: x[1], reverse=True)
        
        if nan_features[0][1] > 0:  # If there are any features with NaNs
            logger.info("Top 10 features with most NaNs:")
            for i, (feature_idx, count) in enumerate(nan_features[:10]):
                if count > 0:
                    logger.info(f"  Feature {feature_idx}: {count} NaNs")
    
    # Update bundle_info
    bundle_info['normalized'] = True
    bundle_info['normalization_method'] = normalization
    bundle_info['normalization_stats'] = stats
    
    # Create a verification function to check random bundles
    def verify_normalized_bundles(n_samples=10):
        """Verify a random sample of normalized bundles for quality control"""
        indices = np.random.choice(total_bundles, min(n_samples, total_bundles), replace=False)
        logger.info(f"Verifying {len(indices)} random normalized bundles")
        
        issues_found = 0
        for idx in indices:
            bundle_path = os.path.join(output_dir, f"bundle_{idx}.npy")
            try:
                bundle = np.load(bundle_path)
                nan_count = np.isnan(bundle).sum()
                inf_count = np.isinf(bundle).sum()
                
                if nan_count > 0 or inf_count > 0:
                    logger.warning(f"Verification: Bundle {idx} has {nan_count} NaNs and {inf_count} infinite values")
                    issues_found += 1
                else:
                    min_val = bundle.min()
                    max_val = bundle.max()
                    mean_val = bundle.mean()
                    std_val = bundle.std()
                    logger.debug(f"Bundle {idx} stats - min: {min_val}, max: {max_val}, mean: {mean_val}, std: {std_val}")
                    
                    # Check for extreme values that might indicate poor normalization
                    if abs(min_val) > 10 or abs(max_val) > 10:
                        logger.warning(f"Bundle {idx} has extreme values: min={min_val}, max={max_val}")
                        issues_found += 1
                
            except Exception as e:
                logger.error(f"Error verifying bundle {idx}: {str(e)}")
                issues_found += 1
        
        if issues_found > 0:
            logger.warning(f"Found issues in {issues_found}/{len(indices)} verified bundles")
        else:
            logger.info("All verified bundles look good")
            
        return issues_found == 0
    
    # Run verification
    verification_ok = verify_normalized_bundles(n_samples=min(50, total_bundles))
    bundle_info['verification_ok'] = verification_ok
    
    logger.info(f"Normalization completed {'successfully' if verification_ok else 'with issues'}")
    
    return bundle_info

def split_bundles_disk(metadata_df, test_ratio=0.2, by_file=True, random_state=42):
    """
    Split bundles stored on disk into training and testing sets
    
    Parameters:
    metadata_df (DataFrame): Metadata DataFrame from create_coherent_time_series_bundles_disk
    test_ratio (float): Ratio of data to use for testing
    by_file (bool): Whether to split by file to avoid data leakage
    random_state (int): Random seed for reproducibility
    
    Returns:
    tuple: (train_indices, test_indices)
    """
    np.random.seed(random_state)
    
    if by_file:
        # Get unique files
        unique_files = metadata_df['file_id'].unique()
        np.random.shuffle(unique_files)
        
        # Split files
        split_idx = int(len(unique_files) * (1 - test_ratio))
        train_files = unique_files[:split_idx]
        test_files = unique_files[split_idx:]
        
        # Get indices
        train_indices = metadata_df[metadata_df['file_id'].isin(train_files)]['bundle_index'].values
        test_indices = metadata_df[metadata_df['file_id'].isin(test_files)]['bundle_index'].values
    else:
        # Get all bundle indices
        all_indices = metadata_df['bundle_index'].values
        np.random.shuffle(all_indices)
        
        # Split indices
        split_idx = int(len(all_indices) * (1 - test_ratio))
        train_indices = all_indices[:split_idx]
        test_indices = all_indices[split_idx:]
    
    print(f"Train set: {len(train_indices)} bundles")
    print(f"Test set: {len(test_indices)} bundles")
    
    return train_indices, test_indices

class DiskBundleLoader:
    """
    Iterator for loading bundles from disk in batches
    """
    def __init__(self, bundle_info, indices=None, batch_size=32, shuffle=True, use_normalized=True):
        """
        Initialize the loader
        
        Parameters:
        bundle_info (dict): Bundle info from normalize_bundles_disk
        indices (ndarray): Indices of bundles to load (None for all)
        batch_size (int): Number of bundles to load at once
        shuffle (bool): Whether to shuffle the indices
        use_normalized (bool): Whether to use normalized bundles
        """
        self.bundle_info = bundle_info
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Determine directory
        if use_normalized and 'normalized_dir' in bundle_info:
            self.data_dir = bundle_info['normalized_dir']
        else:
            self.data_dir = bundle_info['output_dir']
        
        # Load metadata
        metadata_path = bundle_info['metadata_path']
        self.metadata = pd.read_csv(metadata_path)
        
        # Set indices
        if indices is None:
            self.indices = np.arange(bundle_info['total_bundles'])
        else:
            self.indices = np.array(indices)
            
            # Ensure indices are within range
            valid_indices = self.indices[self.indices < len(self.metadata)]
            if len(valid_indices) < len(self.indices):
                print(f"Warning: {len(self.indices) - len(valid_indices)} indices were out of range.")
                self.indices = valid_indices
        
        # Create mapping from bundle indices to (batch_file, position_in_batch)
        self.bundle_to_batch_map = {}
        
        # Group metadata by batch_index to find positions within batches
        batch_groups = self.metadata.groupby('batch_index')
        
        for batch_idx, group in batch_groups:
            # Sort the group by bundle_index to ensure correct position mapping
            group = group.sort_values('bundle_index')
            
            # Map each bundle_index to its position within this batch
            for i, (_, row) in enumerate(group.iterrows()):
                self.bundle_to_batch_map[row['bundle_index']] = (batch_idx, i)
        
        # Check if all requested indices have valid mappings
        for idx in self.indices:
            if idx not in self.bundle_to_batch_map:
                print(f"Warning: Bundle index {idx} not found in metadata. It will be skipped.")
        
        # Filter out indices that don't have mappings
        self.indices = np.array([idx for idx in self.indices if idx in self.bundle_to_batch_map])
        
        # Track batch cache
        self.batch_cache = {}
        self.max_cache_size = 3  # Number of batch files to keep in memory
    
    def __iter__(self):
        """
        Return iterator
        """
        # Shuffle indices if requested
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        # Create batches
        self.batch_indices = [
            self.indices[i:i+self.batch_size]
            for i in range(0, len(self.indices), self.batch_size)
        ]
        
        self.batch_idx = 0
        return self
    
    def __next__(self):
        """
        Get next batch
        """
        if self.batch_idx >= len(self.batch_indices):
            # Clear cache
            self.batch_cache = {}
            
            # End iteration
            raise StopIteration
        
        # Get indices for this batch
        batch_indices = self.batch_indices[self.batch_idx]
        self.batch_idx += 1
        
        # Group indices by batch file
        batch_groups = {}
        for bundle_idx in batch_indices:
            # Look up the batch file and position
            batch_file_idx, position_in_batch = self.bundle_to_batch_map[bundle_idx]
            
            if batch_file_idx not in batch_groups:
                batch_groups[batch_file_idx] = []
            batch_groups[batch_file_idx].append((bundle_idx, position_in_batch))
        
        # Load bundles
        bundle_list = []
        
        for batch_file_idx, bundle_positions in batch_groups.items():
            # Load batch file if not in cache
            if batch_file_idx not in self.batch_cache:
                # If cache is full, remove least recently used
                if len(self.batch_cache) >= self.max_cache_size:
                    # Simple LRU: remove first key
                    del self.batch_cache[next(iter(self.batch_cache))]
                
                # Load batch file
                try:
                    batch_path = os.path.join(self.data_dir, f'bundle_batch_{batch_file_idx}.npy')
                    self.batch_cache[batch_file_idx] = np.load(batch_path)
                    print(f"Loaded batch file {batch_file_idx} with shape {self.batch_cache[batch_file_idx].shape}")
                except Exception as e:
                    print(f"Error loading batch file {batch_file_idx}: {e}")
                    # Try fallback to chunked files
                    try:
                        chunks_dir = os.path.join(self.data_dir, f'chunks_{batch_file_idx}')
                        if os.path.exists(chunks_dir):
                            print(f"Trying to load from chunks in {chunks_dir}")
                            # Read metadata
                            meta_path = os.path.join(self.data_dir, f'chunks_{batch_file_idx}_meta.txt')
                            with open(meta_path, 'r') as f:
                                meta_lines = f.readlines()
                            
                            # Parse metadata
                            batch_shape = eval(meta_lines[0].split(': ')[1])
                            chunk_size = int(meta_lines[1].split(': ')[1])
                            
                            # Allocate array
                            combined_batch = np.zeros(batch_shape, dtype=np.float32)
                            
                            # Load chunks
                            for i in range(0, batch_shape[0], chunk_size):
                                chunk_path = os.path.join(chunks_dir, f'chunk_{i}.npy')
                                if os.path.exists(chunk_path):
                                    chunk = np.load(chunk_path)
                                    end_idx = min(i + chunk_size, batch_shape[0])
                                    combined_batch[i:end_idx] = chunk
                            
                            self.batch_cache[batch_file_idx] = combined_batch
                            print(f"Successfully loaded from chunks with shape {combined_batch.shape}")
                        else:
                            print(f"No chunks directory found for batch {batch_file_idx}")
                            continue
                    except Exception as e2:
                        print(f"Failed to load chunks for batch {batch_file_idx}: {e2}")
                        continue
            
            # Get bundles from this batch
            for bundle_idx, position in bundle_positions:
                try:
                    # Check if position is valid
                    if position >= len(self.batch_cache[batch_file_idx]):
                        print(f"Warning: Position {position} is out of bounds for batch {batch_file_idx} with size {len(self.batch_cache[batch_file_idx])}")
                        continue
                    
                    # Get the bundle
                    bundle = self.batch_cache[batch_file_idx][position]
                    bundle_list.append(bundle)
                except Exception as e:
                    print(f"Error getting bundle {bundle_idx} at position {position} from batch {batch_file_idx}: {e}")
                    continue
        
        # Convert to array
        if not bundle_list:
            print(f"Warning: No valid bundles found for batch {self.batch_idx-1}")
            # Return an empty array with the right shape if we know it
            if self.batch_cache and len(self.batch_cache) > 0:
                first_batch = next(iter(self.batch_cache.values()))
                if len(first_batch) > 0:
                    # Use the shape of the first bundle in the first batch
                    empty_shape = list(first_batch[0].shape)
                    empty_shape.insert(0, 0)  # Add a 0 dimension at the start
                    return np.zeros(empty_shape)
            
            # If we don't know the shape, recursively get the next batch
            return next(self)
        
        return np.array(bundle_list)
    
    def __len__(self):
        """
        Return number of batches
        """
        return (len(self.indices) + self.batch_size - 1) // self.batch_size
        
def load_eeg_data(directory_path=None, csv_files=None, single_file=None):
    """
    Load EEG data from CSV files, maintaining separate sessions
    
    Parameters:
    directory_path (str): Path to directory containing CSV files
    csv_files (list): List of CSV filenames to load
    single_file (str): Path to a single CSV file
    
    Returns:
    dict: Dictionary with DataFrames for each file
    DataFrame: Combined DataFrame with file identifiers
    """
    # Validate inputs
    if directory_path is None and csv_files is None and single_file is None:
        raise ValueError("Either directory_path, csv_files, or single_file must be provided")
    
    # Initialize dictionary to store DataFrames
    file_dfs = {}
    
    if single_file:
        # Load a single file
        df = pd.read_csv(single_file, low_memory=False)
        file_name = os.path.basename(single_file)
        df['file_id'] = file_name
        file_dfs[file_name] = df
        
    elif directory_path:
        # Get list of CSV files if not provided
        if csv_files is None:
            csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
        
        # Load each file
        for file in csv_files:
            file_path = os.path.join(directory_path, file)
            df = pd.read_csv(file_path, low_memory=False)
            df['file_id'] = file
            file_dfs[file] = df
    
    # Create combined DataFrame (for convenience, not for actual processing)
    combined_df = pd.concat(file_dfs.values(), ignore_index=True)
    
    print(f"Loaded {len(file_dfs)} files with {len(combined_df)} total rows")
    
    return file_dfs, combined_df

def preprocess_eeg_data(df):
    """
    Preprocess EEG data with robust timestamp handling - drops rows with invalid timestamps
    
    Parameters:
    df (DataFrame): DataFrame containing EEG data
    
    Returns:
    DataFrame: Preprocessed DataFrame with bad timestamps removed
    """
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Handle TimeStamp column if it exists
    if 'TimeStamp' in df_clean.columns:
        try:
            # First try simple conversion
            df_clean['TimeStamp'] = pd.to_datetime(df_clean['TimeStamp'], errors='coerce')
            
            # Drop rows where conversion to datetime failed (NaT values)
            bad_timestamps = df_clean['TimeStamp'].isna()
            if bad_timestamps.any():
                print(f"Warning: Dropping {bad_timestamps.sum()} rows with invalid timestamps")
                df_clean = df_clean.dropna(subset=['TimeStamp'])
            
            # Add original index for reference
            df_clean['original_index'] = np.arange(len(df_clean))
            
            # Set TimeStamp as index
            df_clean = df_clean.set_index('TimeStamp')
            
            print(f"Successfully converted timestamps. Remaining rows: {len(df_clean)}")
            
        except Exception as e:
            print(f"Error during timestamp conversion: {str(e)}. Will drop problematic rows.")
            try:
                # Try explicit format for your specific timestamp pattern
                df_clean['TimeStamp'] = pd.to_datetime(df_clean['TimeStamp'], 
                                                     format='%Y-%m-%d %H:%M:%S.%f', 
                                                     errors='coerce')
                
                # Drop rows where conversion failed
                bad_timestamps = df_clean['TimeStamp'].isna()
                if bad_timestamps.any():
                    print(f"Warning: Dropping {bad_timestamps.sum()} rows with invalid timestamps")
                    df_clean = df_clean.dropna(subset=['TimeStamp'])
                
                # Add original index for reference
                df_clean['original_index'] = np.arange(len(df_clean))
                
                # Set TimeStamp as index
                df_clean = df_clean.set_index('TimeStamp')
                
                print(f"Successfully converted timestamps with explicit format. Remaining rows: {len(df_clean)}")
                
            except Exception as e2:
                print(f"All timestamp conversion attempts failed: {str(e2)}. Will proceed without timestamp conversion.")
                # Just keep original index without timestamp conversion
                df_clean['original_index'] = np.arange(len(df_clean))
    
    # Drop the 'Elements' column if it exists
    if 'Elements' in df_clean.columns:
        df_clean.drop(columns=['Elements'], inplace=True)
    
    # Calculate the percentage of missing values in each column
    missing_percentage = df_clean.isnull().sum() / len(df_clean) * 100
    
    # Handle columns with less than 10% missing values
    columns_to_drop_na = missing_percentage[missing_percentage < 10].index
    df_clean = df_clean.dropna(subset=columns_to_drop_na)
    
    # Handle columns with more than 10% missing values
    columns_to_fill_median = missing_percentage[missing_percentage >= 10].index
    for column in columns_to_fill_median:
        if column in df_clean.columns:  # Check if column still exists after previous operations
            median_value = df_clean[column].median()
            df_clean[column].fillna(median_value, inplace=True)
    
    # Ensure no NaNs or Infs are present
    df_clean = df_clean.replace([float('inf'), float('-inf')], np.nan)
    df_clean = df_clean.dropna()
    
    return df_clean

def engineer_eeg_features(df):
    """
    Generate advanced features from EEG data for relaxation/focus detection
    
    Parameters:
    df (DataFrame): Dataframe containing EEG data with frequency bands
    
    Returns:
    DataFrame: Original dataframe with additional engineered features
    """
    # Make a copy to avoid modifying the original
    df_features = df.copy()
    
    # 1. Frequency band ratios (particularly useful for relaxation/focus)
    channels = ['TP9', 'AF7', 'AF8', 'TP10']
    
    for channel in channels:
        # Check if the required columns exist
        alpha_col = f'Alpha_{channel}'
        beta_col = f'Beta_{channel}'
        theta_col = f'Theta_{channel}'
        delta_col = f'Delta_{channel}'
        
        # Only calculate if columns exist
        if all(col in df.columns for col in [alpha_col, beta_col]):
            # Alpha/Beta ratio (higher in relaxed states)
            df_features[f'AlphaBeta_Ratio_{channel}'] = df[alpha_col] / df[beta_col]
            
        if all(col in df.columns for col in [theta_col, beta_col]):
            # Theta/Beta ratio (higher in relaxed states)
            df_features[f'ThetaBeta_Ratio_{channel}'] = df[theta_col] / df[beta_col]
            
        if all(col in df.columns for col in [alpha_col, theta_col, beta_col]):
            # (Alpha + Theta)/Beta ratio (useful for relaxation detection)
            df_features[f'RelaxationIndex_{channel}'] = (df[alpha_col] + df[theta_col]) / df[beta_col]
            
            # Beta/(Alpha + Theta) ratio (useful for focus detection)
            df_features[f'FocusIndex_{channel}'] = df[beta_col] / (df[alpha_col] + df[theta_col])
            
        if all(col in df.columns for col in [delta_col, beta_col]):
            # Delta/Beta ratio (attention marker)
            df_features[f'DeltaBeta_Ratio_{channel}'] = df[delta_col] / df[beta_col]
    
    # 2. Frontal asymmetry (emotional and attentional marker)
    if all(col in df.columns for col in ['Alpha_AF8', 'Alpha_AF7']):
        df_features['Frontal_Alpha_Asymmetry'] = (df['Alpha_AF8'] - df['Alpha_AF7']) / (df['Alpha_AF8'] + df['Alpha_AF7'])
    
    if all(col in df.columns for col in ['Beta_AF8', 'Beta_AF7']):
        df_features['Frontal_Beta_Asymmetry'] = (df['Beta_AF8'] - df['Beta_AF7']) / (df['Beta_AF8'] + df['Beta_AF7'])
    
    if all(col in df.columns for col in ['Theta_AF8', 'Theta_AF7']):
        df_features['Frontal_Theta_Asymmetry'] = (df['Theta_AF8'] - df['Theta_AF7']) / (df['Theta_AF8'] + df['Theta_AF7'])
    
    # 3. Power normalization (account for individual differences)
    for channel in channels:
        band_cols = [f'{band}_{channel}' for band in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']]
        
        # Only proceed if all band columns exist
        if all(col in df.columns for col in band_cols):
            total_power = sum(df[col] for col in band_cols)
            
            for band in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']:
                col = f'{band}_{channel}'
                df_features[f'{band}_Norm_{channel}'] = df[col] / total_power
    
    # 4. Movement features
    if all(col in df.columns for col in ['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z']):
        # Calculate movement intensity
        df_features['Movement_Intensity'] = np.sqrt(
            df['Accelerometer_X']**2 + 
            df['Accelerometer_Y']**2 + 
            df['Accelerometer_Z']**2
        )
    
    # Fill NaN values that might have been introduced
    df_features = df_features.ffill().bfill()
    
    return df_features

def select_features(X, y, method='anova', k=20):
    """
    Select the k best features based on the specified method
    
    Parameters:
    X (DataFrame): Feature matrix
    y (Series): Target variable
    method (str): Method to use ('anova' or 'mutual_info')
    k (int): Number of features to select
    
    Returns:
    DataFrame: Selected features
    list: Names of selected features
    """
    if method == 'anova':
        selector = SelectKBest(f_classif, k=k)
    elif method == 'mutual_info':
        selector = SelectKBest(mutual_info_classif, k=k)
    else:
        raise ValueError("Method must be 'anova' or 'mutual_info'")
    
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    return X[selected_features], selected_features

def create_coherent_time_series_bundles(file_dfs=None, combined_df=None, 
                                      bundle_size=30, step_size=15,
                                      bundle_time_seconds=None, step_time_seconds=None,
                                      smoothing_seconds=0, feature_columns=None):
    """
    Create time series bundles ensuring each bundle only contains data from a single file
    
    Parameters:
    file_dfs (dict): Dictionary with DataFrames for each file
    combined_df (DataFrame): Combined DataFrame with file identifiers
    bundle_size (int): Number of consecutive rows to include in each bundle (ignored if bundle_time_seconds is provided)
    step_size (int): Step size for sliding window when creating bundles (ignored if step_time_seconds is provided)
    bundle_time_seconds (float): Duration in seconds for each bundle (overrides bundle_size if provided)
    step_time_seconds (float): Time step in seconds for sliding window (overrides step_size if provided)
    smoothing_seconds (float): Window size in seconds for moving average smoothing (0 for no smoothing)
    feature_columns (list): Specific columns to include in bundles (None for all)
    
    Returns:
    tuple: X_bundles (np.array of shape [n_bundles, bundle_size, n_features]),
           metadata (DataFrame with file_id and start_index for each bundle)
    """
    # Ensure we have either file_dfs or combined_df
    if file_dfs is None and combined_df is None:
        raise ValueError("Either file_dfs or combined_df must be provided")
    
    # If combined_df is provided but file_dfs is not, split it by file_id
    if file_dfs is None and combined_df is not None:
        if 'file_id' not in combined_df.columns:
            raise ValueError("combined_df must have a 'file_id' column")
        
        file_dfs = {file_id: group for file_id, group in combined_df.groupby('file_id')}
    
    # Initialize lists to store bundles and metadata
    all_bundles = []
    bundle_metadata = []
    
    # Process each file separately
    for file_id, df in file_dfs.items():
        print(f"Processing bundles for file: {file_id}")
        
        # Apply smoothing if requested
        if smoothing_seconds > 0:
            df = apply_moving_average(df, window_seconds=smoothing_seconds)
        
        # Convert time-based parameters to sample counts if provided
        if bundle_time_seconds is not None:
            calculated_bundle_size = calculate_samples_from_seconds(df, bundle_time_seconds)
            if calculated_bundle_size is not None:
                bundle_size = calculated_bundle_size
                print(f"  Using bundle_size={bundle_size} samples for {bundle_time_seconds} seconds")
            else:
                print(f"  Cannot convert {bundle_time_seconds} seconds to samples, using default bundle_size={bundle_size}")
        
        if step_time_seconds is not None:
            calculated_step_size = calculate_samples_from_seconds(df, step_time_seconds)
            if calculated_step_size is not None:
                step_size = calculated_step_size
                print(f"  Using step_size={step_size} samples for {step_time_seconds} seconds")
            else:
                print(f"  Cannot convert {step_time_seconds} seconds to samples, using default step_size={step_size}")
        
        # Select feature columns if specified
        if feature_columns is not None:
            # Check which specified columns exist in this file
            available_columns = [col for col in feature_columns if col in df.columns]
            if len(available_columns) < len(feature_columns):
                missing = set(feature_columns) - set(available_columns)
                print(f"  Warning: {len(missing)} requested columns missing in {file_id}")
            
            # Only keep available requested columns
            if not available_columns:
                print(f"  Skipping {file_id} - no requested features available")
                continue
                
            X = df[available_columns].values
        else:
            # Use all numeric columns except file_id and timestamp related
            exclude_cols = ['file_id', 'TimeStamp', 'original_index']
            numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            if not numeric_cols:
                print(f"    Skipping {file_id} - no numeric features available")
                continue
                
            X = df[numeric_cols].values
        
        # Skip files that are too short
        if len(X) < bundle_size:
            print(f"  Skipping {file_id} - too short ({len(X)} < {bundle_size})")
            continue
        
        # Create bundles using sliding window
        bundle_count = 0
        for i in range(0, len(X) - bundle_size + 1, step_size):
            # Extract the bundle
            bundle = X[i:i+bundle_size]
            
            # Store the bundle
            all_bundles.append(bundle)
            
            # Store metadata about this bundle
            bundle_metadata.append({
                'file_id': file_id,
                'start_index': i,
                'end_index': i + bundle_size - 1,
                'time_length_seconds': bundle_time_seconds if bundle_time_seconds is not None else None
            })
            
            bundle_count += 1
        
        print(f"  Created {bundle_count} bundles from {file_id}")
    
    # Convert list of bundles to numpy array
    if not all_bundles:
        raise ValueError("No bundles were created. Check your data and parameters.")
        
    X_bundles = np.array(all_bundles)
    
    # Convert metadata to DataFrame
    metadata_df = pd.DataFrame(bundle_metadata)
    
    print(f"Created {len(all_bundles)} total bundles of size {bundle_size} from {len(file_dfs)} files")
    print(f"Bundle shape: {X_bundles.shape}")
    
    return X_bundles, metadata_df

def normalize_bundles(X_bundles, normalization='per_feature'):
    """
    Normalize time series bundles
    
    Parameters:
    X_bundles (ndarray): Array of time series bundles [n_bundles, bundle_size, n_features]
    normalization (str): Normalization strategy:
                         'global': Normalize all features across all bundles
                         'per_bundle': Normalize each bundle independently
                         'per_feature': Normalize each feature independently across all bundles
                         'none': No normalization
    
    Returns:
    ndarray: Normalized bundles
    """
    if normalization == 'none':
        return X_bundles
        
    n_bundles, bundle_size, n_features = X_bundles.shape
    X_normalized = np.zeros_like(X_bundles)
    
    if normalization == 'global':
        # Reshape to 2D: (n_bundles * bundle_size, n_features)
        X_reshaped = X_bundles.reshape(-1, n_features)
        
        # Normalize all data together
        scaler = StandardScaler()
        X_normalized_flat = scaler.fit_transform(X_reshaped)
        
        # Reshape back to 3D
        X_normalized = X_normalized_flat.reshape(n_bundles, bundle_size, n_features)
        
    elif normalization == 'per_bundle':
        # Normalize each bundle independently
        for i in range(n_bundles):
            scaler = StandardScaler()
            X_normalized[i] = scaler.fit_transform(X_bundles[i])
            
    elif normalization == 'per_feature':
        # Normalize each feature independently
        for j in range(n_features):
            # Extract this feature across all bundles and all time steps
            feature_data = X_bundles[:, :, j].reshape(-1, 1)
            
            # Normalize
            scaler = StandardScaler()
            normalized_feature = scaler.fit_transform(feature_data)
            
            # Put back into the result array
            X_normalized[:, :, j] = normalized_feature.reshape(n_bundles, bundle_size)
            
    else:
        raise ValueError("normalization must be 'global', 'per_bundle', 'per_feature', or 'none'")
    
    return X_normalized

def split_bundles_train_test(X_bundles, metadata_df, test_ratio=0.2, by_file=True, labels=None):
    """
    Split time series bundles into training and testing sets
    
    Parameters:
    X_bundles (ndarray): Array of time series bundles
    metadata_df (DataFrame): Metadata for each bundle
    test_ratio (float): Ratio of data to use for testing
    by_file (bool): If True, split by files to avoid data leakage
    labels (ndarray): Optional labels to split along with the bundles
    
    Returns:
    tuple: X_train, X_test, [y_train, y_test if labels provided], train_metadata, test_metadata
    """
    if by_file:
        # Get unique files
        unique_files = metadata_df['file_id'].unique()
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(unique_files)
        
        # Determine split point
        split_idx = int(len(unique_files) * (1 - test_ratio))
        train_files = unique_files[:split_idx]
        test_files = unique_files[split_idx:]
        
        # Create masks
        train_mask = metadata_df['file_id'].isin(train_files)
        test_mask = ~train_mask
        
    else:
        # Get all bundle indices
        all_indices = metadata_df['bundle_index'].values
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(all_indices)
        
        # Split indices
        split_idx = int(len(all_indices) * (1 - test_ratio))
        train_indices = all_indices[:split_idx]
        test_indices = all_indices[split_idx:]
    
    train_mask = np.zeros(len(X_bundles), dtype=bool)
    train_mask[train_indices] = True
    test_mask = ~train_mask
    
    # Apply the masks
    X_train = X_bundles[train_mask]
    X_test = X_bundles[test_mask]
    train_metadata = metadata_df[train_mask].reset_index(drop=True)
    test_metadata = metadata_df[test_mask].reset_index(drop=True)
    
    print(f"Training set: {len(X_train)} bundles from {len(train_metadata['file_id'].unique())} files")
    print(f"Testing set: {len(X_test)} bundles from {len(test_metadata['file_id'].unique())} files")
    
    if labels is not None:
        y_train = labels[train_mask]
        y_test = labels[test_mask]
        return X_train, X_test, y_train, y_test, train_metadata, test_metadata
    else:
        return X_train, X_test, train_metadata, test_metadata

def visualize_eeg_data(df, feature_cols=None, max_cols=10, time_window=None):
    """
    Visualize EEG data
    
    Parameters:
    df (DataFrame): DataFrame with EEG data
    feature_cols (list): Columns to visualize (None for automatic selection)
    max_cols (int): Maximum number of columns to visualize
    time_window (tuple): Optional (start, end) datetime for subsetting
    
    Returns:
    None
    """
    # Filter by time window if provided
    if time_window and isinstance(df.index, pd.DatetimeIndex):
        start, end = time_window
        df = df[(df.index >= start) & (df.index <= end)]
    
    # Select features to visualize if not specified
    if feature_cols is None:
        # Try to find raw EEG signals
        raw_eeg_columns = [col for col in df.columns if col.startswith('RAW_')]
        
        if raw_eeg_columns:
            feature_cols = raw_eeg_columns[:max_cols]
        else:
            # Try to find band power columns
            band_columns = []
            for band in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']:
                band_cols = [col for col in df.columns if col.startswith(band + '_')]
                band_columns.extend(band_cols)
            
            if band_columns:
                feature_cols = band_columns[:max_cols]
            else:
                # Use all numeric columns as fallback
                feature_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns[:max_cols]
    
    # Limit to maximum number of columns
    feature_cols = feature_cols[:max_cols]
    
    # Plot the selected features
    plt.figure(figsize=(14, 10))
    x_values = df.index if isinstance(df.index, pd.DatetimeIndex) else range(len(df))
    
    for column in feature_cols:
        if column in df.columns:
            plt.plot(x_values, df[column], label=column)
            
    plt.title('EEG Signal Visualization')
    plt.xlabel('Time' if isinstance(df.index, pd.DatetimeIndex) else 'Sample')
    plt.ylabel('Signal Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_bundle_examples(X_bundles, metadata_df, num_examples=3, max_features=5):
    """
    Plot example time series bundles
    
    Parameters:
    X_bundles (ndarray): Array of time series bundles
    metadata_df (DataFrame): Metadata for each bundle
    num_examples (int): Number of examples to plot
    max_features (int): Maximum number of features to plot per bundle
    
    Returns:
    None
    """
    # Select random bundles to plot
    np.random.seed(42)  # For reproducibility 
    indices = np.random.choice(len(X_bundles), min(num_examples, len(X_bundles)), replace=False)
    
    for i, idx in enumerate(indices):
        bundle = X_bundles[idx]
        file_id = metadata_df.iloc[idx]['file_id']
        
        plt.figure(figsize=(14, 6))
        
        # Plot each feature in the bundle (limited to max_features)
        num_features = min(bundle.shape[1], max_features)
        for j in range(num_features):
            plt.plot(range(bundle.shape[0]), bundle[:, j], 
                    label=f'Feature {j}')
        
        plt.title(f'Bundle Example {i+1} from file: {file_id}')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.tight_layout()
        plt.show()

def get_evenly_spaced_samples(df, time_window_seconds, num_samples):
    """
    Get evenly spaced samples from a time window
    
    Parameters:
    df (DataFrame): DataFrame with data for the time window
    time_window_seconds (float): Duration of the time window in seconds
    num_samples (int): Number of evenly spaced samples to extract
    
    Returns:
    DataFrame: Evenly spaced samples
    """
    # Check if we have enough samples
    if len(df) < num_samples:
        return None
    
    # Calculate indices for evenly spaced samples
    indices = np.linspace(0, len(df) - 1, num_samples).astype(int)
    
    # Extract samples
    samples = df.iloc[indices]
    
    return samples

def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file with proper handling of mixed data types.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        pandas DataFrame with the CSV data
        
    Raises:
        FileNotFoundError: If the file does not exist
        pd.errors.EmptyDataError: If the file is empty
        pd.errors.ParserError: If the file cannot be parsed
    """
    logger.debug(f"Loading CSV file: {file_path}")
    try:
        # Set low_memory=False to avoid DtypeWarning for mixed types
        df = pd.read_csv(file_path, low_memory=False)
        logger.debug(f"Successfully loaded CSV with {len(df)} rows")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"Empty CSV file: {file_path}")
        raise
    except pd.errors.ParserError:
        logger.error(f"Failed to parse CSV file: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading CSV file: {file_path}, Error: {str(e)}")
        logger.debug(traceback.format_exc())
        raise

def load_dataset_from_csv(file_path, **kwargs):
    """Load dataset from CSV file."""
    # Replace direct pd.read_csv call with our new function
    df = load_csv(file_path)
    
    # Rest of the existing function

def process_features(df_features):
    """Process features with forward and backward fill for missing values."""
    # Replace deprecated method with recommended alternatives
    df_features = df_features.ffill().bfill()
    return df_features