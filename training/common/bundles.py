"""
Functions for creating and managing time series bundles from EEG data.
"""
import os
import numpy as np
import pandas as pd
import gc
from typing import Dict, List, Optional, Tuple, Union, Any
import traceback

from .utils.logging import logger
from .utils.common import calculate_samples_from_seconds

class DiskBundleLoader:
    """
    Memory-efficient loader for EEG data bundles stored on disk.
    """
    
    def __init__(self, bundle_info: Dict[str, Any], 
                indices: Optional[np.ndarray] = None, 
                batch_size: int = 32, 
                shuffle: bool = True, 
                use_normalized: bool = True):
        """
        Initialize the bundle loader.
        
        Args:
            bundle_info: Bundle information from create_coherent_time_series_bundles_disk
            indices: Specific bundle indices to load (None for all)
            batch_size: Number of bundles to load per batch
            shuffle: Whether to shuffle the bundles
            use_normalized: Whether to use normalized bundles
        """
        self.bundle_info = bundle_info
        self.output_dir = bundle_info['output_dir']
        self.total_bundles = bundle_info['total_bundles']
        self.bundle_size = bundle_info['bundle_size']
        self.feature_dim = bundle_info['feature_dim']
        
        # Normalize path based on platform
        if os.name == 'nt':  # Windows
            self.output_dir = self.output_dir.replace('/', '\\')
        else:  # Unix/Linux
            self.output_dir = self.output_dir.replace('\\', '/')
        
        # Use all indices if not specified
        if indices is None:
            self.indices = np.arange(self.total_bundles)
        else:
            self.indices = indices
            
        self.num_samples = len(self.indices)
        self.batch_size = min(batch_size, self.num_samples)
        self.shuffle = shuffle
        self.use_normalized = use_normalized
        
        # For iteration
        self.current_idx = 0
        
        # Shuffle indices if needed
        if self.shuffle:
            np.random.shuffle(self.indices)
            
        logger.debug(f"Initialized DiskBundleLoader with {self.num_samples} bundles")
    
    def __iter__(self):
        """Reset iterator and return self."""
        self.current_idx = 0
        
        # Shuffle indices if needed
        if self.shuffle:
            np.random.shuffle(self.indices)
            
        return self
    
    def __next__(self):
        """Get next batch of bundles."""
        if self.current_idx >= self.num_samples:
            raise StopIteration
            
        # Calculate batch indices
        end_idx = min(self.current_idx + self.batch_size, self.num_samples)
        batch_indices = self.indices[self.current_idx:end_idx]
        
        # Load bundles for this batch
        batch_data = self._load_bundles(batch_indices)
        
        # Update index
        self.current_idx = end_idx
        
        return batch_data
    
    def _load_bundles(self, indices: np.ndarray) -> np.ndarray:
        """
        Load specific bundles from disk.
        
        Args:
            indices: Array of bundle indices to load
            
        Returns:
            Numpy array of loaded bundles
        """
        # Determine whether to load normalized or raw bundles
        bundle_dir = os.path.join(self.output_dir, 'normalized_bundles' if self.use_normalized else 'bundles')
        
        # Prepare array for batch
        # Check if bundles are 3D (time series) or 2D (flattened)
        sample_path = os.path.join(bundle_dir, f"bundle_{indices[0]}.npy")
        sample_bundle = np.load(sample_path)
        
        batch_shape = [len(indices)] + list(sample_bundle.shape)
        batch_data = np.zeros(batch_shape, dtype=np.float32)
        
        # Load each bundle
        for i, bundle_idx in enumerate(indices):
            try:
                bundle_path = os.path.join(bundle_dir, f"bundle_{bundle_idx}.npy")
                bundle_data = np.load(bundle_path)
                batch_data[i] = bundle_data
            except Exception as e:
                logger.error(f"Error loading bundle {bundle_idx}: {str(e)}")
                # Use zeros for failed loads
        
        return batch_data
    
    def __len__(self):
        """Get the number of batches."""
        return (self.num_samples + self.batch_size - 1) // self.batch_size

def create_coherent_time_series_bundles(file_dfs: Optional[Dict[str, pd.DataFrame]] = None,
                                      combined_df: Optional[pd.DataFrame] = None,
                                      bundle_size: int = 30, 
                                      step_size: int = 15,
                                      bundle_time_seconds: Optional[float] = None, 
                                      step_time_seconds: Optional[float] = None,
                                      smoothing_seconds: float = 0, 
                                      feature_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Create time series bundles from EEG data.
    
    Args:
        file_dfs: Dictionary mapping file IDs to DataFrames
        combined_df: Combined DataFrame with all files
        bundle_size: Number of time steps per bundle
        step_size: Step size for sliding window
        bundle_time_seconds: Bundle size in seconds (alternative to bundle_size)
        step_time_seconds: Step size in seconds (alternative to step_size)
        smoothing_seconds: Time window for smoothing in seconds
        feature_columns: Specific feature columns to use (None for all numeric)
        
    Returns:
        Tuple of (bundles, metadata_df) where:
          - bundles is a 3D array of shape (n_bundles, bundle_size, n_features)
          - metadata_df contains metadata for each bundle
    """
    logger.info("Creating time series bundles")
    
    # Validate input
    if file_dfs is None and combined_df is None:
        raise ValueError("Either file_dfs or combined_df must be provided")
        
    # Create combined_df if not provided
    if combined_df is None:
        combined_df = pd.concat(
            [df.assign(FileID=file_id) for file_id, df in file_dfs.items()],
            ignore_index=True
        )
    
    # Convert time-based parameters to sample counts if provided
    if bundle_time_seconds is not None:
        bundle_size = calculate_samples_from_seconds(combined_df, bundle_time_seconds)
        logger.info(f"Converted {bundle_time_seconds} seconds to {bundle_size} samples for bundle size")
        
    if step_time_seconds is not None:
        step_size = calculate_samples_from_seconds(combined_df, step_time_seconds)
        logger.info(f"Converted {step_time_seconds} seconds to {step_size} samples for step size")
    
    # Get list of DataFrames to process
    if file_dfs:
        dataframes = list(file_dfs.values())
        file_ids = list(file_dfs.keys())
    else:
        # Treat combined_df as a single DataFrame
        dataframes = [combined_df]
        file_ids = ['combined']
    
    # Apply smoothing if requested
    if smoothing_seconds > 0:
        logger.info(f"Applying {smoothing_seconds} second smoothing window")
        for i in range(len(dataframes)):
            dataframes[i] = apply_moving_average(dataframes[i], window_seconds=smoothing_seconds)
    
    # If feature_columns not specified, use all numeric columns
    all_bundles = []
    metadata_records = []
    
    # Process each file
    total_bundles = 0
    
    for df_idx, df in enumerate(dataframes):
        file_id = file_ids[df_idx]
        
        # Determine feature columns for this DataFrame
        if feature_columns is None:
            # Use all numeric columns except metadata
            exclude_cols = ['TimeStamp', 'FileID', 'SourceFile']
            df_features = df.select_dtypes(include=['number']).columns
            df_features = [col for col in df_features if col not in exclude_cols]
        else:
            # Filter for available columns
            df_features = [col for col in feature_columns if col in df.columns]
            
        if not df_features:
            logger.warning(f"No valid feature columns found for {file_id}")
            continue
            
        # Get feature data
        feature_data = df[df_features].values
        
        # Get timestamp data if available
        if 'TimeStamp' in df.columns:
            timestamps = pd.to_datetime(df['TimeStamp']).values
        else:
            # Use sequence numbers as timestamps
            timestamps = np.arange(len(df))
        
        # Create bundles using sliding window
        n_samples = len(feature_data)
        n_features = len(df_features)
        
        # Calculate number of bundles
        n_bundles = max(0, (n_samples - bundle_size) // step_size + 1)
        
        if n_bundles == 0:
            logger.warning(f"DataFrame {file_id} has insufficient samples ({n_samples}) "
                         f"for bundle size {bundle_size}")
            continue
            
        logger.info(f"Creating {n_bundles} bundles from {file_id} with {n_samples} samples")
        
        # Preallocate bundle array
        file_bundles = np.zeros((n_bundles, bundle_size, n_features), dtype=np.float32)
        
        # Create metadata records
        for i in range(n_bundles):
            start_idx = i * step_size
            end_idx = start_idx + bundle_size
            
            # Extract the time window
            file_bundles[i] = feature_data[start_idx:end_idx]
            
            # Store metadata
            metadata_records.append({
                'bundle_idx': total_bundles + i,
                'file_id': file_id,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_time': timestamps[start_idx],
                'end_time': timestamps[end_idx - 1]
            })
        
        # Add to all bundles
        all_bundles.append(file_bundles)
        total_bundles += n_bundles
    
    if not all_bundles:
        raise ValueError("No valid bundles could be created from the provided data")
        
    # Combine all bundles
    X_bundles = np.concatenate(all_bundles, axis=0)
    
    # Create metadata DataFrame
    metadata_df = pd.DataFrame(metadata_records)
    
    logger.info(f"Created {len(X_bundles)} total bundles with shape {X_bundles.shape}")
    
    return X_bundles, metadata_df

def normalize_bundles(X_bundles: np.ndarray, normalization: str = 'per_feature') -> np.ndarray:
    """
    Normalize bundle data.
    
    Args:
        X_bundles: Bundle data array
        normalization: Normalization method ('per_feature', 'per_bundle', 'global')
        
    Returns:
        Normalized bundle data
    """
    logger.info(f"Normalizing bundles with method: {normalization}")
    
    # Make a copy to avoid modifying the original
    X_normalized = X_bundles.copy()
    
    if normalization == 'per_feature':
        # Normalize each feature across all bundles and time steps
        n_bundles, n_timesteps, n_features = X_bundles.shape
        
        # Reshape to combine bundles and time steps
        X_reshaped = X_bundles.reshape(-1, n_features)
        
        # Calculate mean and std for each feature
        feature_means = np.mean(X_reshaped, axis=0)
        feature_stds = np.std(X_reshaped, axis=0)
        
        # Avoid division by zero
        feature_stds[feature_stds == 0] = 1.0
        
        # Normalize
        X_normalized_reshaped = (X_reshaped - feature_means) / feature_stds
        
        # Reshape back to original shape
        X_normalized = X_normalized_reshaped.reshape(n_bundles, n_timesteps, n_features)
        
    elif normalization == 'per_bundle':
        # Normalize each bundle independently
        for i in range(len(X_bundles)):
            bundle = X_bundles[i]
            bundle_mean = np.mean(bundle)
            bundle_std = np.std(bundle)
            
            # Avoid division by zero
            if bundle_std == 0:
                bundle_std = 1.0
                
            X_normalized[i] = (bundle - bundle_mean) / bundle_std
            
    elif normalization == 'global':
        # Normalize using global mean and std
        global_mean = np.mean(X_bundles)
        global_std = np.std(X_bundles)
        
        # Avoid division by zero
        if global_std == 0:
            global_std = 1.0
            
        X_normalized = (X_bundles - global_mean) / global_std
        
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")
    
    logger.info("Bundle normalization complete")
    return X_normalized

def split_bundles_train_test(X_bundles: np.ndarray, 
                           metadata_df: pd.DataFrame, 
                           test_ratio: float = 0.2, 
                           by_file: bool = True,
                           labels: Optional[np.ndarray] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Split bundles into training and test sets.
    
    Args:
        X_bundles: Bundle data array
        metadata_df: Metadata for bundles
        test_ratio: Ratio of data for testing
        by_file: Whether to split by file (True) or by bundle (False)
        labels: Optional labels array for stratified splitting
        
    Returns:
        Tuple of (train_data, test_data) dictionaries
    """
    logger.info(f"Splitting bundles with test_ratio={test_ratio}, by_file={by_file}")
    
    if by_file:
        # Split by file ID
        unique_files = metadata_df['file_id'].unique()
        n_files = len(unique_files)
        n_test_files = max(1, int(n_files * test_ratio))
        
        # Randomly select files for test set
        np.random.seed(42)  # For reproducibility
        test_files = np.random.choice(unique_files, n_test_files, replace=False)
        
        train_indices = metadata_df[~metadata_df['file_id'].isin(test_files)]['bundle_idx'].values
        test_indices = metadata_df[metadata_df['file_id'].isin(test_files)]['bundle_idx'].values
        
        logger.info(f"Split by file: {len(train_indices)} train, {len(test_indices)} test bundles")
        
    else:
        # Split by bundle index
        n_bundles = len(X_bundles)
        indices = np.arange(n_bundles)
        
        # Shuffle indices
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(indices)
        
        # Split indices
        split_idx = int(n_bundles * (1 - test_ratio))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        logger.info(f"Split by bundle: {len(train_indices)} train, {len(test_indices)} test bundles")
    
    # Create train/test dictionaries
    train_data = {
        'X': X_bundles[train_indices],
        'indices': train_indices,
        'metadata': metadata_df.iloc[train_indices].reset_index(drop=True)
    }
    
    test_data = {
        'X': X_bundles[test_indices],
        'indices': test_indices,
        'metadata': metadata_df.iloc[test_indices].reset_index(drop=True)
    }
    
    # Add labels if provided
    if labels is not None:
        train_data['y'] = labels[train_indices]
        test_data['y'] = labels[test_indices]
    
    return train_data, test_data

def create_coherent_time_series_bundles_disk(file_dfs: Optional[Dict[str, pd.DataFrame]] = None, 
                                           combined_df: Optional[pd.DataFrame] = None,
                                           bundle_size: int = 30, 
                                           step_size: int = 15,
                                           feature_columns: Optional[List[str]] = None,
                                           max_files_per_batch: int = 5, 
                                           max_bundles_per_file: Optional[int] = None,
                                           chunk_size: int = 1000,
                                           output_dir: str = './eeg_bundles'
                                           ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Create time series bundles and save them to disk for memory efficiency.
    
    Similar to create_coherent_time_series_bundles but saves bundles to disk
    instead of keeping them in memory.
    
    Args:
        (Same parameters as create_coherent_time_series_bundles)
        max_files_per_batch: Maximum files to process in a single batch
        max_bundles_per_file: Maximum bundles to create per file (None for all)
        max_memory_mb: Maximum memory usage in MB
        chunk_size: Number of bundles to process in each disk write
        output_dir: Directory to save bundles
        sampling_rate: EEG sampling rate in Hz
        
    Returns:
        Tuple of (metadata_df, bundle_info) where:
          - metadata_df contains metadata for each bundle
          - bundle_info is a dictionary with bundle information
    """
    logger.info("Creating time series bundles and saving to disk")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    bundle_dir = os.path.join(output_dir, 'bundles')
    os.makedirs(bundle_dir, exist_ok=True)
    
    # Validate input
    if file_dfs is None and combined_df is None:
        raise ValueError("Either file_dfs or combined_df must be provided")
        
    # Create combined_df if not provided
    if combined_df is None and file_dfs is not None:
        combined_df = pd.concat(
            [df.assign(FileID=file_id) for file_id, df in file_dfs.items()],
            ignore_index=True
        )
    
    # Get list of DataFrames to process
    if file_dfs:
        file_ids = list(file_dfs.keys())
    else:
        # Treat combined_df as a single DataFrame
        file_dfs = {'combined': combined_df}
        file_ids = ['combined']
    
    # Process files in batches
    metadata_records = []
    total_bundles = 0
    feature_dim = None
    
    for batch_start in range(0, len(file_ids), max_files_per_batch):
        batch_end = min(batch_start + max_files_per_batch, len(file_ids))
        batch_file_ids = file_ids[batch_start:batch_end]
        
        logger.info(f"Processing files {batch_start+1}-{batch_end} of {len(file_ids)}")
        
        for file_id in batch_file_ids:
            df = file_dfs[file_id]
            
            
            # Determine feature columns for this DataFrame
            if feature_columns is None:
                # Use all numeric columns except metadata
                exclude_cols = ['TimeStamp', 'FileID', 'SourceFile']
                df_features = df.select_dtypes(include=['number']).columns
                df_features = [col for col in df_features if col not in exclude_cols]
            else:
                # Filter for available columns
                df_features = [col for col in feature_columns if col in df.columns]
                
            if not df_features:
                logger.warning(f"No valid feature columns found for {file_id}")
                continue
                
            # Save feature dimension
            if feature_dim is None:
                feature_dim = len(df_features)
            elif feature_dim != len(df_features):
                logger.warning(f"Feature dimension mismatch: expected {feature_dim}, got {len(df_features)}")
                continue
                
            # Get feature data
            feature_data = df[df_features].values
            
            # Get timestamp data if available
            if 'TimeStamp' in df.columns:
                timestamps = pd.to_datetime(df['TimeStamp']).values
            else:
                # Use sequence numbers as timestamps
                timestamps = np.arange(len(df))
            
            # Create bundles using sliding window
            n_samples = len(feature_data)
            
            # Calculate number of bundles
            n_bundles = max(0, (n_samples - bundle_size) // step_size + 1)
            
            if n_bundles == 0:
                logger.warning(f"DataFrame {file_id} has insufficient samples ({n_samples}) "
                             f"for bundle size {bundle_size}")
                continue
                
            # Limit bundles if requested
            if max_bundles_per_file is not None:
                n_bundles = min(n_bundles, max_bundles_per_file)
                
            logger.info(f"Creating {n_bundles} bundles from {file_id} with {n_samples} samples")
            
            # Process in chunks to limit memory usage
            for chunk_start in range(0, n_bundles, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_bundles)
                chunk_bundles = chunk_end - chunk_start
                
                # Create metadata and save bundles
                for i in range(chunk_start, chunk_end):
                    start_idx = i * step_size
                    end_idx = start_idx + bundle_size
                    
                    # Extract the time window
                    bundle_data = feature_data[start_idx:end_idx]
                    
                    # Save bundle to disk
                    bundle_path = os.path.join(bundle_dir, f"bundle_{total_bundles}.npy")
                    np.save(bundle_path, bundle_data)
                    
                    # Store metadata
                    metadata_records.append({
                        'bundle_idx': total_bundles,
                        'file_id': file_id,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'start_time': timestamps[start_idx],
                        'end_time': timestamps[end_idx - 1],
                        'bundle_path': bundle_path
                    })
                    
                    total_bundles += 1
                    
                # Explicitly collect garbage
                gc.collect()
    
    if total_bundles == 0:
        raise ValueError("No valid bundles could be created from the provided data")
    
    # Create metadata DataFrame
    metadata_df = pd.DataFrame(metadata_records)
    
    # Save metadata to disk
    metadata_path = os.path.join(output_dir, 'bundle_metadata.csv')
    metadata_df.to_csv(metadata_path, index=False)
    
    # Create bundle info
    bundle_info = {
        'total_bundles': total_bundles,
        'bundle_size': bundle_size,
        'feature_dim': feature_dim,
        'output_dir': output_dir,
        'metadata_path': metadata_path
    }
    
    logger.info(f"Created {total_bundles} total bundles on disk")
    
    return metadata_df, bundle_info

def normalize_bundles_disk(bundle_info: Dict[str, Any], normalization: str = 'per_feature', batch_size=100):
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

def split_bundles_disk(metadata_df: pd.DataFrame, test_ratio: float = 0.2, by_file: bool = True, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split bundle indices into training and test sets.
    
    Args:
        metadata_df: Metadata DataFrame for bundles
        test_ratio: Proportion of data to use for testing
        by_file: Whether to split by file ID or by bundle
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_indices, test_indices)
    """
    logger.info(f"Splitting bundles with test_ratio={test_ratio}, by_file={by_file}")
    
    if by_file:
        # Split by file ID
        unique_files = metadata_df['file_id'].unique()
        n_files = len(unique_files)
        n_test_files = max(1, int(n_files * test_ratio))
        
        # Randomly select files for test set
        np.random.seed(random_state)
        test_files = np.random.choice(unique_files, n_test_files, replace=False)
        
        # Get bundle indices
        train_indices = metadata_df[~metadata_df['file_id'].isin(test_files)]['bundle_idx'].values
        test_indices = metadata_df[metadata_df['file_id'].isin(test_files)]['bundle_idx'].values
        
    else:
        # Split by bundle index
        n_bundles = len(metadata_df)
        indices = np.arange(n_bundles)
        
        # Shuffle indices
        np.random.seed(random_state)
        np.random.shuffle(indices)
        
        # Split indices
        split_idx = int(n_bundles * (1 - test_ratio))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
    
    logger.info(f"Split result: {len(train_indices)} training, {len(test_indices)} test bundles")
    
    return train_indices, test_indices 