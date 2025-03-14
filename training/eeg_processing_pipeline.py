#!/usr/bin/env python
"""
EEG Processing Pipeline

This script provides a complete pipeline for EEG data processing and analysis
using disk-based operations for memory efficiency. It performs:
1. Data loading from a directory
2. Data quality analysis to identify NaN patterns
3. Temporal sampling with sliding windows
4. Feature engineering
5. Time series bundling with disk-based operations
6. Data normalization
7. Unsupervised clustering
8. Evaluation of results

Usage:
    python eeg_processing_pipeline.py --data_dir /path/to/data --output_dir /path/to/output

Author: EEG Classification Team
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import joblib
import gc  # For garbage collection

# Import from our refactored modules
from training.common import (
    # Logger
    logger, setup_logger,
    
    # Data loading
    load_eeg_data,
    
    # Preprocessing
    preprocess_eeg_data, engineer_eeg_features, 
    sample_by_sliding_window, process_eeg_files_with_sampling,
    analyze_data_quality,
    
    # Disk-based bundle functions
    create_coherent_time_series_bundles_disk, normalize_bundles_disk,
    
    # Models
    UnsupervisedModelTrainer,
    
    # Visualization
    visualize_eeg_data
)

def print_section_header(title):
    """Print a section header to make output easier to read"""
    logger.info("\n" + "="*80)
    logger.info(f" {title}")
    logger.info("="*80)

def setup_directories(base_dir):
    """Create output directories if they don't exist"""
    os.makedirs(base_dir, exist_ok=True)
    
    # Create subdirectories
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    bundle_dir = os.path.join(base_dir, "bundles")
    os.makedirs(bundle_dir, exist_ok=True)
    
    return {
        "base": base_dir,
        "models": models_dir,
        "plots": plots_dir,
        "bundles": bundle_dir
    }

def save_plot(fig, filename, output_dir):
    """Save matplotlib figure to file"""
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved plot: {filepath}")

def load_bundle_info(bundle_dir):
    """
    Load bundle information from a directory
    
    Parameters:
        bundle_dir (str): Path to the directory containing bundle information
        
    Returns:
        dict: Bundle information dictionary
    """
    # Check for bundle info file
    info_path = os.path.join(bundle_dir, "bundle_info.joblib")
    
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Bundle info file not found in {bundle_dir}")
    
    # Load bundle info
    bundle_info = joblib.load(info_path)
    
    # Validate the bundle info
    required_keys = ['total_bundles', 'bundle_size', 'feature_dim', 'output_dir']
    for key in required_keys:
        if key not in bundle_info:
            raise ValueError(f"Bundle info is missing required key: {key}")
    
    # Make sure paths are correct if the bundles were moved
    if os.path.abspath(bundle_info['output_dir']) != os.path.abspath(bundle_dir):
        logger.warning(f"Bundle directory has changed from {bundle_info['output_dir']} to {bundle_dir}")
        bundle_info['output_dir'] = os.path.abspath(bundle_dir)
    
    return bundle_info

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="EEG Processing Pipeline")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing EEG data files")
    parser.add_argument("--output_dir", type=str, default="./eeg_results", help="Output directory")
    parser.add_argument("--window_size", type=float, default=5.0, help="Sliding window size in seconds")
    parser.add_argument("--samples_per_window", type=int, default=10, help="Number of samples to select per window")
    parser.add_argument("--bundle_size", type=int, default=30, help="Size of each time bundle")
    parser.add_argument("--step_size", type=int, default=15, help="Step size for sliding window")
    parser.add_argument("--n_clusters", type=int, default=None, help="Number of clusters (default: auto-detect)")
    parser.add_argument("--log_level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--load_existing", action="store_true", help="Load existing bundles if available")
    
    args = parser.parse_args()
    
    # Configure logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")
    
    # Convert string log level to actual level
    numeric_level = getattr(logging, args.log_level.upper(), None)
    setup_logger(name="eeg_classification", level=numeric_level, log_file=log_file)
    
    # Start timer
    start_time = time.time()
    
    logger.info("Starting EEG Processing Pipeline")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Log file: {log_file}")
    
    # Setup directories
    dirs = setup_directories(args.output_dir)
    
    # Set bundle directory
    bundle_dir = dirs["bundles"]
    
    # Check if we should load existing bundles
    bundle_info = None
    if args.load_existing:
        try:
            logger.info("Checking for existing bundles...")
            bundle_info = load_bundle_info(bundle_dir)
            logger.info(f"Found existing bundles: {bundle_info['total_bundles']} bundles with shape " 
                       f"({bundle_info['bundle_size']}, {bundle_info['feature_dim']})")
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Could not load existing bundles: {str(e)}")
            bundle_info = None
    
    # If we don't have existing bundles, create them
    if bundle_info is None:
        #---------------------------------------------------------------------------
        # Step 1: Load Data
        #---------------------------------------------------------------------------
        print_section_header("Step 1: Loading EEG data")
        step_start = time.time()
        
        try:
            logger.info(f"Loading data from {args.data_dir}")
            file_dfs, combined_df = load_eeg_data(directory_path=args.data_dir)
            
            n_files = len(file_dfs)
            total_rows = sum(len(df) for df in file_dfs.values())
            logger.info(f"Loaded {n_files} files with {total_rows} total rows")
            
            # Get first file for visualization
            sample_file_id = list(file_dfs.keys())[0]
            sample_df = file_dfs[sample_file_id]
            logger.info(f"Sample file {sample_file_id} shape: {sample_df.shape}")
            
            # Visualize raw data
            fig = visualize_eeg_data(sample_df, max_cols=5)
            save_plot(fig, "raw_eeg_data.png", dirs["plots"])
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            logger.exception("Stack trace:")
            return
        
        logger.info(f"Step 1 completed in {time.time() - step_start:.2f} seconds")
        
        #---------------------------------------------------------------------------
        # Step 2: Data Quality Analysis
        #---------------------------------------------------------------------------
        print_section_header("Step 2: Analyzing data quality")
        step_start = time.time()
        
        try:
            # Analyze NaN patterns in the sample file first
            logger.info(f"Analyzing data quality for sample file {sample_file_id}")
            sample_quality = analyze_data_quality(sample_df)
            
            # Save the plot if it exists
            if 'plot' in sample_quality and sample_quality['plot'] is not None:
                save_plot(sample_quality['plot'], "sample_data_quality.png", dirs["plots"])
            
            # Analyze combined data quality if it's not too large
            if combined_df is not None and len(combined_df) <= 100000:  # Only for reasonably sized datasets
                logger.info("Analyzing data quality for combined dataset")
                combined_quality = analyze_data_quality(combined_df)
                
                if 'plot' in combined_quality and combined_quality['plot'] is not None:
                    save_plot(combined_quality['plot'], "combined_data_quality.png", dirs["plots"])
            else:
                logger.info("Combined dataset too large for full quality analysis")
                
                # Summary statistics across all files
                total_nans_all_files = 0
                total_cells_all_files = 0
                high_nan_cols_all_files = set()
                
                # Analyze each file individually
                for file_id, df in file_dfs.items():
                    if file_id == sample_file_id:
                        continue  # Skip, already analyzed
                        
                    logger.info(f"Calculating NaN statistics for file {file_id}")
                    total_nans = df.isna().sum().sum()
                    total_cells = df.size
                    total_nans_all_files += total_nans
                    total_cells_all_files += total_cells
                    
                    # Get high NaN columns
                    nan_percentages = (df.isna().sum() / len(df)) * 100
                    high_nan_cols = nan_percentages[nan_percentages > 50].index.tolist()
                    high_nan_cols_all_files.update(high_nan_cols)
                
                # Report summary
                if total_cells_all_files > 0:
                    overall_nan_percent = (total_nans_all_files / total_cells_all_files) * 100
                    logger.info(f"Overall NaN percentage across all files: {overall_nan_percent:.2f}%")
                    logger.info(f"Found {len(high_nan_cols_all_files)} columns with >50% NaNs in at least one file")
                    
                    if high_nan_cols_all_files:
                        logger.warning("Columns with high NaN percentages:")
                        for col in list(high_nan_cols_all_files)[:20]:  # Show up to 20 columns
                            logger.warning(f"  - {col}")
                        if len(high_nan_cols_all_files) > 20:
                            logger.warning(f"  ... and {len(high_nan_cols_all_files) - 20} more columns")
            
            # Save high NaN columns for later use
            high_nan_cols = sample_quality.get('high_nan_columns', [])
            joblib.dump(high_nan_cols, os.path.join(dirs["base"], "high_nan_columns.joblib"))
            logger.info(f"Saved list of {len(high_nan_cols)} high-NaN columns for reference")
            
        except Exception as e:
            logger.error(f"Error in data quality analysis: {str(e)}")
            logger.exception("Stack trace:")
            # Continue with pipeline despite analysis error
        
        logger.info(f"Step 2 completed in {time.time() - step_start:.2f} seconds")
        
        #---------------------------------------------------------------------------
        # Step 3: Temporal Sampling
        #---------------------------------------------------------------------------
        print_section_header("Step 3: Temporal sampling with sliding windows")
        step_start = time.time()
        
        try:
            # Log sampling parameters
            logger.info(f"Sampling parameters:")
            logger.info(f"  Window size: {args.window_size} seconds")
            logger.info(f"  Samples per window: {args.samples_per_window}")
            
            # Get total data size before sampling
            total_before = sum(len(df) for df in file_dfs.values())
            logger.info(f"Total data points before sampling: {total_before}")
            
            # Apply sliding window sampling to each file
            sampled_dfs = process_eeg_files_with_sampling(
                file_dfs=file_dfs,
                window_size_seconds=args.window_size,
                samples_per_window=args.samples_per_window,
                timestamp_col='TimeStamp'
            )
            
            # Get total data size after sampling
            total_after = sum(len(df) for df in sampled_dfs.values())
            logger.info(f"Total data points after sampling: {total_after}")
            logger.info(f"Reduction ratio: {total_after / total_before:.2f}")
            
            # Replace original data with sampled data
            file_dfs = sampled_dfs
            
            # Update sample_df for visualization
            sample_df = file_dfs[sample_file_id]
            
            # Visualize sampled data
            fig = visualize_eeg_data(sample_df, max_cols=5)
            save_plot(fig, "sampled_eeg_data.png", dirs["plots"])
            
            # Create scatter plot to show sampling distribution for sample file
            if 'TimeStamp' in sample_df.columns:
                # Get a representative EEG channel
                eeg_cols = [col for col in sample_df.columns if col.startswith('EEG_') or col.startswith('CH')]
                if eeg_cols:
                    channel = eeg_cols[0]
                    plt.figure(figsize=(12, 6))
                    plt.scatter(sample_df['TimeStamp'], sample_df[channel], 
                                alpha=0.7, label='Sampled Data', s=15)
                    plt.title(f'Temporal Distribution of Sampled Data - {channel}')
                    plt.xlabel('Time')
                    plt.ylabel('Amplitude')
                    plt.legend()
                    plt.grid(True)
                    save_plot(plt.gcf(), "sampling_distribution.png", dirs["plots"])
            
        except Exception as e:
            logger.error(f"Error in temporal sampling: {str(e)}")
            logger.exception("Stack trace:")
            return
        
        logger.info(f"Step 3 completed in {time.time() - step_start:.2f} seconds")
        
        #---------------------------------------------------------------------------
        # Step 4: Preprocess Data and Engineer Features
        #---------------------------------------------------------------------------
        print_section_header("Step 4: Processing data and engineering features")
        step_start = time.time()
        
        try:
            processed_dfs = {}
            
            for file_id, df in file_dfs.items():
                logger.info(f"Processing file {file_id} with {len(df)} rows")
                
                # Check for NaN values in input
                nan_count_before = df.isna().sum().sum()
                if nan_count_before > 0:
                    logger.warning(f"Input data for file {file_id} contains {nan_count_before} NaN values")
                
                # Clean data
                clean_df = preprocess_eeg_data(df)
                logger.info(f"Cleaned data shape: {clean_df.shape}")
                
                # Check for NaN values after preprocessing
                nan_count_after_preprocess = clean_df.isna().sum().sum()
                if nan_count_after_preprocess > 0:
                    logger.warning(f"Preprocessed data for file {file_id} contains {nan_count_after_preprocess} NaN values")
                    
                    # Fill NaN values with column means
                    numeric_cols = clean_df.select_dtypes(include=['number']).columns
                    clean_df[numeric_cols] = clean_df[numeric_cols].fillna(clean_df[numeric_cols].mean())
                    logger.info(f"Filled NaN values with column means")
                
                # Engineer features
                features_df = engineer_eeg_features(clean_df)
                logger.info(f"Features shape: {features_df.shape}")
                
                # Check for NaN values after feature engineering
                nan_count_after_features = features_df.isna().sum().sum()
                if nan_count_after_features > 0:
                    logger.warning(f"Engineered features for file {file_id} contains {nan_count_after_features} NaN values")
                    
                    # Fill any remaining NaN values
                    numeric_cols = features_df.select_dtypes(include=['number']).columns
                    features_df[numeric_cols] = features_df[numeric_cols].fillna(features_df[numeric_cols].mean())
                    
                    # If any columns are all NaN, drop them
                    all_nan_cols = features_df.columns[features_df.isna().all()].tolist()
                    if all_nan_cols:
                        logger.warning(f"Dropping {len(all_nan_cols)} columns that are all NaN: {all_nan_cols}")
                        features_df = features_df.drop(columns=all_nan_cols)
                    
                    # Verify all NaNs have been handled
                    remaining_nans = features_df.isna().sum().sum()
                    if remaining_nans > 0:
                        logger.error(f"Failed to handle all NaN values in features. {remaining_nans} NaNs remain.")
                        # For any remaining NaNs, use forward and backward fill
                        features_df = features_df.ffill().bfill()
                        # Last resort: replace with zeros
                        features_df = features_df.fillna(0)
                        logger.warning("Filled remaining NaNs with forward/backward fill and zeros")
                
                # Store processed DataFrame
                processed_dfs[file_id] = features_df
                
                # If this is the first file, visualize the processed data
                if file_id == sample_file_id:
                    fig = visualize_eeg_data(features_df, max_cols=5)
                    save_plot(fig, "processed_eeg_data.png", dirs["plots"])
                
                # Free memory
                del clean_df
                gc.collect()
                
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            logger.exception("Stack trace:")
            return
        
        logger.info(f"Step 4 completed in {time.time() - step_start:.2f} seconds")
        
        # Additional check for NaN values across all processed files
        total_nans = sum(df.isna().sum().sum() for df in processed_dfs.values())
        if total_nans > 0:
            logger.warning(f"Processed data still contains {total_nans} NaN values across all files")
            logger.warning("These will be handled during the bundling process")
        else:
            logger.info("No NaN values detected in processed data")
        
        #---------------------------------------------------------------------------
        # Step 5: Create Time Series Bundles on Disk
        #---------------------------------------------------------------------------
        print_section_header("Step 5: Creating time series bundles on disk")
        step_start = time.time()
        
        try:
            # Configuration for bundle creation
            bundle_config = {
                'bundle_size': args.bundle_size,         # Number of time steps per bundle
                'step_size': args.step_size,             # Step size for sliding window
                'smoothing_seconds': 0.1,                # Apply 100ms smoothing
                'max_files_per_batch': 2,                # Process 2 files at a time
                'max_bundles_per_file': 1000,            # Maximum 1000 bundles per file
                'chunk_size': 500,                       # Process 500 bundles at a time
                'output_dir': bundle_dir,                # Where to save bundles
                'sampling_rate': 256                     # Assumed sampling rate in Hz
            }
            
            logger.info("Bundle creation configuration:")
            for key, value in bundle_config.items():
                logger.info(f"  {key}: {value}")
            
            # Create time series bundles and save to disk
            logger.info("Creating time series bundles with timestamp-based sampling...")
            metadata_df, bundle_info = create_coherent_time_series_bundles_disk(
                file_dfs=processed_dfs,
                **bundle_config
            )
            
            logger.info(f"Created {bundle_info['total_bundles']} bundles on disk")
            logger.info(f"Bundle size: {bundle_info['bundle_size']} time steps")
            logger.info(f"Feature dimension: {bundle_info['feature_dim']} features")
            logger.info(f"Bundles saved to: {bundle_info['output_dir']}")
            
            # Save metadata to a more readable format
            metadata_csv = os.path.join(dirs['base'], "bundle_metadata.csv")
            metadata_df.to_csv(metadata_csv, index=False)
            logger.info(f"Saved bundle metadata to {metadata_csv}")
            
            # Free memory
            del processed_dfs
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error creating bundles: {str(e)}")
            logger.exception("Stack trace:")
            return
        
        logger.info(f"Step 5 completed in {time.time() - step_start:.2f} seconds")
        
        #---------------------------------------------------------------------------
        # Step 6: Normalize Bundles on Disk
        #---------------------------------------------------------------------------
        print_section_header("Step 6: Normalizing bundles on disk")
        step_start = time.time()
        
        try:
            # Normalize bundles
            logger.info("Normalizing bundles...")
            bundle_info = normalize_bundles_disk(
                bundle_info=bundle_info,
                normalization='per_feature'  # Normalize each feature independently
            )
            
            logger.info(f"Normalized bundles using {bundle_info['normalization']} method")
            logger.info(f"Normalization parameters saved to {bundle_info['normalization_params_path']}")
            
        except Exception as e:
            logger.error(f"Error normalizing bundles: {str(e)}")
            logger.exception("Stack trace:")
            return
        
        logger.info(f"Step 6 completed in {time.time() - step_start:.2f} seconds")
    
    else:
        # We loaded existing bundles, so let's load the metadata too
        metadata_csv = os.path.join(dirs['base'], "bundle_metadata.csv")
        if os.path.exists(metadata_csv):
            metadata_df = pd.read_csv(metadata_csv)
            logger.info(f"Loaded existing metadata with {len(metadata_df)} rows")
        else:
            # Try to find metadata in the bundle directory
            metadata_path = os.path.join(bundle_dir, "metadata.csv")
            if os.path.exists(metadata_path):
                metadata_df = pd.read_csv(metadata_path)
                logger.info(f"Loaded metadata from bundle directory with {len(metadata_df)} rows")
            else:
                # Try joblib format
                metadata_path = os.path.join(bundle_dir, "metadata.joblib")
                if os.path.exists(metadata_path):
                    metadata_df = joblib.load(metadata_path)
                    logger.info(f"Loaded metadata from bundle directory with {len(metadata_df)} rows")
                else:
                    logger.error("Could not find metadata file")
                    return
    
    #---------------------------------------------------------------------------
    # Step 7: Unsupervised Learning (Clustering)
    #---------------------------------------------------------------------------
    print_section_header("Step 7: Performing unsupervised clustering")
    step_start = time.time()
    
    try:
        # Create loader for the data
        from training.common import DiskBundleLoader
        
        # For clustering, we need to create a batch loader
        all_indices = np.arange(bundle_info['total_bundles'])
        
        # Split into analysis (80%) and evaluation (20%) sets
        from sklearn.model_selection import train_test_split
        analysis_indices, evaluation_indices = train_test_split(
            all_indices, test_size=0.2, random_state=42
        )
        
        logger.info(f"Split {len(all_indices)} bundles into {len(analysis_indices)} for analysis " 
                  f"and {len(evaluation_indices)} for evaluation")
        
        # Save indices for future reference
        np.save(os.path.join(dirs['base'], 'analysis_indices.npy'), analysis_indices)
        np.save(os.path.join(dirs['base'], 'evaluation_indices.npy'), evaluation_indices)
        
        # Initialize the clustering model
        trainer = UnsupervisedModelTrainer(
            model_type='kmeans',
            random_state=42,
            config={'n_clusters': args.n_clusters if args.n_clusters is not None else 2}  # Default to 2 if not specified
        )
        
        # Get a sample for finding optimal number of clusters
        sample_size = min(1000, len(analysis_indices))
        sample_indices = np.random.choice(analysis_indices, sample_size, replace=False)
        
        logger.info(f"Using {sample_size} samples to determine optimal number of clusters")
        
        # Create loader for sample data
        sample_loader = DiskBundleLoader(
            bundle_info=bundle_info,
            indices=sample_indices,
            batch_size=sample_size,  # Load all in one batch
            shuffle=False,
            use_normalized=True
        )
        
        # Load sample data for cluster optimization
        sample_data = next(iter(sample_loader))
        
        # If data is 3D, flatten time dimension by taking mean
        if sample_data.ndim == 3:
            sample_data = sample_data.mean(axis=1)
            
        logger.info(f"Sample data shape for clustering: {sample_data.shape}")
        
        # Check for NaN values and handle them
        nan_count = np.isnan(sample_data).sum()
        logger.info(f"Number of NaN values in sample data: {nan_count}")
        
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in the data - applying NaN handling")
            
            # Option 1: Replace NaN with feature means
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            sample_data = imputer.fit_transform(sample_data)
            logger.info("Replaced NaN values with feature means")
            
            # Save the imputer for later use with evaluation data
            imputer_path = os.path.join(dirs["models"], "imputer.joblib")
            joblib.dump(imputer, imputer_path)
            logger.info(f"Saved imputer to {imputer_path}")
        
        # Verify all NaNs have been handled
        remaining_nans = np.isnan(sample_data).sum()
        if remaining_nans > 0:
            logger.error(f"Still found {remaining_nans} NaN values after imputation")
            raise ValueError("Failed to handle all NaN values in the data")
        else:
            logger.info("No NaN values remaining after preprocessing")
            
        # Find optimal number of clusters if not specified
        if args.n_clusters is None:
            optimal_n, score = trainer.find_optimal_clusters(
                sample_data, min_clusters=2, max_clusters=5
            )
            logger.info(f"Optimal number of clusters: {optimal_n} (score: {score:.4f})")
        else:
            optimal_n = args.n_clusters
            logger.info(f"Using specified number of clusters: {optimal_n}")
            
        # Set the optimal number of clusters in the trainer
        trainer.create_model(n_clusters=optimal_n)
        
        # Process all analysis data in batches
        logger.info(f"Clustering {len(analysis_indices)} bundles...")
        
        batch_size = 500  # Process 500 samples at a time
        analysis_batches = [analysis_indices[i:i+batch_size] for i in range(0, len(analysis_indices), batch_size)]
        
        all_labels = []
        all_data_for_vis = None  # For visualization
        
        for i, batch_indices in enumerate(analysis_batches):
            # Create loader for this batch
            batch_loader = DiskBundleLoader(
                bundle_info=bundle_info,
                indices=batch_indices,
                batch_size=len(batch_indices),
                shuffle=False,
                use_normalized=True
            )
            
            # Load batch
            batch_data = next(iter(batch_loader))
            
            # If bundles are 3D, flatten the time dimension
            if batch_data.ndim == 3:
                batch_data = batch_data.mean(axis=1)
            
            # Handle NaN values in batch data
            batch_nan_count = np.isnan(batch_data).sum()
            if batch_nan_count > 0:
                logger.warning(f"Found {batch_nan_count} NaN values in batch {i+1} - applying imputation")
                batch_data = imputer.transform(batch_data)
            
            # Store first batch for visualization
            if i == 0 and all_data_for_vis is None:
                # Keep only a subset for visualization to save memory
                vis_size = min(500, len(batch_data))
                all_data_for_vis = batch_data[:vis_size].copy()
                
            # Fit or predict with the model
            if i == 0:
                # First batch - fit the model
                batch_labels = trainer.train(batch_data)
                logger.info("Fitted clustering model on first batch")
            else:
                # Subsequent batches - predict only
                batch_labels = trainer.model.predict(batch_data)
            
            all_labels.append(batch_labels)
            
            logger.info(f"  Processed batch {i+1}/{len(analysis_batches)} ({len(batch_indices)} bundles)")
            
            # Free memory
            del batch_data, batch_loader
            gc.collect()
        
        # Combine all labels
        all_labels = np.concatenate(all_labels)
        
        # Get cluster distribution
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        logger.info("Cluster distribution:")
        for label, count in zip(unique_labels, counts):
            percentage = (count / len(all_labels)) * 100
            logger.info(f"  Cluster {label}: {count} samples ({percentage:.1f}%)")
        
        # Visualize clusters using stored data subset
        if all_data_for_vis is not None:
            # Reduce dimensions for visualization
            # Try t-SNE first, fall back to PCA if not available
            try:
                from sklearn.manifold import TSNE
                logger.info("Using t-SNE for dimension reduction...")
                X_reduced = TSNE(n_components=2, random_state=42).fit_transform(all_data_for_vis)
            except:
                logger.info("Falling back to PCA for dimension reduction...")
                from sklearn.decomposition import PCA
                X_reduced = PCA(n_components=2, random_state=42).fit_transform(all_data_for_vis)
            
            # Get corresponding labels
            vis_labels = trainer.model.predict(all_data_for_vis)
            
            # Create scatter plot
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=vis_labels, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, label='Cluster')
            plt.title(f'EEG Data Clusters (n={optimal_n})')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.tight_layout()
            save_plot(plt.gcf(), "cluster_visualization.png", dirs["plots"])
            
            # Free memory
            del all_data_for_vis
            gc.collect()
        
        # Save the model
        model_path = os.path.join(dirs["models"], "unsupervised_model.joblib")
        trainer.save_model(model_path)
        logger.info(f"Saved clustering model to {model_path}")
        
        # Add cluster labels to metadata
        metadata_with_clusters = metadata_df.copy()
        metadata_with_clusters['cluster'] = -1  # Initialize with -1 (unclustered)
        
        # Map indices to bundle_idx in metadata
        for i, idx in enumerate(analysis_indices):
            if i < len(all_labels):
                metadata_with_clusters.loc[metadata_with_clusters['bundle_idx'] == idx, 'cluster'] = all_labels[i]
        
        # Save updated metadata
        metadata_path = os.path.join(dirs["base"], "metadata_with_clusters.csv")
        metadata_with_clusters.to_csv(metadata_path, index=False)
        logger.info(f"Saved metadata with cluster labels to {metadata_path}")
        
    except Exception as e:
        logger.error(f"Error in unsupervised learning: {str(e)}")
        logger.exception("Stack trace:")
        return
    
    logger.info(f"Step 7 completed in {time.time() - step_start:.2f} seconds")
    
    #---------------------------------------------------------------------------
    # Step 8: Cluster Evaluation
    #---------------------------------------------------------------------------
    print_section_header("Step 8: Evaluating clustering results")
    step_start = time.time()
    
    try:
        # Evaluate on the held-out evaluation set
        logger.info(f"Evaluating clustering on {len(evaluation_indices)} held-out bundles")
        
        # Process in batches
        batch_size = 500
        evaluation_batches = [evaluation_indices[i:i+batch_size] for i in range(0, len(evaluation_indices), batch_size)]
        
        eval_labels = []
        
        for i, batch_indices in enumerate(evaluation_batches):
            # Create loader for this batch
            batch_loader = DiskBundleLoader(
                bundle_info=bundle_info,
                indices=batch_indices,
                batch_size=len(batch_indices),
                shuffle=False,
                use_normalized=True
            )
            
            # Load batch
            batch_data = next(iter(batch_loader))
            
            # If bundles are 3D, flatten the time dimension
            if batch_data.ndim == 3:
                batch_data = batch_data.mean(axis=1)
                
            # Handle NaN values
            nan_count = np.isnan(batch_data).sum()
            if nan_count > 0:
                logger.warning(f"Found {nan_count} NaN values in evaluation data - applying imputation")
                # Load the imputer if it exists
                imputer_path = os.path.join(dirs["models"], "imputer.joblib")
                if os.path.exists(imputer_path):
                    imputer = joblib.load(imputer_path)
                    batch_data = imputer.transform(batch_data)
                else:
                    # Create a new imputer if needed
                    imputer = SimpleImputer(strategy='mean')
                    batch_data = imputer.fit_transform(batch_data)
            
            # Predict clusters
            batch_labels = trainer.model.predict(batch_data)
            eval_labels.append(batch_labels)
            
            logger.info(f"  Processed evaluation batch {i+1}/{len(evaluation_batches)} ({len(batch_indices)} bundles)")
            
            # Free memory
            del batch_data, batch_loader
            gc.collect()
        
        # Combine all evaluation labels
        eval_labels = np.concatenate(eval_labels)
        
        # Get cluster distribution
        eval_unique_labels, eval_counts = np.unique(eval_labels, return_counts=True)
        logger.info("Cluster distribution in evaluation data:")
        for label, count in zip(eval_unique_labels, eval_counts):
            percentage = (count / len(eval_labels)) * 100
            logger.info(f"  Cluster {label}: {count} samples ({percentage:.1f}%)")
        
        # Compare distributions between analysis and evaluation sets
        logger.info("Comparing cluster distributions:")
        logger.info("  Cluster  |  Analysis  |  Evaluation")
        logger.info("  --------|------------|------------")
        for label in range(optimal_n):
            # Find index of this label in the analysis set
            analysis_idx = np.where(unique_labels == label)[0]
            analysis_count = counts[analysis_idx[0]] if len(analysis_idx) > 0 else 0
            analysis_pct = (analysis_count / len(all_labels)) * 100 if len(all_labels) > 0 else 0
            
            # Find index of this label in the evaluation set
            eval_idx = np.where(eval_unique_labels == label)[0]
            eval_count = eval_counts[eval_idx[0]] if len(eval_idx) > 0 else 0
            eval_pct = (eval_count / len(eval_labels)) * 100 if len(eval_labels) > 0 else 0
            
            logger.info(f"  {label}       |  {analysis_pct:.1f}%     |  {eval_pct:.1f}%")
        
        # Calculate silhouette scores if possible
        try:
            from sklearn.metrics import silhouette_score
            
            # We need a sample of data for silhouette calculation
            # This might be memory intensive for large datasets
            sample_size = min(10000, len(evaluation_indices))
            sample_indices = np.random.choice(evaluation_indices, sample_size, replace=False)
            
            # Load the sample
            sample_loader = DiskBundleLoader(
                bundle_info=bundle_info,
                indices=sample_indices,
                batch_size=sample_size,
                shuffle=False,
                use_normalized=True
            )
            
            # Get the data
            sample_data = next(iter(sample_loader))
            
            # If 3D, flatten
            if sample_data.ndim == 3:
                sample_data = sample_data.mean(axis=1)
            
            # Handle NaN values
            nan_count = np.isnan(sample_data).sum()
            if nan_count > 0:
                logger.warning(f"Found {nan_count} NaN values in evaluation data - applying imputation")
                # Load the imputer if it exists
                imputer_path = os.path.join(dirs["models"], "imputer.joblib")
                if os.path.exists(imputer_path):
                    imputer = joblib.load(imputer_path)
                    sample_data = imputer.transform(sample_data)
                else:
                    # Create a new imputer if needed
                    imputer = SimpleImputer(strategy='mean')
                    sample_data = imputer.fit_transform(sample_data)
            
            # Predict labels
            sample_labels = trainer.model.predict(sample_data)
            
            # Calculate silhouette score
            sil_score = silhouette_score(sample_data, sample_labels)
            logger.info(f"Silhouette Score on evaluation data: {sil_score:.4f}")
            
            # Add to summary dict
            eval_scores = {"silhouette_score": sil_score}
            
            # Free memory
            del sample_data, sample_loader
            gc.collect()
            
        except Exception as e:
            logger.warning(f"Could not calculate silhouette score: {str(e)}")
            eval_scores = {}
        
        # Calculate cluster evaluation metrics
        metrics = trainer.evaluate(sample_data, sample_labels)
        logger.info("Cluster evaluation metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
            eval_scores[metric] = value
        
        # Save evaluation metrics
        eval_path = os.path.join(dirs["base"], "clustering_evaluation.joblib")
        joblib.dump(eval_scores, eval_path)
        logger.info(f"Saved evaluation metrics to {eval_path}")
        
    except Exception as e:
        logger.error(f"Error in cluster evaluation: {str(e)}")
        logger.exception("Stack trace:")
    
    logger.info(f"Step 8 completed in {time.time() - step_start:.2f} seconds")
    
    #---------------------------------------------------------------------------
    # Pipeline Summary
    #---------------------------------------------------------------------------
    total_time = time.time() - start_time
    
    print_section_header("EEG PROCESSING PIPELINE COMPLETE")
    logger.info(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info(f"Results saved to {args.output_dir}")
    logger.info("\nSummary:")
    logger.info(f"  - Bundles created/loaded: {bundle_info['total_bundles']}")
    logger.info(f"  - Number of clusters: {optimal_n}")
    
    if eval_scores:
        logger.info("  - Evaluation scores:")
        for metric, value in eval_scores.items():
            logger.info(f"    * {metric}: {value:.4f}")
    
    logger.info("\nOutput Files:")
    logger.info(f"  - Bundle info: {os.path.join(bundle_dir, 'bundle_info.joblib')}")
    logger.info(f"  - Metadata: {os.path.join(dirs['base'], 'metadata_with_clusters.csv')}")
    logger.info(f"  - Data quality reports: {dirs['plots']}/sample_data_quality.png")
    logger.info(f"  - Trained model: {os.path.join(dirs['models'], 'unsupervised_model.joblib')}")
    logger.info(f"  - Visualizations: {dirs['plots']}")
    logger.info(f"  - Log file: {log_file}")
    
    logger.info("\nNext steps:")
    logger.info("  1. Review the cluster visualizations to understand the patterns")
    logger.info("  2. Analyze clusters in relation to EEG channel characteristics")
    logger.info("  3. Use clusters for supervised learning or further analysis")

if __name__ == "__main__":
    main() 