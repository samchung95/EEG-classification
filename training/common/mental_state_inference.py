import numpy as np
import pandas as pd
import os
import joblib
import time
import matplotlib.pyplot as plt
from collections import deque
import gc
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import traceback

# Import necessary modules
from .helpers import (
    preprocess_eeg_data, engineer_eeg_features,
    normalize_bundles, DiskBundleLoader, logger
)

def label_clusters_by_mental_state(cluster_profiles, feature_names=None, method='feature_based'):
    """
    Label clusters as 'focused' or 'relaxed' based on their characteristics
    
    Parameters:
    cluster_profiles (DataFrame): Mean values of each feature for each cluster
    feature_names (list): Names of features to consider
    method (str): Method to use for labeling ('feature_based', 'spectral_ratio')
    
    Returns:
    dict: Mapping of cluster IDs to mental states ('focused' or 'relaxed')
    """
    cluster_labels = {}
    
    # If feature_names is None, use all features
    if feature_names is None:
        feature_names = cluster_profiles.columns
    
    # Get all cluster IDs
    clusters = cluster_profiles.index.tolist()
    
    # Feature-based method (more robust with engineered features)
    if method == 'feature_based':
        # Define features known to relate to focus and relaxation
        focus_indicators = [col for col in feature_names 
                            if any(term in col.lower() for term in 
                                ['beta', 'focus', 'attention', 'concentration'])]
        
        relaxed_indicators = [col for col in feature_names 
                             if any(term in col.lower() for term in 
                                ['alpha', 'theta', 'relax', 'meditation', 'alphabeta_ratio'])]
        
        # Compute focus and relaxation scores for each cluster
        focus_scores = {}
        relaxed_scores = {}
        
        # First, calculate z-scores of each feature relative to overall mean
        overall_means = cluster_profiles.mean()
        overall_stds = cluster_profiles.std()
        
        for cluster in clusters:
            # Calculate relative strength of focus vs relaxation indicators
            cluster_means = cluster_profiles.loc[cluster]
            
            # Z-scores (how many std devs above/below overall mean)
            z_scores = (cluster_means - overall_means) / overall_stds
            
            # Sum z-scores for focus and relaxation indicators
            focus_score = sum(z_scores[col] for col in focus_indicators if col in z_scores) / max(len(focus_indicators), 1)
            relaxed_score = sum(z_scores[col] for col in relaxed_indicators if col in z_scores) / max(len(relaxed_indicators), 1)
            
            focus_scores[cluster] = focus_score
            relaxed_scores[cluster] = relaxed_score
            
            # Assign label based on scores
            if focus_score > relaxed_score:
                cluster_labels[cluster] = 'focused'
            else:
                cluster_labels[cluster] = 'relaxed'
                
            print(f"Cluster {cluster}: Focus score={focus_score:.3f}, Relaxation score={relaxed_score:.3f} → {cluster_labels[cluster]}")
    
    # Spectral ratio method (using raw band powers)
    elif method == 'spectral_ratio':
        # Focus typically has higher beta/alpha ratio
        # Relaxation typically has higher alpha/beta ratio
        
        for cluster in clusters:
            alpha_power = 0
            beta_power = 0
            
            # Try to find alpha and beta power features
            for col in feature_names:
                if 'alpha' in col.lower() and 'power' in col.lower():
                    alpha_power = cluster_profiles.loc[cluster, col]
                if 'beta' in col.lower() and 'power' in col.lower():
                    beta_power = cluster_profiles.loc[cluster, col]
            
            # If we found both bands
            if alpha_power > 0 and beta_power > 0:
                alpha_beta_ratio = alpha_power / beta_power
                beta_alpha_ratio = beta_power / alpha_power
                
                if beta_alpha_ratio > alpha_beta_ratio:
                    cluster_labels[cluster] = 'focused'
                else:
                    cluster_labels[cluster] = 'relaxed'
                    
                print(f"Cluster {cluster}: Alpha/Beta ratio={alpha_beta_ratio:.3f}, "
                      f"Beta/Alpha ratio={beta_alpha_ratio:.3f} → {cluster_labels[cluster]}")
            else:
                # Default to feature-based as fallback
                print(f"Could not find alpha/beta power for cluster {cluster}. Using feature-based method.")
                cluster_labels[cluster] = label_clusters_by_mental_state(
                    cluster_profiles.loc[[cluster]], feature_names, 'feature_based'
                )[cluster]
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Count and report labels
    counts = {'focused': 0, 'relaxed': 0}
    for state in cluster_labels.values():
        counts[state] += 1
    
    print(f"\nCluster labeling summary: {counts['focused']} focused, {counts['relaxed']} relaxed")
    
    return cluster_labels

class EEGRealTimePredictor:
    """
    Class for real-time EEG mental state prediction using trained clustering model
    """
    
    def __init__(self, model_path, buffer_size=30, step_size=5, 
                 feature_columns=None, mental_state_mapping=None):
        """
        Initialize the predictor
        
        Parameters:
        model_path (str): Path to directory containing saved models/parameters
        buffer_size (int): Size of the data buffer (number of samples)
        step_size (int): Step size for sliding window
        feature_columns (list): List of feature columns to use
        mental_state_mapping (dict): Mapping of cluster IDs to mental states
        """
        self.model_path = model_path
        self.buffer_size = buffer_size
        self.step_size = step_size
        self.feature_columns = feature_columns
        
        # Load the clustering model
        model_file = os.path.join(model_path, 'kmeans_clustering.joblib')
        if os.path.exists(model_file):
            self.clustering_model = joblib.load(model_file)
        else:
            # Try other clustering models
            for method in ['gmm', 'dbscan', 'hierarchical']:
                model_file = os.path.join(model_path, f'{method}_clustering.joblib')
                if os.path.exists(model_file):
                    self.clustering_model = joblib.load(model_file)
                    break
            else:
                raise FileNotFoundError(f"No clustering model found in {model_path}")
        
        # Load normalization parameters
        try:
            self.normalization_params = joblib.load(os.path.join(model_path, 'normalization_params.joblib'))
        except:
            print("Normalization parameters not found. Using default normalization.")
            self.normalization_params = None
        
        # Set mental state mapping
        self.mental_state_mapping = mental_state_mapping
        if self.mental_state_mapping is None:
            # Try to load from disk
            try:
                self.mental_state_mapping = joblib.load(os.path.join(model_path, 'mental_state_mapping.joblib'))
            except:
                print("Mental state mapping not found. Using default mapping (0=relaxed, 1=focused).")
                self.mental_state_mapping = {0: 'relaxed', 1: 'focused'}
        
        # Initialize buffer as a deque
        self.buffer = deque(maxlen=buffer_size)
        
        # Track predictions for smoothing
        self.recent_predictions = deque(maxlen=5)
        
        # Metrics for monitoring prediction quality
        self.prediction_times = []
        self.confidence_scores = []
        
        print(f"EEG real-time predictor initialized with buffer size {buffer_size}")
        print(f"Mental state mapping: {self.mental_state_mapping}")
    
    def add_sample(self, sample_data):
        """
        Add a new sample to the buffer
        
        Parameters:
        sample_data (dict or pd.Series): New EEG data sample
        
        Returns:
        bool: True if buffer is full, False otherwise
        """
        # Convert to pandas Series if it's a dict
        if isinstance(sample_data, dict):
            sample_data = pd.Series(sample_data)
        
        # Add to buffer
        self.buffer.append(sample_data)
        
        # Return True if buffer is full
        return len(self.buffer) >= self.buffer_size
    
    def get_buffer_as_dataframe(self):
        """
        Convert the current buffer to a pandas DataFrame
        
        Returns:
        pd.DataFrame: DataFrame containing buffer data
        """
        # Convert buffer to DataFrame
        df = pd.DataFrame(list(self.buffer))
        
        # Add a timestamp index if doesn't exist
        if 'TimeStamp' not in df.columns:
            df['TimeStamp'] = pd.date_range(
                end=pd.Timestamp.now(), 
                periods=len(df), 
                freq='100ms'  # Assuming 10 Hz sampling rate - adjust as needed
            )
        
        return df
    
    def normalize_bundle(self, bundle):
        """
        Normalize a bundle using training parameters
        
        Parameters:
        bundle (ndarray): Bundle to normalize
        
        Returns:
        ndarray: Normalized bundle
        """
        # If we have saved parameters, use them
        if self.normalization_params is not None:
            # Extract parameters
            means = self.normalization_params['means']
            stds = self.normalization_params['stds']
            
            # Apply normalization (z-score)
            if bundle.ndim == 3:  # 3D bundle (samples, time steps, features)
                # Apply along samples and time steps (axis=(0,1))
                normalized = (bundle - means) / stds
            else:  # 2D bundle (samples, features)
                normalized = (bundle - means) / stds
        else:
            # Otherwise, normalize using default method
            normalized = normalize_bundles(bundle, normalization='per_feature')
        
        return normalized
    
    def predict(self):
        """
        Make a prediction based on the current buffer
        
        Returns:
        dict: Prediction results
        """
        if len(self.buffer) < self.buffer_size:
            return {
                'state': None,
                'cluster': None,
                'confidence': 0.0,
                'buffer_fill': len(self.buffer) / self.buffer_size
            }
        
        start_time = time.time()
        
        try:
            # Step 1: Convert buffer to DataFrame
            df_buffer = self.get_buffer_as_dataframe()
            
            # Step 2: Preprocess the data
            df_clean = preprocess_eeg_data(df_buffer)
            
            # Step 3: Engineer features
            df_features = engineer_eeg_features(df_clean)
            
            # Step 4: Create a bundle (already in correct shape for the model)
            if self.feature_columns is not None:
                available_columns = [col for col in self.feature_columns if col in df_features.columns]
                if len(available_columns) < len(self.feature_columns):
                    missing = set(self.feature_columns) - set(available_columns)
                    print(f"Warning: Missing features: {missing}")
                feature_data = df_features[available_columns].values
            else:
                # Use all numeric columns
                numeric_cols = df_features.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
                feature_data = df_features[numeric_cols].values
            
            # Reshape if needed
            if feature_data.ndim == 1:
                feature_data = feature_data.reshape(1, -1)
            
            # Step 5: Normalize the data
            normalized_data = self.normalize_bundle(feature_data)
            
            # If 3D, take mean across time dimension
            if normalized_data.ndim == 3:
                normalized_data = normalized_data.mean(axis=1)
            
            # Step 6: Predict cluster
            cluster = self.clustering_model.predict(normalized_data)[0]
            
            # Step 7: Map to mental state
            mental_state = self.mental_state_mapping.get(cluster, 'unknown')
            
            # Add to recent predictions for smoothing
            self.recent_predictions.append(mental_state)
            
            # Simple majority voting for smoothing
            if len(self.recent_predictions) >= 3:
                counts = {}
                for pred in self.recent_predictions:
                    if pred not in counts:
                        counts[pred] = 0
                    counts[pred] += 1
                
                # Find the most common prediction
                smoothed_state = max(counts, key=counts.get)
                confidence = counts[smoothed_state] / len(self.recent_predictions)
            else:
                smoothed_state = mental_state
                confidence = 0.5  # Lower confidence when we don't have enough samples for smoothing
            
            # Track metrics
            self.prediction_times.append(time.time() - start_time)
            self.confidence_scores.append(confidence)
            
            # Return results
            return {
                'state': smoothed_state,
                'raw_state': mental_state,
                'cluster': int(cluster),
                'confidence': confidence,
                'prediction_time': self.prediction_times[-1],
                'buffer_fill': 1.0
            }
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return {
                'state': 'error',
                'cluster': None,
                'confidence': 0.0,
                'error': str(e),
                'buffer_fill': len(self.buffer) / self.buffer_size
            }
    
    def clear_buffer(self):
        """Clear the buffer"""
        self.buffer.clear()
        self.recent_predictions.clear()
    
    def get_metrics(self):
        """Return prediction metrics"""
        if not self.prediction_times:
            return {
                'avg_prediction_time': 0,
                'avg_confidence': 0,
                'total_predictions': 0
            }
            
        return {
            'avg_prediction_time': np.mean(self.prediction_times),
            'avg_confidence': np.mean(self.confidence_scores),
            'total_predictions': len(self.prediction_times)
        }

def generate_mental_state_mapping(model_dir, data_sample_path=None, method='feature_based'):
    """
    Generate a mapping from cluster IDs to mental states
    
    Parameters:
    model_dir (str): Directory containing clustering model
    data_sample_path (str): Path to sample data for analysis (optional)
    method (str): Method to use for mapping ('feature_based', 'spectral_ratio')
    
    Returns:
    dict: Mapping of cluster IDs to mental states
    """
    # Try to load cluster profiles if they exist
    profile_path = os.path.join(model_dir, 'cluster_profiles.joblib')
    
    if os.path.exists(profile_path):
        print(f"Loading cluster profiles from {profile_path}")
        cluster_profiles = joblib.load(profile_path)
        
    elif data_sample_path is not None:
        # We need to generate profiles from sample data
        from .unsupervised import analyze_cluster_characteristics
        
        print(f"Generating cluster profiles from sample data: {data_sample_path}")
        
        # Load clustering model
        model_files = [f for f in os.listdir(model_dir) if f.endswith('_clustering.joblib')]
        if not model_files:
            raise FileNotFoundError(f"No clustering model found in {model_dir}")
        
        clustering_model = joblib.load(os.path.join(model_dir, model_files[0]))
        
        # Load and preprocess sample data
        from .helpers import load_eeg_data, preprocess_eeg_data, engineer_eeg_features
        
        if os.path.isdir(data_sample_path):
            file_dfs, _ = load_eeg_data(directory_path=data_sample_path)
        else:
            file_dfs, _ = load_eeg_data(single_file=data_sample_path)
        
        # Take a sample file
        sample_file_id = list(file_dfs.keys())[0]
        sample_df = file_dfs[sample_file_id]
        
        # Preprocess and engineer features
        clean_df = preprocess_eeg_data(sample_df)
        feature_df = engineer_eeg_features(clean_df)
        
        # Get feature data
        feature_data = feature_df.select_dtypes(include=['float64', 'float32', 'int64', 'int32'])
        
        # Predict clusters
        clusters = clustering_model.predict(feature_data)
        
        # Generate profiles
        cluster_profiles = analyze_cluster_characteristics(
            feature_data.values, clusters, feature_data.columns
        )
        
        # Save profiles for future use
        joblib.dump(cluster_profiles, profile_path)
    
    else:
        # No profiles and no sample data, use default mapping
        print("No cluster profiles or sample data available. Using default mapping.")
        return {0: 'relaxed', 1: 'focused'}
    
    # Generate mapping
    mental_state_mapping = label_clusters_by_mental_state(
        cluster_profiles, method=method
    )
    
    # Save mapping
    mapping_path = os.path.join(model_dir, 'mental_state_mapping.joblib')
    joblib.dump(mental_state_mapping, mapping_path)
    
    return mental_state_mapping

def real_time_eeg_demo(predictor, data_path, duration_seconds=60, sample_interval=0.1):
    """
    Demo function to simulate real-time EEG prediction using stored data
    
    Parameters:
    predictor (EEGRealTimePredictor): Initialized predictor
    data_path (str): Path to data file or directory
    duration_seconds (int): Duration of the simulation
    sample_interval (float): Time between samples in seconds
    
    Returns:
    dict: Prediction results
    """
    from .helpers import load_eeg_data
    
    print(f"Starting real-time EEG demo using data from {data_path}")
    
    # Load data
    if os.path.isdir(data_path):
        file_dfs, _ = load_eeg_data(directory_path=data_path)
        # Use first file
        sample_df = list(file_dfs.values())[0]
    else:
        _, sample_df = load_eeg_data(single_file=data_path)
    
    # Reset predictor
    predictor.clear_buffer()
    
    # Track results
    results = {
        'time': [],
        'state': [],
        'cluster': [],
        'confidence': []
    }
    
    # Number of samples to process
    num_samples = min(int(duration_seconds / sample_interval), len(sample_df))
    
    print(f"Processing {num_samples} samples over {duration_seconds} seconds")
    
    # Process samples
    for i in range(num_samples):
        # Get sample
        sample = sample_df.iloc[i]
        
        # Add to predictor
        predictor.add_sample(sample)
        
        # Make prediction if buffer is full
        if len(predictor.buffer) >= predictor.buffer_size:
            prediction = predictor.predict()
            
            # Store results
            results['time'].append(i * sample_interval)
            results['state'].append(prediction['state'])
            results['cluster'].append(prediction['cluster'])
            results['confidence'].append(prediction['confidence'])
            
            # Print update every second
            if i % int(1 / sample_interval) == 0:
                print(f"Time: {i * sample_interval:.1f}s, State: {prediction['state']}, "
                      f"Confidence: {prediction['confidence']:.2f}")
        
        # Simulate processing time
        time.sleep(sample_interval)
    
    # Display summary
    metrics = predictor.get_metrics()
    print("\nDemo complete!")
    print(f"Total predictions: {metrics['total_predictions']}")
    print(f"Average prediction time: {metrics['avg_prediction_time'] * 1000:.2f}ms")
    print(f"Average confidence: {metrics['avg_confidence']:.2f}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    # Convert states to numeric
    state_map = {'focused': 1, 'relaxed': 0, 'error': -1, None: -1}
    numeric_states = [state_map.get(s, -1) for s in results['state']]
    
    plt.subplot(2, 1, 1)
    plt.plot(results['time'], numeric_states, 'o-')
    plt.ylabel('Mental State')
    plt.yticks([0, 1], ['Relaxed', 'Focused'])
    plt.title('Real-time EEG Mental State Prediction')
    
    plt.subplot(2, 1, 2)
    plt.plot(results['time'], results['confidence'])
    plt.xlabel('Time (s)')
    plt.ylabel('Confidence')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    return results 

class MentalStateInference:
    """
    Class for inferring mental states from EEG data.
    
    This class provides methods for loading models, preprocessing data,
    and making predictions about mental states based on EEG signals.
    """
    
    def __init__(self, model_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the mental state inference engine.
        
        Args:
            model_path: Path to the trained model file
            config: Configuration dictionary with parameters for the inference
        """
        self.model_path = model_path
        self.config = config or {}
        self.model = None
        self.is_initialized = False
        
        logger.info("Mental State Inference engine initialized")
        
    def load_model(self) -> bool:
        """
        Load the trained model from disk.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if not self.model_path:
            logger.error("No model path specified")
            return False
            
        try:
            # Model loading code would go here
            logger.info(f"Model loaded from {self.model_path}")
            self.is_initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            logger.debug(traceback.format_exc())
            return False
            
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the input data for model inference.
        
        Args:
            data: Raw EEG data
            
        Returns:
            Preprocessed data ready for model inference
        """
        logger.debug(f"Preprocessing data with shape {data.shape}")
        try:
            # Data preprocessing would go here
            return data
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
            
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions on the input data.
        
        Args:
            data: Preprocessed EEG data
            
        Returns:
            Dictionary with prediction results
            
        Raises:
            RuntimeError: If the model is not initialized
        """
        if not self.is_initialized:
            error_msg = "Model not initialized. Call load_model() first."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        try:
            # Prediction code would go here
            logger.info(f"Predictions made on data with shape {data.shape}")
            return {"status": "success", "results": []}
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            logger.debug(traceback.format_exc())
            return {"status": "error", "message": str(e)} 