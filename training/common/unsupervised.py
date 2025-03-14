import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import umap
import joblib
import os
import time
import gc
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
import traceback
from sklearn.base import BaseEstimator

# Import helper functions
from .helpers import (
    load_eeg_data, preprocess_eeg_data, engineer_eeg_features,
    create_coherent_time_series_bundles, normalize_bundles,
    split_bundles_train_test, logger
)

class UnsupervisedModelTrainer:
    """
    Class for training unsupervised machine learning models on EEG data.
    
    This class provides methods for data preparation, dimensionality reduction,
    clustering, and evaluation of unsupervised models.
    """
    
    def __init__(self, 
                 model_type: str = 'kmeans', 
                 random_state: int = 42,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the unsupervised model trainer.
        
        Args:
            model_type: Type of model to train ('kmeans', 'dbscan', etc.)
            random_state: Random seed for reproducibility
            config: Configuration dictionary with training parameters
        """
        self.model_type = model_type
        self.random_state = random_state
        self.config = config or {}
        self.model = None
        self.scaler = None
        self.dim_reducer = None
        
        logger.info(f"Initialized UnsupervisedModelTrainer with model_type={model_type}")
        
    def preprocess_data(self, 
                        data: pd.DataFrame,
                        feature_columns: Optional[List[str]] = None,
                        scale: bool = True) -> np.ndarray:
        """
        Preprocess data for unsupervised learning.
        
        Args:
            data: DataFrame containing features
            feature_columns: List of column names to use as features (if None, use all)
            scale: Whether to standardize the features
            
        Returns:
            Preprocessed features as numpy array
        """
        logger.info(f"Preprocessing data with shape {data.shape}")
        
        try:
            # Extract features
            if feature_columns is None:
                feature_columns = data.columns.tolist()
                
            X = data[feature_columns].values
            
            # Scale features if requested
            if scale:
                self.scaler = StandardScaler()
                X = self.scaler.fit_transform(X)
                logger.info("Data scaled using StandardScaler")
                
            logger.info(f"Data preprocessed: shape={X.shape}")
            return X
        
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
            
    def reduce_dimensions(self, 
                          X: np.ndarray, 
                          n_components: int = 2,
                          method: str = 'pca') -> np.ndarray:
        """
        Reduce dimensionality of the data.
        
        Args:
            X: Input features
            n_components: Number of components to keep
            method: Dimensionality reduction method ('pca', etc.)
            
        Returns:
            Reduced feature matrix
        """
        logger.info(f"Reducing dimensions from {X.shape[1]} to {n_components} using {method}")
        
        try:
            if method == 'pca':
                self.dim_reducer = PCA(n_components=n_components, random_state=self.random_state)
                X_reduced = self.dim_reducer.fit_transform(X)
                
                explained_variance = np.sum(self.dim_reducer.explained_variance_ratio_)
                logger.info(f"PCA explained variance: {explained_variance:.4f}")
            else:
                raise ValueError(f"Unsupported dimensionality reduction method: {method}")
                
            logger.info(f"Dimensions reduced: shape={X_reduced.shape}")
            return X_reduced
            
        except Exception as e:
            logger.error(f"Error reducing dimensions: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
            
    def create_model(self, n_clusters: Optional[int] = None) -> BaseEstimator:
        """
        Create a model instance based on the specified model type.
        
        Args:
            n_clusters: Number of clusters (for applicable algorithms)
        
        Returns:
            A scikit-learn compatible model instance
        """
        try:
            if self.model_type == 'kmeans':
                if n_clusters is None:
                    n_clusters = self.config.get('n_clusters', 3)
                model = KMeans(
                    n_clusters=n_clusters,
                    random_state=self.random_state,
                    n_init=self.config.get('n_init', 10)
                )
            elif self.model_type == 'dbscan':
                model = DBSCAN(
                    eps=self.config.get('eps', 0.5),
                    min_samples=self.config.get('min_samples', 5)
                )
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
            logger.info(f"Created model of type {self.model_type}")
            return model
            
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
            
    def find_optimal_clusters(self, 
                              X: np.ndarray, 
                              min_clusters: int = 2, 
                              max_clusters: int = 10) -> Tuple[int, float]:
        """
        Find the optimal number of clusters using silhouette score.
        
        Args:
            X: Input features
            min_clusters: Minimum number of clusters to try
            max_clusters: Maximum number of clusters to try
            
        Returns:
            Tuple of (optimal_n_clusters, best_silhouette_score)
        """
        if self.model_type != 'kmeans':
            logger.warning(f"Finding optimal clusters is only supported for kmeans, not {self.model_type}")
            return self.config.get('n_clusters', 3), 0.0
            
        logger.info(f"Finding optimal number of clusters between {min_clusters} and {max_clusters}")
        
        try:
            best_score = -1
            best_n = min_clusters
            
            for n in range(min_clusters, max_clusters + 1):
                kmeans = KMeans(n_clusters=n, random_state=self.random_state, n_init=10)
                labels = kmeans.fit_predict(X)
                
                # Skip if only one label is assigned (degenerate case)
                if len(np.unique(labels)) < 2:
                    continue
                    
                score = silhouette_score(X, labels)
                logger.debug(f"Clusters: {n}, Silhouette Score: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_n = n
                    
            logger.info(f"Optimal number of clusters: {best_n} (silhouette score: {best_score:.4f})")
            return best_n, best_score
            
        except Exception as e:
            logger.error(f"Error finding optimal clusters: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
            
    def train(self, X: np.ndarray, auto_clusters: bool = False) -> np.ndarray:
        """
        Train the model on the provided data.
        
        Args:
            X: Training features
            auto_clusters: Whether to automatically determine optimal clusters
            
        Returns:
            Cluster assignments for each sample
        """
        logger.info(f"Training {self.model_type} model on data with shape {X.shape}")
        
        try:
            # Find optimal clusters if requested
            n_clusters = None
            if auto_clusters and self.model_type == 'kmeans':
                n_clusters, _ = self.find_optimal_clusters(X)
            
            # Create and train model
            model = self.create_model(n_clusters)
            if hasattr(model, 'fit_predict'):
                labels = model.fit_predict(X)
            else:
                model.fit(X)
                labels = model.labels_
                
            self.model = model
            
            # Log cluster statistics
            unique_labels = np.unique(labels)
            n_clusters_found = len(unique_labels)
            if -1 in unique_labels:  # DBSCAN noise points
                n_clusters_found -= 1
                n_noise = np.sum(labels == -1)
                logger.info(f"Found {n_clusters_found} clusters and {n_noise} noise points")
            else:
                logger.info(f"Found {n_clusters_found} clusters")
                
            # Log sample distribution in clusters
            for label in unique_labels:
                if label != -1:  # Skip noise
                    count = np.sum(labels == label)
                    logger.info(f"Cluster {label}: {count} samples ({count/len(labels)*100:.1f}%)")
                
            return labels
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
            
    def evaluate(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the clustering results.
        
        Args:
            X: Features used for clustering
            labels: Cluster assignments
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating clustering results")
        
        try:
            metrics = {}
            
            # Calculate silhouette score if there are at least 2 clusters and no noise
            unique_labels = np.unique(labels)
            if len(unique_labels) >= 2 and -1 not in unique_labels:
                silhouette = silhouette_score(X, labels)
                metrics['silhouette_score'] = silhouette
                logger.info(f"Silhouette score: {silhouette:.4f}")
            
            # For KMeans, calculate inertia
            if self.model_type == 'kmeans' and hasattr(self.model, 'inertia_'):
                metrics['inertia'] = self.model.inertia_
                logger.info(f"Inertia: {self.model.inertia_:.4f}")
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            logger.debug(traceback.format_exc())
            return {}
            
    def save_model(self, filepath: str) -> bool:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path where to save the model
            
        Returns:
            True if successful, False otherwise
        """
        if self.model is None:
            error_msg = "No trained model to save. Call train() first."
            logger.error(error_msg)
            return False
            
        try:
            import joblib
            
            # Create a dictionary with all components
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'dim_reducer': self.dim_reducer,
                'model_type': self.model_type,
                'config': self.config
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            logger.debug(traceback.format_exc())
            return False
            
    def load_model(self, filepath: str) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import joblib
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.dim_reducer = model_data['dim_reducer']
            self.model_type = model_data['model_type']
            self.config = model_data['config']
            
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.debug(traceback.format_exc())
            return False

def benchmark_clustering(X, methods=['kmeans', 'dbscan']):
    """
    Benchmark CPU vs GPU performance for clustering algorithms
    
    Parameters:
    X (ndarray): Feature matrix to cluster
    methods (list): List of clustering methods to benchmark
    
    Returns:
    dict: Benchmark results
    """
    results = {}
    
    for method in methods:
        if method not in ['kmeans', 'dbscan']:
            print(f"Skipping {method} - no GPU implementation available")
            continue
            
        print(f"\nBenchmarking {method.upper()}:")
        
        # CPU timing
        cpu_start = time.time()
        _ = cluster_data(X, methods=[method], use_gpu=False)
        cpu_time = time.time() - cpu_start
        print(f"CPU time: {cpu_time:.2f} seconds")
        
        try:
            # GPU timing
            gpu_start = time.time()
            _ = cluster_data(X, methods=[method], use_gpu=True)
            gpu_time = time.time() - gpu_start
            print(f"GPU time: {gpu_time:.2f} seconds")
            
            speedup = cpu_time / gpu_time
            print(f"Speedup with GPU: {speedup:.2f}x")
            
            results[method] = {
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': speedup
            }
        except Exception as e:
            print(f"GPU benchmark failed: {e}")
            results[method] = {
                'cpu_time': cpu_time,
                'gpu_time': None,
                'speedup': None
            }
    
    return results

def reduce_dimensions(X_scaled, method='pca', n_components=2, use_gpu=False):
    """
    Reduce dimensions of the data for visualization and better clustering
    
    Parameters:
    X_scaled (ndarray): Scaled feature matrix
    method (str): Dimensionality reduction method ('pca', 'tsne', or 'umap')
    n_components (int): Number of components to reduce to
    use_gpu (bool): Whether to attempt using GPU acceleration if available
    
    Returns:
    ndarray: Reduced features
    object: Fitted reducer object
    """
    start_time = time.time()
    
    # Try GPU implementation if requested
    if use_gpu:
        try:
            if method == 'pca':
                # Using cuML for GPU-accelerated PCA
                from cuml.decomposition import PCA as cuPCA
                
                reducer = cuPCA(n_components=n_components)
                X_reduced = reducer.fit_transform(X_scaled)
                
                # Convert back to CPU if needed
                try:
                    X_reduced = X_reduced.get()  # If cupy array
                except:
                    try:
                        X_reduced = X_reduced.to_pandas().values  # If cuDF DataFrame
                    except:
                        pass  # Already in a compatible format
                
                print(f"GPU PCA completed in {time.time() - start_time:.2f} seconds")
                return X_reduced, reducer
                
            elif method == 'tsne':
                # Using cuML for GPU-accelerated t-SNE
                from cuml.manifold import TSNE as cuTSNE
                
                reducer = cuTSNE(n_components=n_components, random_state=42)
                X_reduced = reducer.fit_transform(X_scaled)
                
                # Convert back to CPU if needed
                try:
                    X_reduced = X_reduced.get()
                except:
                    try:
                        X_reduced = X_reduced.to_pandas().values
                    except:
                        pass
                
                print(f"GPU t-SNE completed in {time.time() - start_time:.2f} seconds")
                return X_reduced, reducer
                
            elif method == 'umap':
                # Using cuML for GPU-accelerated UMAP
                from cuml.manifold import UMAP as cuUMAP
                
                reducer = cuUMAP(n_components=n_components, random_state=42)
                X_reduced = reducer.fit_transform(X_scaled)
                
                # Convert back to CPU if needed
                try:
                    X_reduced = X_reduced.get()
                except:
                    try:
                        X_reduced = X_reduced.to_pandas().values
                    except:
                        pass
                
                print(f"GPU UMAP completed in {time.time() - start_time:.2f} seconds")
                return X_reduced, reducer
                
        except ImportError:
            print(f"GPU libraries not available. Falling back to CPU implementation.")
        except Exception as e:
            print(f"Error with GPU implementation: {e}. Falling back to CPU.")
    
    # Fall back to CPU implementation
    if method == 'pca':
        reducer = PCA(n_components=n_components)
        X_reduced = reducer.fit_transform(X_scaled)
        print(f"PCA completed in {time.time() - start_time:.2f} seconds")
        print(f"Explained variance ratio: {reducer.explained_variance_ratio_}")
        
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
        X_reduced = reducer.fit_transform(X_scaled)
        print(f"t-SNE completed in {time.time() - start_time:.2f} seconds")
        
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        X_reduced = reducer.fit_transform(X_scaled)
        print(f"UMAP completed in {time.time() - start_time:.2f} seconds")
        
    else:
        raise ValueError("Method must be 'pca', 'tsne', or 'umap'")
    
    return X_reduced, reducer

def cluster_data(X, X_reduced=None, methods=['kmeans', 'gmm'], n_clusters=2, use_gpu=False):
    """
    Apply multiple clustering algorithms and evaluate them
    
    Parameters:
    X (ndarray): Feature matrix (original data, not reduced)
    X_reduced (ndarray): Dimensionality-reduced feature matrix for visualization
    methods (list): List of clustering methods to apply
    n_clusters (int): Number of clusters to find
    use_gpu (bool): Whether to use GPU-accelerated clustering if available
    
    Returns:
    dict: Dictionary containing cluster labels and evaluation metrics for each method
    """
    results = {}
    
    # If X_reduced not provided, use X for visualization
    if X_reduced is None:
        X_reduced = X
        if X_reduced.ndim > 2:
            # If X is a 3D tensor (bundles), take mean across time dimension
            X_reduced = X_reduced.mean(axis=1)
    
    for method in methods:
        print(f"\nRunning {method.upper()} clustering...")
        start_time = time.time()
        
        # Try GPU implementation if requested
        if use_gpu:
            try:
                if method == 'kmeans':
                    # Using cuML for GPU-accelerated K-means
                    from cuml.cluster import KMeans as cuKMeans
                    
                    clustering = cuKMeans(n_clusters=n_clusters, random_state=42)
                    labels = clustering.fit_predict(X)
                    
                    # Convert back to CPU if needed
                    try:
                        labels = labels.get()
                    except:
                        pass
                    
                    print(f"  GPU K-means completed in {time.time() - start_time:.2f} seconds")
                    
                elif method == 'dbscan':
                    # Using cuML for GPU-accelerated DBSCAN
                    from cuml.cluster import DBSCAN as cuDBSCAN
                    
                    clustering = cuDBSCAN(eps=0.5, min_samples=5)
                    labels = clustering.fit_predict(X)
                    
                    # Convert back to CPU if needed
                    try:
                        labels = labels.get()
                    except:
                        pass
                    
                    # DBSCAN can produce -1 labels for noise, set to max_label+1 for evaluation
                    if -1 in labels:
                        labels[labels == -1] = labels.max() + 1
                    
                    print(f"  GPU DBSCAN completed in {time.time() - start_time:.2f} seconds")
                    
                elif method == 'gmm':
                    # No cuML implementation for GMM yet, fall back to CPU
                    raise ImportError("No GPU implementation available for GMM")
                    
                elif method == 'hierarchical':
                    # No cuML implementation for hierarchical clustering yet, fall back to CPU
                    raise ImportError("No GPU implementation available for hierarchical clustering")
                    
                else:
                    raise ValueError(f"Unknown clustering method: {method}")
                
            except ImportError as e:
                print(f"  {e}. Falling back to CPU implementation.")
                use_gpu_for_this_method = False
            except Exception as e:
                print(f"  Error with GPU implementation: {e}. Falling back to CPU.")
                use_gpu_for_this_method = False
        else:
            use_gpu_for_this_method = False
        
        # Fall back to CPU implementation if GPU failed or wasn't requested
        if not use_gpu or not use_gpu_for_this_method:
            cpu_start_time = time.time()
            
            if method == 'kmeans':
                # K-means with k=n_clusters
                clustering = KMeans(n_clusters=n_clusters, random_state=42)
                labels = clustering.fit_predict(X)
                print(f"  CPU K-means completed in {time.time() - cpu_start_time:.2f} seconds")
                
            elif method == 'gmm':
                # Gaussian Mixture Model
                clustering = GaussianMixture(n_components=n_clusters, random_state=42)
                labels = clustering.fit_predict(X)
                print(f"  CPU GMM completed in {time.time() - cpu_start_time:.2f} seconds")
                
            elif method == 'dbscan':
                # DBSCAN - epsilon needs to be tuned for your specific data
                clustering = DBSCAN(eps=0.5, min_samples=5)
                labels = clustering.fit_predict(X)
                
                # DBSCAN can produce -1 labels for noise, set to max_label+1 for evaluation
                if -1 in labels:
                    labels[labels == -1] = labels.max() + 1
                
                print(f"  CPU DBSCAN completed in {time.time() - cpu_start_time:.2f} seconds")
                    
            elif method == 'hierarchical':
                # Hierarchical Clustering
                clustering = AgglomerativeClustering(n_clusters=n_clusters)
                labels = clustering.fit_predict(X)
                print(f"  CPU hierarchical clustering completed in {time.time() - cpu_start_time:.2f} seconds")
                
            else:
                raise ValueError(f"Unknown clustering method: {method}")
        
        # Evaluate clustering only if we have at least 2 clusters
        unique_labels = np.unique(labels)
        if len(unique_labels) >= 2:
            try:
                sil_score = silhouette_score(X, labels)
                db_score = davies_bouldin_score(X, labels)
                ch_score = calinski_harabasz_score(X, labels)
                
                eval_metrics = {
                    'silhouette_score': sil_score,
                    'davies_bouldin_score': db_score,
                    'calinski_harabasz_score': ch_score
                }
            except Exception as e:
                print(f"  Warning: Could not compute evaluation metrics. Error: {e}")
                eval_metrics = {
                    'silhouette_score': None,
                    'davies_bouldin_score': None,
                    'calinski_harabasz_score': None
                }
        else:
            eval_metrics = {
                'silhouette_score': None,
                'davies_bouldin_score': None,
                'calinski_harabasz_score': None
            }
            
        results[method] = {
            'labels': labels,
            'clustering': clustering,
            'eval_metrics': eval_metrics
        }
        
        # Print evaluation metrics
        print("  Evaluation metrics:")
        for metric, value in eval_metrics.items():
            if value is not None:
                print(f"    {metric}: {value:.4f}")
        
        # Visualize clusters (only if X_reduced is 2D)
        if X_reduced.shape[1] == 2:
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis', alpha=0.7)
            plt.title(f'Clustering with {method.upper()}')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.colorbar(scatter, label='Cluster')
            plt.tight_layout()
            plt.show()
        
    return results

def analyze_cluster_characteristics(X, labels, feature_names=None):
    """
    Analyze the characteristics of each cluster
    
    Parameters:
    X (ndarray): Feature matrix
    labels (ndarray): Cluster labels
    feature_names (list): Names of features (if available)
    
    Returns:
    DataFrame: Cluster profile summary
    """
    # If X is 3D (bundles), flatten time dimension for analysis
    if X.ndim == 3:
        # Take mean across time dimension
        X_flat = X.mean(axis=1)
    else:
        X_flat = X
    
    # Create DataFrame with features and cluster labels
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X_flat.shape[1])]
        
    df = pd.DataFrame(X_flat, columns=feature_names)
    df['Cluster'] = labels
    
    # Get mean values for each cluster
    cluster_profiles = df.groupby('Cluster').mean()
    
    # For each cluster, find which features are most distinctive
    unique_clusters = sorted(df['Cluster'].unique())
    
    for cluster in unique_clusters:
        # Calculate z-scores for this cluster's means compared to overall means
        overall_means = df[feature_names].mean()
        overall_stds = df[feature_names].std()
        
        cluster_means = cluster_profiles.loc[cluster]
        z_scores = (cluster_means - overall_means) / overall_stds
        
        # Find top distinctive features (highest absolute z-scores)
        top_features = z_scores.abs().sort_values(ascending=False).head(5)
        
        print(f"\nCluster {cluster} distinctive features:")
        for feat in top_features.index:
            direction = "higher" if z_scores[feat] > 0 else "lower"
            print(f"  {feat}: {z_scores[feat]:.2f} std {direction} than average")
        
        # Try to interpret if this is relaxed or focused state
        # Focus: higher Beta, lower Alpha/Beta ratio
        # Relaxed: higher Alpha, lower Beta, higher Alpha/Beta ratio
        focus_indicators = [col for col in feature_names if 'Beta' in col or 'FocusIndex' in col]
        relaxed_indicators = [col for col in feature_names if 'Alpha' in col or 'RelaxationIndex' in col or 'AlphaBeta_Ratio' in col]
        
        if focus_indicators and relaxed_indicators:
            focus_score = sum(z_scores[col] for col in focus_indicators if col in z_scores)
            relaxed_score = sum(z_scores[col] for col in relaxed_indicators if col in z_scores)
            
            if focus_score > relaxed_score:
                print(f"  Cluster {cluster} likely represents FOCUSED state")
            else:
                print(f"  Cluster {cluster} likely represents RELAXED state")
    
    return cluster_profiles

def run_clustering_workflow(data_path, output_dir='./models/clustering', 
                           bundle_size=30, step_size=15, 
                           reducer_method='umap', use_gpu=False,
                           cluster_methods=['kmeans', 'gmm']):
    """
    Run the complete unsupervised clustering workflow
    
    Parameters:
    data_path (str): Path to directory containing EEG CSV files or single CSV file
    output_dir (str): Directory to save models and results
    bundle_size (int): Size of time series bundles
    step_size (int): Step size for sliding window
    reducer_method (str): Dimensionality reduction method to use
    use_gpu (bool): Whether to attempt using GPU acceleration
    cluster_methods (list): Clustering methods to apply
    
    Returns:
    dict: Results of clustering workflow
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load the data
    print("\nStep 1: Loading EEG data...")
    if os.path.isdir(data_path):
        file_dfs, combined_df = load_eeg_data(directory_path=data_path)
    else:
        file_dfs, combined_df = load_eeg_data(single_file=data_path)
    
    # Step 2: Preprocess and engineer features for each file
    print("\nStep 2: Preprocessing and feature engineering...")
    processed_dfs = {}
    
    for file_id, df in file_dfs.items():
        # Preprocess
        df_clean = preprocess_eeg_data(df)
        
        # Engineer features
        df_features = engineer_eeg_features(df_clean)
        
        processed_dfs[file_id] = df_features
    
    # Step 3: Create coherent time series bundles
    print("\nStep 3: Creating time series bundles...")
    X_bundles, metadata = create_coherent_time_series_bundles(
        file_dfs=processed_dfs,
        bundle_size=bundle_size,
        step_size=step_size
    )
    
    # Step 4: Normalize bundles
    print("\nStep 4: Normalizing bundles...")
    X_normalized = normalize_bundles(X_bundles, normalization='per_feature')
    
    # Step 5: Reduce dimensions for visualization
    print("\nStep 5: Reducing dimensions with " + reducer_method.upper() + "...")
    # Flatten time dimension by taking mean across time steps
    X_flat = X_normalized.mean(axis=1) if X_normalized.ndim == 3 else X_normalized
    X_reduced, reducer = reduce_dimensions(
        X_flat, method=reducer_method, n_components=2, use_gpu=use_gpu
    )
    
    # Save the reducer
    try:
        joblib.dump(reducer, os.path.join(output_dir, f'{reducer_method}_reducer.joblib'))
    except:
        print(f"Warning: Could not save the {reducer_method} reducer. It may not be serializable.")
    
    # Step 6: Cluster the data
    print("\nStep 6: Clustering data...")
    clustering_results = cluster_data(
        X_flat, X_reduced, methods=cluster_methods, n_clusters=2, use_gpu=use_gpu
    )
    
    # Step 7: Analyze cluster characteristics
    print("\nStep 7: Analyzing cluster characteristics...")
    # Find the best clustering method based on silhouette score
    best_method = max(
        clustering_results,
        key=lambda m: clustering_results[m]['eval_metrics']['silhouette_score'] or -1
    )
    
    print(f"\nBest clustering method: {best_method}")
    
    # Analyze the best clustering
    best_labels = clustering_results[best_method]['labels']
    
    # Get feature names if they are in metadata
    if X_normalized.ndim == 3:
        # These are generic feature names since we flattened the time dimension
        feature_names = [f'Feature_{i}' for i in range(X_flat.shape[1])]
    else:
        feature_names = None
    
    cluster_profiles = analyze_cluster_characteristics(
        X_normalized, best_labels, feature_names
    )
    
    # Save the best clustering model
    try:
        joblib.dump(
            clustering_results[best_method]['clustering'],
            os.path.join(output_dir, f'{best_method}_clustering.joblib')
        )
        
        # Save labels for later use
        np.save(os.path.join(output_dir, 'cluster_labels.npy'), best_labels)
        
        # Save metadata with cluster labels
        metadata_with_clusters = metadata.copy()
        metadata_with_clusters['cluster'] = best_labels
        metadata_with_clusters.to_csv(os.path.join(output_dir, 'metadata_with_clusters.csv'), index=False)
        
    except Exception as e:
        print(f"Warning: Could not save clustering results. Error: {e}")
    
    return {
        'X_bundles': X_bundles,
        'X_normalized': X_normalized,
        'X_reduced': X_reduced,
        'metadata': metadata,
        'clustering_results': clustering_results,
        'best_method': best_method,
        'cluster_profiles': cluster_profiles
    }

def memory_efficient_clustering(bundle_info, max_samples=10000, methods=['kmeans'], n_clusters=2, use_gpu=False):
    """
    Run clustering on a subset of data from disk to avoid memory issues
    
    Parameters:
    bundle_info (dict): Bundle info from normalize_bundles_disk
    max_samples (int): Maximum number of samples to use for clustering
    methods (list): Clustering methods to use
    n_clusters (int): Number of clusters
    use_gpu (bool): Whether to use GPU acceleration
    
    Returns:
    dict: Clustering results
    """
    from .helpers import DiskBundleLoader
    
    print(f"\nPerforming memory-efficient clustering using max {max_samples} samples...")
    
    # Determine total bundles
    total_bundles = bundle_info['total_bundles']
    
    # Sample indices (either all or a subset)
    if max_samples >= total_bundles:
        indices = np.arange(total_bundles)
    else:
        indices = np.random.choice(total_bundles, max_samples, replace=False)
    
    print(f"Using {len(indices)} bundles for clustering")
    
    # Create data loader
    loader = DiskBundleLoader(
        bundle_info=bundle_info,
        indices=indices,
        batch_size=10000,  # Load in batches of 1000
        shuffle=False,
        use_normalized=True
    )
    
    # Load all sampled data (this is memory-intensive but necessary for clustering)
    # We're only using a subset of the data to make it manageable
    bundles = []
    for batch in loader:
        bundles.append(batch)
    
    # Concatenate all batches
    X = np.concatenate(bundles, axis=0)
    
    # Flatten time dimension by taking mean across time steps
    if X.ndim == 3:
        X_flat = X.mean(axis=1)
    else:
        X_flat = X
    
    print(f"Data shape for clustering: {X_flat.shape}")
    
    # Perform clustering
    clustering_results = {}
    
    for method in methods:
        print(f"\nRunning {method.upper()} clustering...")
        start_time = time.time()
        
        # Try GPU implementation if requested
        if use_gpu:
            try:
                if method == 'kmeans':
                    # Using cuML for GPU-accelerated K-means
                    from cuml.cluster import KMeans as cuKMeans
                    
                    clustering = cuKMeans(n_clusters=n_clusters, random_state=42)
                    labels = clustering.fit_predict(X_flat)
                    
                    # Convert back to CPU if needed
                    try:
                        labels = labels.get()
                    except:
                        pass
                    
                    print(f"  GPU K-means completed in {time.time() - start_time:.2f} seconds")
                    
                elif method == 'dbscan':
                    # Using cuML for GPU-accelerated DBSCAN
                    from cuml.cluster import DBSCAN as cuDBSCAN
                    
                    clustering = cuDBSCAN(eps=0.5, min_samples=5)
                    labels = clustering.fit_predict(X_flat)
                    
                    # Convert back to CPU if needed
                    try:
                        labels = labels.get()
                    except:
                        pass
                    
                    # DBSCAN can produce -1 labels for noise, set to max_label+1 for evaluation
                    if -1 in labels:
                        labels[labels == -1] = labels.max() + 1
                    
                    print(f"  GPU DBSCAN completed in {time.time() - start_time:.2f} seconds")
                    
                else:
                    raise ImportError(f"No GPU implementation available for {method}")
                    
            except ImportError as e:
                print(f"  {e}. Falling back to CPU implementation.")
                use_gpu_for_this_method = False
            except Exception as e:
                print(f"  Error with GPU implementation: {e}. Falling back to CPU.")
                use_gpu_for_this_method = False
        else:
            use_gpu_for_this_method = False
        
        # Fall back to CPU implementation if GPU failed or wasn't requested
        if not use_gpu or not use_gpu_for_this_method:
            cpu_start_time = time.time()
            
            if method == 'kmeans':
                # K-means with k=n_clusters
                clustering = KMeans(n_clusters=n_clusters, random_state=42)
                labels = clustering.fit_predict(X_flat)
                print(f"  CPU K-means completed in {time.time() - cpu_start_time:.2f} seconds")
                
            elif method == 'gmm':
                # Gaussian Mixture Model
                clustering = GaussianMixture(n_components=n_clusters, random_state=42)
                labels = clustering.fit_predict(X_flat)
                print(f"  CPU GMM completed in {time.time() - cpu_start_time:.2f} seconds")
                
            elif method == 'dbscan':
                # DBSCAN - epsilon needs to be tuned for your specific data
                clustering = DBSCAN(eps=0.5, min_samples=5)
                labels = clustering.fit_predict(X_flat)
                
                # DBSCAN can produce -1 labels for noise, set to max_label+1 for evaluation
                if -1 in labels:
                    labels[labels == -1] = labels.max() + 1
                
                print(f"  CPU DBSCAN completed in {time.time() - cpu_start_time:.2f} seconds")
                    
            elif method == 'hierarchical':
                # Hierarchical Clustering
                clustering = AgglomerativeClustering(n_clusters=n_clusters)
                labels = clustering.fit_predict(X_flat)
                print(f"  CPU hierarchical clustering completed in {time.time() - cpu_start_time:.2f} seconds")
                
            else:
                raise ValueError(f"Unknown clustering method: {method}")
        
        # Evaluate clustering only if we have at least 2 clusters
        unique_labels = np.unique(labels)
        if len(unique_labels) >= 2:
            try:
                sil_score = silhouette_score(X_flat, labels)
                db_score = davies_bouldin_score(X_flat, labels)
                ch_score = calinski_harabasz_score(X_flat, labels)
                
                eval_metrics = {
                    'silhouette_score': sil_score,
                    'davies_bouldin_score': db_score,
                    'calinski_harabasz_score': ch_score
                }
            except Exception as e:
                print(f"  Warning: Could not compute evaluation metrics. Error: {e}")
                eval_metrics = {
                    'silhouette_score': None,
                    'davies_bouldin_score': None,
                    'calinski_harabasz_score': None
                }
        else:
            eval_metrics = {
                'silhouette_score': None,
                'davies_bouldin_score': None,
                'calinski_harabasz_score': None
            }
            
        clustering_results[method] = {
            'labels': labels,
            'clustering': clustering,
            'eval_metrics': eval_metrics,
            'indices': indices  # Save which bundles were used
        }
        
        # Print evaluation metrics
        print("  Evaluation metrics:")
        for metric, value in eval_metrics.items():
            if value is not None:
                print(f"    {metric}: {value:.4f}")
    
    # Clean up
    del X
    del X_flat
    gc.collect()
    
    # Save for later use
    clustering_path = os.path.join(bundle_info['output_dir'], 'clustering_results.pkl')
    with open(clustering_path, 'wb') as f:
        import pickle
        pickle.dump(clustering_results, f)
    
    return clustering_results

def predict_clusters_disk(bundle_info, clustering_results, output_file=None):
    """
    Predict clusters for all bundles using trained clustering model
    
    Parameters:
    bundle_info (dict): Bundle info from normalize_bundles_disk
    clustering_results (dict): Results from memory_efficient_clustering
    output_file (str): Path to save results (None to skip saving)
    
    Returns:
    dict: Dictionary mapping bundle indices to cluster labels
    """
    from .helpers import DiskBundleLoader
    
    # Find best clustering method
    best_method = max(
        clustering_results.keys(),
        key=lambda m: (
            clustering_results[m]['eval_metrics']['silhouette_score'] 
            if clustering_results[m]['eval_metrics']['silhouette_score'] is not None 
            else -1
        )
    )
    
    print(f"\nUsing {best_method} clustering model to predict all clusters")
    
    # Get model
    model = clustering_results[best_method]['clustering']
    
    # Create loader for all bundles
    total_bundles = bundle_info['total_bundles']
    all_indices = np.arange(total_bundles)
    
    # Map to store predictions
    bundle_clusters = {}
    
    # Process in batches to avoid memory issues
    loader = DiskBundleLoader(
        bundle_info=bundle_info,
        indices=all_indices,
        batch_size=1000,
        shuffle=False,
        use_normalized=True
    )
    
    print(f"Predicting clusters for {total_bundles} bundles...")
    
    for i, batch in enumerate(loader):
        # Get batch indices
        start_idx = i * loader.batch_size
        end_idx = min(start_idx + loader.batch_size, total_bundles)
        batch_indices = all_indices[start_idx:end_idx]
        
        # Flatten time dimension
        if batch.ndim == 3:
            batch_flat = batch.mean(axis=1)
        else:
            batch_flat = batch
        
        # Predict clusters
        batch_clusters = model.predict(batch_flat)
        
        # Store predictions
        for idx, cluster in zip(batch_indices, batch_clusters):
            bundle_clusters[idx] = int(cluster)
        
        # Progress
        if (i + 1) % 10 == 0 or end_idx == total_bundles:
            print(f"  Processed {end_idx}/{total_bundles} bundles")
    
    # Save results if requested
    if output_file:
        import json
        with open(output_file, 'w') as f:
            # Convert keys to strings (JSON requirement)
            json_clusters = {str(k): v for k, v in bundle_clusters.items()}
            json.dump(json_clusters, f)
    
    return bundle_clusters

def memory_efficient_clustering_workflow(directory_path, output_dir='./models/clustering',
                                       bundle_size=30, step_size=15, max_files_per_batch=5,
                                       max_samples_clustering=10000, cluster_methods=['kmeans', 'gmm'],
                                       use_gpu=True):
    """
    Run the complete memory-efficient clustering workflow
    
    Parameters:
    directory_path (str): Path to directory with EEG CSV files
    output_dir (str): Directory to save models and results
    bundle_size (int): Size of time series bundles
    step_size (int): Step size for sliding window
    max_files_per_batch (int): Maximum files to process at once
    max_samples_clustering (int): Maximum samples to use for clustering
    cluster_methods (list): Clustering methods to use
    use_gpu (bool): Whether to use GPU acceleration
    
    Returns:
    tuple: (bundle_info, clustering_results, bundle_clusters)
    """
    from .helpers import (
        load_eeg_data, preprocess_eeg_data, engineer_eeg_features,
        create_coherent_time_series_bundles_disk, normalize_bundles_disk
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load and preprocess data
    print("\nStep 1: Loading EEG data...")
    if os.path.isdir(directory_path):
        file_dfs, _ = load_eeg_data(directory_path=directory_path)
    else:
        file_dfs, _ = load_eeg_data(single_file=directory_path)
    
    # Step 2: Preprocess and engineer features
    print("\nStep 2: Preprocessing and feature engineering...")
    processed_dfs = {}
    
    # Process files in batches to save memory
    file_ids = list(file_dfs.keys())
    
    for batch_start in range(0, len(file_ids), max_files_per_batch):
        batch_end = min(batch_start + max_files_per_batch, len(file_ids))
        batch_file_ids = file_ids[batch_start:batch_end]
        
        print(f"  Processing files {batch_start+1}-{batch_end} of {len(file_ids)}")
        
        for file_id in batch_file_ids:
            # Preprocess
            df_clean = preprocess_eeg_data(file_dfs[file_id])
            
            # Engineer features
            df_features = engineer_eeg_features(df_clean)
            
            processed_dfs[file_id] = df_features
            
            # Remove original dataframe to save memory
            del file_dfs[file_id]
        
        # Force garbage collection
        gc.collect()
    
    # Step 3: Create bundles and save to disk
    print("\nStep 3: Creating time series bundles and saving to disk...")
    metadata, bundle_info = create_coherent_time_series_bundles_disk(
        file_dfs=processed_dfs,
        bundle_size=bundle_size,
        step_size=step_size,
        max_files_per_batch=max_files_per_batch,
        output_dir=os.path.join(output_dir, 'bundles')
    )
    
    # Clear memory
    del processed_dfs
    gc.collect()
    
    # Step 4: Normalize bundles
    print("\nStep 4: Normalizing bundles on disk...")
    bundle_info = normalize_bundles_disk(
        bundle_info=bundle_info,
        normalization='per_feature'
    )
    
    # Step 5: Cluster a subset of the data
    print("\nStep 5: Clustering a subset of the data...")
    clustering_results = memory_efficient_clustering(
        bundle_info=bundle_info,
        max_samples=max_samples_clustering,
        methods=cluster_methods,
        use_gpu=use_gpu
    )
    
    # Step 6: Predict clusters for all bundles
    print("\nStep 6: Predicting clusters for all bundles...")
    bundle_clusters = predict_clusters_disk(
        bundle_info=bundle_info,
        clustering_results=clustering_results,
        output_file=os.path.join(output_dir, 'bundle_clusters.json')
    )
    
    # Step 7: Analyze results
    print("\nStep 7: Analyzing cluster characteristics...")
    best_method = max(
        clustering_results.keys(),
        key=lambda m: (
            clustering_results[m]['eval_metrics']['silhouette_score'] 
            if clustering_results[m]['eval_metrics']['silhouette_score'] is not None 
            else -1
        )
    )
    
    # Count bundles per cluster
    cluster_counts = {}
    for cluster in bundle_clusters.values():
        if cluster not in cluster_counts:
            cluster_counts[cluster] = 0
        cluster_counts[cluster] += 1
    
    print(f"\nCluster distribution from {best_method}:")
    for cluster, count in sorted(cluster_counts.items()):
        print(f"  Cluster {cluster}: {count} bundles ({count/len(bundle_clusters)*100:.1f}%)")
    
    return bundle_info, clustering_results, bundle_clusters

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run EEG clustering workflow')
    parser.add_argument('--data_path', type=str, required=True, 
                      help='Path to EEG data directory or single CSV file')
    parser.add_argument('--output_dir', type=str, default='./models/clustering',
                      help='Directory to save models and results')
    parser.add_argument('--bundle_size', type=int, default=30,
                      help='Size of time series bundles')
    parser.add_argument('--step_size', type=int, default=15,
                      help='Step size for sliding window')
    parser.add_argument('--reducer', type=str, default='umap', choices=['pca', 'tsne', 'umap'],
                      help='Dimensionality reduction method')
    parser.add_argument('--use_gpu', action='store_true',
                      help='Use GPU acceleration if available')
    parser.add_argument('--cluster_methods', nargs='+', 
                      default=['kmeans', 'gmm'],
                      choices=['kmeans', 'gmm', 'dbscan', 'hierarchical'],
                      help='Clustering methods to apply')
    parser.add_argument('--benchmark', action='store_true',
                      help='Run GPU vs CPU benchmark for dimensionality reduction and clustering')
    
    args = parser.parse_args()
    
    if args.benchmark:
        print("\n=== RUNNING BENCHMARKS ===")
        # First load some data for benchmarking
        if os.path.isdir(args.data_path):
            file_dfs, combined_df = load_eeg_data(directory_path=args.data_path)
        else:
            file_dfs, combined_df = load_eeg_data(single_file=args.data_path)
        
        # Preprocess a sample for benchmarking
        for file_id, df in file_dfs.items():
            print(f"\nPreprocessing sample file {file_id} for benchmark...")
            df_clean = preprocess_eeg_data(df)
            df_features = engineer_eeg_features(df_clean)
            
            # Create bundles for this file
            X_bundles, _ = create_coherent_time_series_bundles(
                file_dfs={file_id: df_features},
                bundle_size=args.bundle_size,
                step_size=args.step_size
            )
            
            # Normalize bundles
            X_normalized = normalize_bundles(X_bundles, normalization='per_feature')
            
            # Flatten for dimensionality reduction benchmark
            X_flat = X_normalized.mean(axis=1) if X_normalized.ndim == 3 else X_normalized
            
            # Benchmark dimensionality reduction
            print("\n=== DIMENSIONALITY REDUCTION BENCHMARKS ===")
            for reducer in ['pca', 'umap', 'tsne']:
                print(f"\nBenchmarking {reducer.upper()}:")
                # CPU timing
                cpu_start = time.time()
                _, _ = reduce_dimensions(X_flat, method=reducer, use_gpu=False)
                cpu_time = time.time() - cpu_start
                
                # GPU timing
                try:
                    gpu_start = time.time()
                    _, _ = reduce_dimensions(X_flat, method=reducer, use_gpu=True)
                    gpu_time = time.time() - gpu_start
                    
                    speedup = cpu_time / gpu_time
                    print(f"CPU: {cpu_time:.2f}s, GPU: {gpu_time:.2f}s, Speedup: {speedup:.2f}x")
                except Exception as e:
                    print(f"GPU failed: {e}")
                    print(f"CPU: {cpu_time:.2f}s")
            
            # Benchmark clustering
            print("\n=== CLUSTERING BENCHMARKS ===")
            benchmark_clustering(X_flat, methods=['kmeans', 'dbscan'])
            
            # Only benchmark with one file to save time
            break
    else:
        # Run normal workflow
        run_clustering_workflow(
            data_path=args.data_path,
            output_dir=args.output_dir,
            bundle_size=args.bundle_size,
            step_size=args.step_size,
            reducer_method=args.reducer,
            use_gpu=args.use_gpu,
            cluster_methods=args.cluster_methods
        )