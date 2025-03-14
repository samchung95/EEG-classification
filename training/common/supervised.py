import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
import os
import traceback
from sklearn.base import BaseEstimator

# PyTorch imports

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn.utils import clip_grad_norm_

import joblib
import os
import time

# Import helper functions
from .helpers import (
    load_eeg_data, preprocess_eeg_data, engineer_eeg_features, select_features,
    create_coherent_time_series_bundles, normalize_bundles,
    split_bundles_train_test
)

# Import from our local modules
from .helpers import logger

class SupervisedModelTrainer:
    """
    Class for training supervised machine learning models on EEG data.
    
    This class provides methods for data preparation, model training,
    hyperparameter tuning, and evaluation of supervised models.
    """
    
    def __init__(self, 
                 model_type: str = 'random_forest', 
                 random_state: int = 42,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the supervised model trainer.
        
        Args:
            model_type: Type of model to train ('random_forest', 'svm', etc.)
            random_state: Random seed for reproducibility
            config: Configuration dictionary with training parameters
        """
        self.model_type = model_type
        self.random_state = random_state
        self.config = config or {}
        self.model = None
        self.feature_importance = None
        
        logger.info(f"Initialized SupervisedModelTrainer with model_type={model_type}")
        
    def prepare_data(self, 
                     data: pd.DataFrame, 
                     target_column: str,
                     feature_columns: Optional[List[str]] = None,
                     test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for supervised learning.
        
        Args:
            data: DataFrame containing features and target
            target_column: Name of the column containing the target variable
            feature_columns: List of column names to use as features (if None, use all except target)
            test_size: Proportion of data to use for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Preparing data with shape {data.shape}")
        
        try:
            # Extract features and target
            if feature_columns is None:
                feature_columns = [col for col in data.columns if col != target_column]
                
            X = data[feature_columns].values
            y = data[target_column].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
            
            logger.info(f"Data prepared: X_train shape={X_train.shape}, X_test shape={X_test.shape}")
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
            
    def create_model(self) -> BaseEstimator:
        """
        Create a model instance based on the specified model type.
        
        Returns:
            A scikit-learn compatible model instance
        """
        try:
            if self.model_type == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(
                    n_estimators=self.config.get('n_estimators', 100),
                    random_state=self.random_state
                )
            elif self.model_type == 'svm':
                from sklearn.svm import SVC
                model = SVC(
                    probability=True,
                    random_state=self.random_state
                )
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
            logger.info(f"Created model of type {self.model_type}")
            return model
            
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
            
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              hyperparameter_tuning: bool = False,
              param_grid: Optional[Dict[str, List[Any]]] = None) -> BaseEstimator:
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            param_grid: Parameter grid for hyperparameter tuning
            
        Returns:
            Trained model
        """
        logger.info(f"Training {self.model_type} model on data with shape {X_train.shape}")
        
        try:
            model = self.create_model()
            
            if hyperparameter_tuning and param_grid:
                logger.info("Performing hyperparameter tuning")
                grid_search = GridSearchCV(
                    model, 
                    param_grid, 
                    cv=self.config.get('cv', 5),
                    n_jobs=self.config.get('n_jobs', -1),
                    verbose=self.config.get('verbose', 1)
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                logger.info(f"Best parameters: {grid_search.best_params_}")
            else:
                model.fit(X_train, y_train)
                
            self.model = model
            
            # Store feature importance if available
            if hasattr(model, 'feature_importances_'):
                self.feature_importance = model.feature_importances_
                
            return model
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
            
    def evaluate(self, 
                X_test: np.ndarray, 
                y_test: np.ndarray,
                class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            class_names: Names of the classes for reporting
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            error_msg = "Model not trained yet. Call train() first."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        logger.info("Evaluating model on test data")
        
        try:
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            report = classification_report(
                y_test, 
                y_pred, 
                target_names=class_names,
                output_dict=True
            )
            
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            logger.info(f"Model evaluation completed. Accuracy: {accuracy:.4f}")
            
            return {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': conf_matrix
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
            
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
            joblib.dump(self.model, filepath)
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
            self.model = joblib.load(filepath)
            logger.info(f"Model loaded from {filepath}")
            
            # Try to extract feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = self.model.feature_importances_
                
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.debug(traceback.format_exc())
            return False

class EEGDiskDataset(Dataset):
    """
    PyTorch Dataset for EEG bundles stored on disk
    """
    def __init__(self, bundle_info, indices, label_mapping=None):
        """
        Initialize the dataset
        
        Parameters:
        bundle_info (dict): Bundle info dictionary from normalize_bundles_disk
        indices (list): Indices of bundles to include
        label_mapping (dict): Mapping from bundle indices to labels (None for unlabeled)
        """
        self.bundle_info = bundle_info
        self.indices = np.array(indices)
        self.label_mapping = label_mapping
        
        # Determine data directory
        if 'normalized_dir' in bundle_info:
            self.data_dir = bundle_info['normalized_dir']
        else:
            self.data_dir = bundle_info['output_dir']
        
        # Load metadata
        metadata_path = bundle_info['metadata_path']
        self.metadata = pd.read_csv(metadata_path)
        
        # Create bundle index to batch file mapping
        self.bundle_to_batch = {}
        self.bundle_to_position = {}
        
        for _, row in self.metadata.iterrows():
            bundle_idx = row['bundle_index']
            batch_idx = row['batch_index']
            
            # Find position within batch
            batch_indices = self.metadata[self.metadata['batch_index'] == batch_idx]['bundle_index'].values
            position = np.where(batch_indices == bundle_idx)[0][0]
            
            self.bundle_to_batch[bundle_idx] = batch_idx
            self.bundle_to_position[bundle_idx] = position
        
        # Initialize batch cache
        self.batch_cache = {}
        self.max_cache_size = 5
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get bundle index
        bundle_idx = self.indices[idx]
        
        # Get batch file and position
        batch_idx = self.bundle_to_batch.get(bundle_idx)
        position = self.bundle_to_position.get(bundle_idx)
        
        if batch_idx is None or position is None:
            raise ValueError(f"Bundle index {bundle_idx} not found in metadata")
        
        # Load batch if not in cache
        if batch_idx not in self.batch_cache:
            # If cache is full, remove oldest entry
            if len(self.batch_cache) >= self.max_cache_size:
                oldest_key = next(iter(self.batch_cache.keys()))
                del self.batch_cache[oldest_key]
            
            # Load batch
            batch_path = os.path.join(self.data_dir, f'bundle_batch_{batch_idx}.npy')
            self.batch_cache[batch_idx] = np.load(batch_path)
        
        # Get bundle
        bundle = self.batch_cache[batch_idx][position]
        
        # Convert to tensor
        bundle_tensor = torch.FloatTensor(bundle)
        
        # Return with label if available
        if self.label_mapping is not None:
            label = self.label_mapping.get(bundle_idx, 0)  # Default to 0 if not found
            return bundle_tensor, torch.LongTensor([label])[0]
        else:
            return bundle_tensor

def create_disk_data_loaders(bundle_info, train_indices, test_indices, labels=None, 
                           batch_size=32, num_workers=4):
    """
    Create PyTorch DataLoaders for training and testing with disk-stored data
    
    Parameters:
    bundle_info (dict): Bundle info from normalize_bundles_disk
    train_indices (list): Indices for training set
    test_indices (list): Indices for test set
    labels (dict): Dictionary mapping bundle indices to labels (None for unlabeled)
    batch_size (int): Batch size for DataLoader
    num_workers (int): Number of worker processes for data loading
    
    Returns:
    tuple: (train_loader, val_loader, test_loader)
    """
    # Create validation set from training data
    np.random.seed(42)
    np.random.shuffle(train_indices)
    
    val_size = min(int(len(train_indices) * 0.2), 1000)  # Cap at 1000 for efficiency
    val_indices = train_indices[:val_size]
    train_indices = train_indices[val_size:]
    
    # Create datasets
    train_dataset = EEGDiskDataset(bundle_info, train_indices, labels)
    val_dataset = EEGDiskDataset(bundle_info, val_indices, labels)
    test_dataset = EEGDiskDataset(bundle_info, test_indices, labels)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def train_pytorch_model_disk(model, bundle_info, train_indices, test_indices, 
                            label_mapping, model_path=None, 
                            epochs=50, batch_size=32, num_workers=4):
    """
    Train PyTorch model using disk-stored bundles to avoid memory issues
    
    Parameters:
    model (nn.Module): PyTorch model to train
    bundle_info (dict): Bundle info from normalize_bundles_disk
    train_indices (ndarray): Indices for training set
    test_indices (ndarray): Indices for test set
    label_mapping (dict): Mapping from bundle indices to labels
    model_path (str): Path to save the best model
    epochs (int): Number of epochs to train
    batch_size (int): Batch size for training
    num_workers (int): Number of worker processes for data loading
    
    Returns:
    dict: Dictionary with trained model and performance metrics
    """
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_disk_data_loaders(
        bundle_info, train_indices, test_indices, label_mapping,
        batch_size=batch_size, num_workers=num_workers
    )
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience = 10
    patience_counter = 0
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    start_time = time.time()
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            # Clip gradients to prevent exploding gradients
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Calculate training statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Calculate validation statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Training time
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Save model if requested
    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    # Test phase
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Calculate test statistics
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            # Store for metrics
            all_labels.append(labels.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())
            all_probs.append(outputs.cpu().numpy())
    
    # Calculate metrics
    test_loss = test_loss / test_total
    test_acc = test_correct / test_total
    
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_proba = np.concatenate(all_probs)
    
    f1 = f1_score(y_true, y_pred)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - PyTorch Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
    
    # Return results
    return {
        'model': model,
        'history': history,
        'accuracy': test_acc,
        'f1_score': f1,
        'y_test': y_true,
        'y_pred': y_pred,
        'train_time': train_time
    }

def create_ground_truth(df, method='threshold', **kwargs):
    """
    Create ground truth labels based on different methods
    
    Parameters:
    df (DataFrame): DataFrame with EEG data
    method (str): Method to use ('threshold', 'cluster', 'mellow_concentration', 'time_periods')
    **kwargs: Additional parameters specific to each method
    
    Returns:
    Series: Ground truth labels (0 for relaxed, 1 for focused)
    """
    if method == 'threshold':
        # Use Alpha/Beta ratio threshold
        # Higher Alpha/Beta ratio typically indicates relaxation
        channel = kwargs.get('channel', 'AF7')
        threshold = kwargs.get('threshold', None)
        
        ratio_col = f'AlphaBeta_Ratio_{channel}'
        
        if ratio_col not in df.columns:
            # Check if we can calculate it
            alpha_col = f'Alpha_{channel}'
            beta_col = f'Beta_{channel}'
            
            if alpha_col in df.columns and beta_col in df.columns:
                # Calculate Alpha/Beta ratio
                df[ratio_col] = df[alpha_col] / df[beta_col]
            else:
                raise ValueError(f"Cannot calculate {ratio_col}. Required columns not found.")
        
        if threshold is None:
            # Use median as default threshold
            threshold = df[ratio_col].median()
            
        # 0 = Relaxed (high Alpha/Beta), 1 = Focused (low Alpha/Beta)
        labels = (df[ratio_col] < threshold).astype(int)
        
    elif method == 'cluster':
        # Use cluster labels from unsupervised learning
        cluster_labels = kwargs.get('cluster_labels')
        relaxed_cluster = kwargs.get('relaxed_cluster', 0)  # Default cluster 0 is relaxed
        
        if cluster_labels is None:
            raise ValueError("cluster_labels must be provided for 'cluster' method")
            
        # Map clusters to binary labels (0 = relaxed, 1 = focused)
        labels = np.where(cluster_labels == relaxed_cluster, 0, 1)
            
    elif method == 'mellow_concentration':
        # Use built-in Muse metrics if available
        if 'Mellow' in df.columns and 'Concentration' in df.columns:
            threshold = kwargs.get('threshold', 0)
            # If Concentration > Mellow + threshold, then focused (1)
            labels = (df['Concentration'] > (df['Mellow'] + threshold)).astype(int)
        else:
            raise ValueError("Mellow and Concentration columns not found")
            
    elif method == 'time_periods':
        # Use specified time periods for each state
        time_periods = kwargs.get('time_periods', {})
        
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have datetime index for time_periods method")
            
        labels = pd.Series(index=df.index, data=np.nan)
        
        for state, periods in time_periods.items():
            for start, end in periods:
                mask = (df.index >= start) & (df.index <= end)
                labels.loc[mask] = 0 if state == 'relaxed' else 1
                
        # Fill any remaining NaNs with a default value
        labels = labels.fillna(kwargs.get('default_label', 0)).astype(int)
        
    else:
        raise ValueError(f"Unknown method: {method}")
        
    return labels

def train_traditional_models(X_train, y_train, X_test, y_test, models=None):
    """
    Train and evaluate multiple traditional ML models
    
    Parameters:
    X_train (ndarray): Training features
    y_train (ndarray): Training labels
    X_test (ndarray): Test features
    y_test (ndarray): Test labels
    models (dict): Dictionary of models to train (None for default)
    
    Returns:
    dict: Dictionary of trained models and their performances
    """
    # Define models to evaluate if not provided
    if models is None:
        models = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(probability=True, random_state=42),
            'MLP': MLPClassifier(random_state=42, max_iter=1000)
        }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Get probabilities if model supports it
        try:
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            has_proba = True
        except:
            y_prob = None
            has_proba = False
        
        # Calculate training time
        train_time = time.time() - start_time
        
        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Training Time: {train_time:.2f} seconds")
        
        # Calculate AUC if probabilities are available
        if has_proba:
            auc = roc_auc_score(y_test, y_prob)
            print(f"  ROC AUC: {auc:.4f}")
        else:
            auc = None
        
        # Store results
        results[name] = {
            'pipeline': pipeline,
            'accuracy': acc,
            'f1_score': f1,
            'roc_auc': auc,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'train_time': train_time
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
        
        # ROC curve if probabilities are available
        if has_proba:
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {name}')
            plt.legend()
            plt.tight_layout()
            plt.show()
    
    # Determine best model based on F1 score (more balanced than accuracy)
    best_model = max(results, key=lambda k: results[k]['f1_score'])
    
    print(f"\nBest model: {best_model} with F1 Score = {results[best_model]['f1_score']:.4f}")
    
    # Feature importance for the best model (if it supports it)
    if best_model in ['RandomForest', 'GradientBoosting']:
        feature_importances = results[best_model]['pipeline'].named_steps['model'].feature_importances_
        
        if hasattr(X_train, 'columns'):  # If X_train is a DataFrame
            feature_names = X_train.columns
        else:
            feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
        
        # Create DataFrame for feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False)
        
        # Plot top 15 features
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
        plt.title(f'Top 15 Feature Importances - {best_model}')
        plt.tight_layout()
        plt.show()
    
    return results, best_model

class CNNEEG(nn.Module):
    """
    PyTorch implementation of 1D CNN + LSTM model for EEG sequence classification
    """
    def __init__(self, input_size, num_classes=2):
        """
        Initialize the model
        
        Parameters:
        input_size (int): Number of input features
        num_classes (int): Number of output classes
        """
        super(CNNEEG, self).__init__()
        
        # 1D CNN layers
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.2)
        
        # Calculate the size after CNN layers
        # With two max pooling layers (kernel=2), the sequence length is reduced by a factor of 4
        # If original sequence length is L, now it's L/4 (integer division)
        self.lstm_input_size = 128  # Output channels from conv2
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=128,
            batch_first=True
        )
        self.dropout3 = nn.Dropout(0.3)
        
        # Dense layers
        self.fc1 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        x (Tensor): Input tensor of shape [batch_size, sequence_length, features]
        
        Returns:
        Tensor: Output tensor of shape [batch_size, num_classes]
        """
        # Transpose to [batch_size, features, sequence_length] for 1D convolution
        x = x.permute(0, 2, 1)
        
        # CNN layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Transpose back to [batch_size, sequence_length, features] for LSTM
        x = x.permute(0, 2, 1)
        
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        # Take the output from the last time step
        x = lstm_out[:, -1, :]
        x = self.dropout3(x)
        
        # Dense layers
        x = self.relu(self.bn3(self.fc1(x)))
        x = self.dropout4(x)
        x = self.fc2(x)
        
        return x

def create_1D_CNN_LSTM(input_shape, num_classes=2):
    """
    Create a 1D CNN + LSTM model for EEG sequence classification
    
    Parameters:
    input_shape (tuple): Shape of input data (time_steps, features)
    num_classes (int): Number of classes to predict
    
    Returns:
    CNNEEG: PyTorch model
    """
    _, input_size = input_shape
    
    # Create the model
    model = CNNEEG(input_size=input_size, num_classes=num_classes)
    
    return model

def train_deep_learning_model(X_train, y_train, X_test, y_test, model_path=None, epochs=50, batch_size=32):
    """
    Train and evaluate deep learning model for EEG classification using PyTorch
    
    Parameters:
    X_train (ndarray): Training sequences [samples, time_steps, features]
    y_train (ndarray): Training labels
    X_test (ndarray): Test sequences [samples, time_steps, features]
    y_test (ndarray): Test labels
    model_path (str): Path to save the model
    epochs (int): Number of epochs to train
    batch_size (int): Batch size for training
    
    Returns:
    dict: Dictionary with trained model and performance metrics
    """
    print("\nTraining Deep Learning Model (1D CNN + LSTM) with PyTorch...")
    start_time = time.time()
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Split training data to create validation set
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Create model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_1D_CNN_LSTM(input_shape=input_shape)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience = 10
    patience_counter = 0
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            # Clip gradients to prevent exploding gradients
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Calculate training statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Calculate validation statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Save the model if path is provided
    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    # Calculate training time
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Calculate test statistics
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            # Store outputs and labels for metrics
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    test_loss = test_loss / test_total
    test_acc = test_correct / test_total
    
    # Convert predictions to numpy for sklearn metrics
    y_pred_proba = np.vstack(all_outputs)
    y_true = np.concatenate(all_labels)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Calculate AUC for binary classification
    if y_pred_proba.shape[1] == 2:
        auc = roc_auc_score(y_true, y_pred_proba[:, 1])
    else:
        auc = None
    
    # Print results
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    if auc:
        print(f"  ROC AUC: {auc:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - PyTorch CNN+LSTM Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
    
    # ROC curve if binary classification
    if y_pred_proba.shape[1] == 2:
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        plt.plot(fpr, tpr, label=f'CNN+LSTM (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - PyTorch CNN+LSTM Model')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return {
        'model': model,
        'history': history,
        'accuracy': acc,
        'f1_score': f1,
        'roc_auc': auc,
        'y_test': y_true,
        'y_pred': y_pred,
        'train_time': train_time
    }

def run_supervised_workflow(data_path, output_dir='./models/supervised',
                           bundle_size=30, step_size=15,
                           labeling_method='threshold',
                           use_dl_model=True,
                           **labeling_kwargs):
    """
    Run the complete supervised learning workflow
    
    Parameters:
    data_path (str): Path to directory containing EEG CSV files or single CSV file
    output_dir (str): Directory to save models and results
    bundle_size (int): Size of time series bundles
    step_size (int): Step size for sliding window
    labeling_method (str): Method to create ground truth labels
    use_dl_model (bool): Whether to train deep learning model
    **labeling_kwargs: Additional parameters for labeling method
    
    Returns:
    dict: Results of supervised workflow
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
    
    # Step 3: Create labels for each file
    print(f"\nStep 3: Creating ground truth labels using method '{labeling_method}'...")
    labeled_dfs = {}
    all_labels = []
    
    # Handle special case for 'cluster' method
    if labeling_method == 'cluster':
        # Load cluster labels from unsupervised workflow
        cluster_labels_path = labeling_kwargs.get('cluster_labels_path')
        if cluster_labels_path and os.path.exists(cluster_labels_path):
            cluster_labels = np.load(cluster_labels_path)
            
            # Apply cluster labels to each file based on metadata
            metadata_path = os.path.join(os.path.dirname(cluster_labels_path), 'metadata_with_clusters.csv')
            if os.path.exists(metadata_path):
                metadata = pd.read_csv(metadata_path)
                
                # Create a mapping from file_id and start_index to cluster label
                for file_id, df in processed_dfs.items():
                    file_metadata = metadata[metadata['file_id'] == file_id]
                    if not file_metadata.empty:
                        labels = create_ground_truth(df, method='cluster', 
                                              cluster_labels=cluster_labels, **labeling_kwargs)
                        labeled_dfs[file_id] = df.copy()
                        labeled_dfs[file_id]['label'] = labels
                        all_labels.extend(labels)
                    else:
                        print(f"Warning: No metadata found for file {file_id}")
            else:
                print(f"Warning: Metadata file not found at {metadata_path}")
                # Fall back to threshold method
                labeling_method = 'threshold'
        else:
            print(f"Warning: Cluster labels not found at {cluster_labels_path}")
            # Fall back to threshold method
            labeling_method = 'threshold'
    
    # Apply selected labeling method to each file
    if labeling_method != 'cluster' or not labeled_dfs:
        for file_id, df in processed_dfs.items():
            try:
                labels = create_ground_truth(df, method=labeling_method, **labeling_kwargs)
                labeled_dfs[file_id] = df.copy()
                labeled_dfs[file_id]['label'] = labels
                all_labels.extend(labels)
            except Exception as e:
                print(f"Warning: Could not create labels for file {file_id}. Error: {e}")
    
    # Check if we have enough labeled data
    if not labeled_dfs:
        raise ValueError("No files could be labeled. Check your labeling method and parameters.")
    
    print(f"Created labels for {len(labeled_dfs)} files.")
    print(f"Class distribution: {np.bincount(np.array(all_labels).astype(int))}")
    
    # Step 4: Create coherent time series bundles with labels
    print("\nStep 4: Creating time series bundles...")
    X_bundles, metadata = create_coherent_time_series_bundles(
        file_dfs=labeled_dfs,
        bundle_size=bundle_size,
        step_size=step_size
    )
    
    # Extract labels from each bundle's last timestep
    bundle_labels = []
    for idx, row in metadata.iterrows():
        file_id = row['file_id']
        end_idx = row['end_index']
        
        # Get the label at the end of the bundle
        label = labeled_dfs[file_id]['label'].iloc[end_idx]
        bundle_labels.append(label)
    
    bundle_labels = np.array(bundle_labels)
    
    # Step 5: Normalize bundles
    print("\nStep 5: Normalizing bundles...")
    X_normalized = normalize_bundles(X_bundles, normalization='per_feature')
    
    # Step 6: Split into training and testing sets
    print("\nStep 6: Splitting into training and testing sets...")
    split_result = split_bundles_train_test(
        X_normalized, metadata, test_ratio=0.2, by_file=True, labels=bundle_labels
    )
    
    X_train, X_test, y_train, y_test, train_metadata, test_metadata = split_result
    
    # Step 7: Train traditional models
    print("\nStep 7: Training traditional machine learning models...")
    # For traditional models, we need to flatten the time dimension
    # Take mean across time steps for each feature
    X_train_flat = X_train.mean(axis=1) if X_train.ndim == 3 else X_train
    X_test_flat = X_test.mean(axis=1) if X_test.ndim == 3 else X_test
    
    # Train the models
    traditional_results, best_traditional = train_traditional_models(
        X_train_flat, y_train, X_test_flat, y_test
    )
    
    # Save the best traditional model
    joblib.dump(
        traditional_results[best_traditional]['pipeline'],
        os.path.join(output_dir, f'{best_traditional}_model.joblib')
    )
    
    # Step 8: Train deep learning model if requested
    dl_results = None
    if use_dl_model:
        print("\nStep 8: Training deep learning model...")
        
        # For DL model, we use the full 3D data (don't flatten time dimension)
        if X_train.ndim == 3:
            dl_results = train_deep_learning_model(
                X_train, y_train, X_test, y_test,
                model_path=os.path.join(output_dir, 'deep_learning_model'),
                epochs=50, batch_size=32
            )
    
    # Step 9: Save results and metadata
    print("\nStep 9: Saving results and metadata...")
    results_summary = {
        'bundle_size': bundle_size,
        'step_size': step_size,
        'labeling_method': labeling_method,
        'class_distribution': np.bincount(bundle_labels).tolist(),
        'traditional_models': {
            model: {
                'accuracy': traditional_results[model]['accuracy'],
                'f1_score': traditional_results[model]['f1_score'],
                'roc_auc': traditional_results[model]['roc_auc'],
                'train_time': traditional_results[model]['train_time']
            } for model in traditional_results
        },
        'best_traditional_model': best_traditional
    }
    
    if dl_results:
        results_summary['deep_learning'] = {
            'accuracy': dl_results['accuracy'],
            'f1_score': dl_results['f1_score'],
            'roc_auc': dl_results['roc_auc'],
            'train_time': dl_results['train_time']
        }
    
    # Save results summary
    with open(os.path.join(output_dir, 'results_summary.txt'), 'w') as f:
        for key, value in results_summary.items():
            f.write(f"{key}: {value}\n")
    
    return {
        'X_bundles': X_bundles,
        'X_normalized': X_normalized,
        'bundle_labels': bundle_labels,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'traditional_results': traditional_results,
        'best_traditional': best_traditional,
        'dl_results': dl_results,
        'results_summary': results_summary
    }

def predict_mental_state(X, model_path, is_dl_model=False):
    """
    Predict mental state (relaxed or focused) using a trained model
    
    Parameters:
    X (ndarray): Input features or sequences
    model_path (str): Path to the saved model
    is_dl_model (bool): Whether the model is a deep learning model
    
    Returns:
    ndarray: Predicted labels (0 = relaxed, 1 = focused)
    """
    # Ensure X has the right dimensions
    if not is_dl_model and X.ndim == 3:
        # Flatten time dimension for traditional models
        X = X.mean(axis=1)
    
    # Load the model
    if is_dl_model:
        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create the model architecture
        input_shape = (X.shape[1], X.shape[2])
        model = create_1D_CNN_LSTM(input_shape=input_shape)
        
        # Load the trained weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        # Convert data to PyTorch tensor
        X_tensor = torch.FloatTensor(X).to(device)
        
        # Make predictions
        with torch.no_grad():
            outputs = model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            labels = predicted.cpu().numpy()
    else:
        model = joblib.load(model_path)
        labels = model.predict(X)
    
    return labels

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run EEG supervised learning workflow')
    parser.add_argument('--data_path', type=str, required=True, 
                      help='Path to EEG data directory or single CSV file')
    parser.add_argument('--output_dir', type=str, default='./models/supervised',
                      help='Directory to save models and results')
    parser.add_argument('--bundle_size', type=int, default=30,
                      help='Size of time series bundles')
    parser.add_argument('--step_size', type=int, default=15,
                      help='Step size for sliding window')
    parser.add_argument('--labeling_method', type=str, default='threshold',
                      choices=['threshold', 'cluster', 'mellow_concentration', 'time_periods'],
                      help='Method to create ground truth labels')
    parser.add_argument('--use_dl_model', action='store_true',
                      help='Train deep learning model')
    parser.add_argument('--channel', type=str, default='AF7',
                      help='EEG channel to use for threshold labeling')
    parser.add_argument('--threshold', type=float, default=None,
                      help='Threshold value for labeling (None for automatic)')
    parser.add_argument('--cluster_labels_path', type=str, default=None,
                      help='Path to cluster labels from unsupervised workflow')
    
    args = parser.parse_args()
    
    labeling_kwargs = {
        'channel': args.channel,
        'threshold': args.threshold,
        'cluster_labels_path': args.cluster_labels_path
    }
    
    run_supervised_workflow(
        data_path=args.data_path,
        output_dir=args.output_dir,
        bundle_size=args.bundle_size,
        step_size=args.step_size,
        labeling_method=args.labeling_method,
        use_dl_model=args.use_dl_model,
        **labeling_kwargs
    )