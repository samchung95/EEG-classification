"""
Functions for visualizing EEG data and analysis results.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any

from .utils.logging import logger

def visualize_eeg_data(df: pd.DataFrame, 
                      feature_cols: Optional[List[str]] = None, 
                      max_cols: int = 10, 
                      time_window: Optional[Tuple[str, str]] = None) -> plt.Figure:
    """
    Visualize EEG data as time series.
    
    Args:
        df: DataFrame containing EEG data
        feature_cols: List of feature columns to plot (None for auto-selection)
        max_cols: Maximum number of columns to plot
        time_window: Optional tuple of (start_time, end_time) to limit the view
        
    Returns:
        Matplotlib figure object
    """
    # Handle feature columns
    if feature_cols is None:
        # Auto-select numeric columns
        exclude_cols = ['FileID', 'SourceFile']
        feature_cols = [col for col in df.select_dtypes(include=['number']).columns 
                       if col not in exclude_cols]
    
    # Limit to max_cols
    if len(feature_cols) > max_cols:
        logger.warning(f"Too many columns ({len(feature_cols)}). "
                     f"Showing only the first {max_cols}.")
        feature_cols = feature_cols[:max_cols]
    
    # Create figure
    n_cols = len(feature_cols)
    fig, axes = plt.subplots(n_cols, 1, figsize=(12, n_cols * 2), sharex=True)
    
    # Handle single column case
    if n_cols == 1:
        axes = [axes]
    
    # Create x-axis values (time)
    if 'TimeStamp' in df.columns:
        x = pd.to_datetime(df['TimeStamp'])
        
        # Apply time window if specified
        if time_window:
            start_time, end_time = time_window
            mask = (x >= start_time) & (x <= end_time)
            x = x[mask]
            df = df[mask]
    else:
        x = np.arange(len(df))
    
    # Plot each feature
    for i, col in enumerate(feature_cols):
        axes[i].plot(x, df[col])
        axes[i].set_ylabel(col)
        axes[i].grid(True)
    
    # Set common x label
    if 'TimeStamp' in df.columns:
        axes[-1].set_xlabel('Time')
    else:
        axes[-1].set_xlabel('Sample')
    
    # Set title
    title = "EEG Data Visualization"
    if 'SourceFile' in df.columns:
        title += f" - {df['SourceFile'].iloc[0]}"
    fig.suptitle(title)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_bundle_examples(X_bundles: np.ndarray, 
                        metadata_df: pd.DataFrame, 
                        num_examples: int = 3, 
                        max_features: int = 5) -> plt.Figure:
    """
    Plot examples of time series bundles.
    
    Args:
        X_bundles: Bundle data array
        metadata_df: Metadata for bundles
        num_examples: Number of example bundles to plot
        max_features: Maximum number of features to plot per bundle
        
    Returns:
        Matplotlib figure object
    """
    # Get bundle dimensions
    n_bundles, bundle_size, n_features = X_bundles.shape
    
    # Select random examples
    np.random.seed(42)  # For reproducibility
    example_indices = np.random.choice(n_bundles, min(num_examples, n_bundles), replace=False)
    
    # Limit features
    feature_indices = list(range(min(max_features, n_features)))
    
    # Create figure
    fig, axes = plt.subplots(num_examples, 1, figsize=(12, num_examples * 3))
    
    # Handle single example case
    if num_examples == 1:
        axes = [axes]
    
    # Plot each example
    for i, idx in enumerate(example_indices):
        bundle = X_bundles[idx]
        ax = axes[i]
        
        # Get metadata
        meta = metadata_df.iloc[idx]
        
        # Create x-axis values (time steps)
        x = np.arange(bundle_size)
        
        # Plot selected features
        for j in feature_indices:
            ax.plot(x, bundle[:, j], label=f"Feature {j}")
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Add metadata as title
        title = f"Bundle {idx}"
        if 'file_id' in meta:
            title += f" - File: {meta['file_id']}"
        if 'start_time' in meta and 'end_time' in meta:
            try:
                # Format timestamps
                start = pd.to_datetime(meta['start_time']).strftime('%H:%M:%S')
                end = pd.to_datetime(meta['end_time']).strftime('%H:%M:%S')
                title += f" - Time: {start} to {end}"
            except:
                pass
                
        ax.set_title(title)
        ax.grid(True)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_feature_importance(feature_importance: np.ndarray, 
                           feature_names: Optional[List[str]] = None,
                           top_n: int = 20) -> plt.Figure:
    """
    Plot feature importance from a trained model.
    
    Args:
        feature_importance: Array of feature importance values
        feature_names: List of feature names (if None, uses indices)
        top_n: Number of top features to display
        
    Returns:
        Matplotlib figure object
    """
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(feature_importance))]
    else:
        # Ensure we have the right number of names
        if len(feature_names) != len(feature_importance):
            logger.warning(f"Feature names length ({len(feature_names)}) doesn't match "
                         f"feature importance length ({len(feature_importance)})")
            feature_names = [f"Feature {i}" for i in range(len(feature_importance))]
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Limit to top_n
    if len(importance_df) > top_n:
        importance_df = importance_df.head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot horizontal bar chart
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
    
    # Add labels and title
    ax.set_title(f'Top {top_n} Feature Importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_confusion_matrix(confusion_matrix: np.ndarray,
                         class_names: Optional[List[str]] = None,
                         normalize: bool = True) -> plt.Figure:
    """
    Plot a confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        normalize: Whether to normalize by row
        
    Returns:
        Matplotlib figure object
    """
    # Create class names if not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(confusion_matrix))]
    
    # Normalize if requested
    if normalize:
        # Normalize by row (true label)
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        norm_cm = confusion_matrix / row_sums
        title = 'Normalized Confusion Matrix'
        fmt = '.2f'
    else:
        norm_cm = confusion_matrix
        title = 'Confusion Matrix'
        fmt = 'd'
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    im = ax.imshow(norm_cm, interpolation='nearest', cmap=plt.cm.Blues)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    
    # Add ticks and labels
    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotate x tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    thresh = norm_cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(norm_cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if norm_cm[i, j] > thresh else "black")
    
    # Adjust layout
    fig.tight_layout()
    
    return fig

def plot_learning_curves(train_scores: List[float],
                        validation_scores: List[float],
                        epochs: Optional[List[int]] = None) -> plt.Figure:
    """
    Plot learning curves from model training.
    
    Args:
        train_scores: List of training scores
        validation_scores: List of validation scores
        epochs: List of epoch numbers (if None, uses indices)
        
    Returns:
        Matplotlib figure object
    """
    # Create epoch values if not provided
    if epochs is None:
        epochs = list(range(1, len(train_scores) + 1))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot curves
    ax.plot(epochs, train_scores, label='Training Score', marker='o')
    ax.plot(epochs, validation_scores, label='Validation Score', marker='s')
    
    # Add labels and title
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Learning Curves')
    
    # Add grid and legend
    ax.grid(True)
    ax.legend()
    
    # Adjust layout
    fig.tight_layout()
    
    return fig 