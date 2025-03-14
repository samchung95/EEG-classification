# EEG Classification Common Modules

This directory contains core modules for EEG data processing, feature engineering, and model training.

## Module Structure

### Core Modules

- **data_loading.py**: Functions for loading EEG data from various sources
  - `load_csv`: Load CSV files with proper handling of mixed data types
  - `load_eeg_data`: Load EEG data from files or directories

- **preprocessing.py**: Functions for preprocessing and feature engineering
  - `preprocess_eeg_data`: Clean and prepare raw EEG data
  - `engineer_eeg_features`: Generate features from EEG signals
  - `select_features`: Identify and select important features
  - `apply_moving_average`: Apply smoothing to EEG signals

- **bundles.py**: Functions for time series bundle management
  - `create_coherent_time_series_bundles`: Create time-aligned data bundles
  - `normalize_bundles`: Normalize bundle data
  - `split_bundles_train_test`: Split bundles into training and test sets
  - `DiskBundleLoader`: Memory-efficient loader for bundles stored on disk

- **visualization.py**: Functions for data visualization
  - `visualize_eeg_data`: Plot EEG time series data
  - `plot_bundle_examples`: Visualize examples of time series bundles
  - `plot_feature_importance`: Display feature importance from models
  - `plot_confusion_matrix`: Visualize classification performance
  - `plot_learning_curves`: Show training progress over time

### Model Modules

- **supervised.py**: Classes and functions for supervised learning
  - `SupervisedModelTrainer`: Comprehensive trainer for classification models

- **unsupervised.py**: Classes and functions for unsupervised learning
  - `UnsupervisedModelTrainer`: Clustering and pattern discovery 

- **mental_state_inference.py**: Classes for model inference
  - `MentalStateInference`: Apply trained models to new data
  - `EEGRealTimePredictor`: Real-time prediction from EEG signals

### Utilities

- **utils/logging.py**: Logging configuration and utilities
  - `logger`: Shared logger instance
  - `setup_logger`: Configure custom loggers

- **utils/common.py**: Shared utility functions
  - `calculate_samples_from_seconds`: Convert time to sample counts

## Usage Examples

See the main README.md file in the project root directory for comprehensive usage examples.

## Import Structure

All major functions and classes are available directly from the `training.common` namespace:

```python
from training.common import (
    # Data loading
    load_csv, load_eeg_data,
    
    # Preprocessing
    preprocess_eeg_data, engineer_eeg_features, select_features,
    
    # Bundles
    create_coherent_time_series_bundles, normalize_bundles, 
    split_bundles_train_test, DiskBundleLoader,
    
    # Models
    SupervisedModelTrainer, UnsupervisedModelTrainer, MentalStateInference,
    
    # Utilities
    logger, setup_logger
)
``` 