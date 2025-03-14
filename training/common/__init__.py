"""
Common modules for EEG classification training.
"""

# Import utils first as other modules depend on it
from .utils import logger, setup_logger

# Import core modules
from .data_loading import load_csv, load_eeg_data
from .preprocessing import (
    preprocess_eeg_data, engineer_eeg_features, select_features,
    sample_by_sliding_window, process_eeg_files_with_sampling,
    apply_moving_average, process_features, analyze_data_quality, analyze_data_quality
)
from .bundles import (
    create_coherent_time_series_bundles, normalize_bundles, split_bundles_train_test,
    create_coherent_time_series_bundles_disk, normalize_bundles_disk, split_bundles_disk,
    DiskBundleLoader
)
from .visualization import (
    visualize_eeg_data, plot_bundle_examples, plot_feature_importance,
    plot_confusion_matrix, plot_learning_curves
)

# Import model modules
from .supervised import SupervisedModelTrainer
from .unsupervised import UnsupervisedModelTrainer
from .mental_state_inference import MentalStateInference