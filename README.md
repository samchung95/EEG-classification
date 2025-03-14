# EEG Classification

A Python library for preprocessing, analyzing, and classifying EEG data for mental state detection.

## Overview

This project provides tools for:
- Loading and preprocessing EEG signal data
- Performing feature extraction and engineering
- Training supervised and unsupervised machine learning models
- Inferring mental states from EEG signals

## Installation

```bash
git clone https://github.com/username/EEG-classification.git
cd EEG-classification
pip install -r requirements.txt
```

## Usage Examples

### Data Loading and Preprocessing

```python
from training.common.helpers import load_csv, process_features

# Load data
df = load_csv('path/to/eeg_data.csv')

# Preprocess data
df_features = process_features(df)
```

### Supervised Learning

```python
import pandas as pd
from training.common.supervised import SupervisedModelTrainer

# Load your processed dataset
data = pd.read_csv('processed_eeg_data.csv')

# Initialize the trainer
trainer = SupervisedModelTrainer(
    model_type='random_forest',
    random_state=42,
    config={'n_estimators': 100}
)

# Prepare data
X_train, X_test, y_train, y_test = trainer.prepare_data(
    data=data,
    target_column='mental_state',
    test_size=0.2
)

# Train the model
model = trainer.train(X_train, y_train)

# Evaluate the model
evaluation = trainer.evaluate(
    X_test, 
    y_test,
    class_names=['relaxed', 'focused', 'stressed']
)

print(f"Accuracy: {evaluation['accuracy']:.4f}")

# Save the model
trainer.save_model('models/eeg_classifier.joblib')
```

### Unsupervised Learning

```python
import pandas as pd
from training.common.unsupervised import UnsupervisedModelTrainer

# Load your processed dataset
data = pd.read_csv('processed_eeg_data.csv')

# Initialize the trainer
trainer = UnsupervisedModelTrainer(
    model_type='kmeans',
    random_state=42
)

# Preprocess data
X = trainer.preprocess_data(data)

# Reduce dimensions for visualization
X_reduced = trainer.reduce_dimensions(X, n_components=2)

# Train the model with automatic cluster determination
labels = trainer.train(X, auto_clusters=True)

# Evaluate the clustering
metrics = trainer.evaluate(X, labels)
print(f"Silhouette score: {metrics.get('silhouette_score', 'N/A')}")

# Save the model
trainer.save_model('models/eeg_clusters.joblib')
```

### Mental State Inference

```python
import pandas as pd
from training.common.mental_state_inference import MentalStateInference

# Initialize the inference engine
inference = MentalStateInference(
    model_path='models/eeg_classifier.joblib'
)

# Load the trained model
inference.load_model()

# Load and preprocess new data
new_data = pd.read_csv('new_eeg_recording.csv')
preprocessed_data = inference.preprocess_data(new_data)

# Make predictions
results = inference.predict(preprocessed_data)

if results['status'] == 'success':
    print(f"Predicted mental state: {results['prediction']}")
    print(f"Confidence: {results['confidence']:.2f}")
```

## Project Structure

- `training/common/`: Core modules for data processing and modeling
  - `helpers.py`: Utility functions for data loading and preprocessing
  - `supervised.py`: Classes for supervised learning
  - `unsupervised.py`: Classes for unsupervised learning
  - `mental_state_inference.py`: Classes for model inference

## Data Processing Pipeline

1. **Data Loading**: Load raw EEG data from CSV files
2. **Preprocessing**: 
   - Handle missing values
   - Filter noise
   - Normalize signals
3. **Feature Engineering**:
   - Extract time-domain features
   - Extract frequency-domain features
   - Calculate statistical measures
4. **Model Training**:
   - Train supervised models for classification
   - Train unsupervised models for pattern discovery
5. **Inference**:
   - Apply trained models to new data
   - Interpret and explain results

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
