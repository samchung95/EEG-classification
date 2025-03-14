"""
Common utility functions for EEG classification.
"""
import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, List, Tuple, Any

def calculate_samples_from_seconds(df: pd.DataFrame, seconds: float) -> int:
    """
    Calculate the number of samples that correspond to a given time in seconds.
    
    Args:
        df: DataFrame with time series data
        seconds: Time in seconds
        
    Returns:
        Number of samples
    """
    # Check if we have a timestamp column
    if 'TimeStamp' not in df.columns:
        return int(seconds)  # Default to seconds if no timestamp
    
    # Get timestamps
    timestamps = pd.to_datetime(df['TimeStamp'])
    
    # Calculate average time difference between samples
    time_diffs = timestamps.diff().dropna()
    if len(time_diffs) == 0:
        return int(seconds)  # Default to seconds if can't calculate
    
    # Calculate average time per sample in seconds
    avg_time_per_sample = time_diffs.mean().total_seconds()
    
    # Calculate number of samples
    if avg_time_per_sample > 0:
        samples = int(seconds / avg_time_per_sample)
    else:
        samples = int(seconds)  # Default if time per sample is zero
    
    return max(1, samples)  # At least 1 sample 