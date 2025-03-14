#!/usr/bin/env python
# Demo script for mental state inference from EEG data

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from common.mental_state_inference import (
    generate_mental_state_mapping,
    EEGRealTimePredictor,
    real_time_eeg_demo
)

def main():
    parser = argparse.ArgumentParser(description='EEG Mental State Inference Demo')
    parser.add_argument('--model_dir', type=str, default='./models/clustering',
                      help='Directory containing clustering model')
    parser.add_argument('--data_path', type=str, default='../data',
                      help='Path to sample EEG data (file or directory)')
    parser.add_argument('--buffer_size', type=int, default=30,
                      help='Buffer size for real-time prediction')
    parser.add_argument('--step_size', type=int, default=5,
                      help='Step size for sliding window')
    parser.add_argument('--duration', type=int, default=60,
                      help='Duration of real-time demo in seconds')
    parser.add_argument('--mapping_method', type=str, default='feature_based',
                      choices=['feature_based', 'spectral_ratio'],
                      help='Method for mapping clusters to mental states')
    parser.add_argument('--demo', action='store_true',
                      help='Run real-time demo')
    
    args = parser.parse_args()
    
    print("EEG Mental State Inference Demo")
    print(f"Model directory: {args.model_dir}")
    print(f"Data path: {args.data_path}")
    
    # Step 1: Generate mapping from clusters to mental states
    print("\nStep 1: Generating mental state mapping...")
    mental_state_mapping = generate_mental_state_mapping(
        model_dir=args.model_dir,
        data_sample_path=args.data_path,
        method=args.mapping_method
    )
    
    print(f"\nMental state mapping: {mental_state_mapping}")
    
    # Step 2: Initialize real-time predictor
    print("\nStep 2: Initializing real-time predictor...")
    predictor = EEGRealTimePredictor(
        model_path=args.model_dir,
        buffer_size=args.buffer_size,
        step_size=args.step_size,
        mental_state_mapping=mental_state_mapping
    )
    
    # Step 3: Run real-time demo if requested
    if args.demo:
        print("\nStep 3: Running real-time demo...")
        real_time_eeg_demo(
            predictor=predictor,
            data_path=args.data_path,
            duration_seconds=args.duration
        )
    else:
        print("\nSkipping real-time demo. Use --demo to run the demo.")
        print("You can now use the predictor in your own application.")
        print("\nExample usage:")
        print("  # Add new samples to the buffer")
        print("  predictor.add_sample(sample_data)")
        print("  # Make a prediction when buffer is full")
        print("  prediction = predictor.predict()")
        print("  print(f\"Mental state: {prediction['state']}\")")

if __name__ == "__main__":
    main() 