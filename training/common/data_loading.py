"""
Functions for loading EEG data from various sources.
"""
import os
import glob
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

from .utils.logging import logger

def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file with proper handling of mixed data types.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        pandas DataFrame with the CSV data
        
    Raises:
        FileNotFoundError: If the file does not exist
        pd.errors.EmptyDataError: If the file is empty
        pd.errors.ParserError: If the file cannot be parsed
    """
    logger.debug(f"Loading CSV file: {file_path}")
    try:
        # Set low_memory=False to avoid DtypeWarning for mixed types
        df = pd.read_csv(file_path, low_memory=False)
        logger.debug(f"Successfully loaded CSV with {len(df)} rows")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"Empty CSV file: {file_path}")
        raise
    except pd.errors.ParserError:
        logger.error(f"Failed to parse CSV file: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading CSV file: {file_path}, Error: {str(e)}")
        raise

def load_eeg_data(directory_path: Optional[str] = None, 
                 csv_files: Optional[List[str]] = None, 
                 single_file: Optional[str] = None) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Load EEG data from specified source.
    
    Args:
        directory_path: Path to directory containing CSV files
        csv_files: List of specific CSV file paths
        single_file: Path to a single CSV file
        
    Returns:
        Tuple of (file_dfs, combined_df) where:
          - file_dfs is a dictionary mapping file IDs to DataFrames
          - combined_df is a concatenated DataFrame with a 'FileID' column
    """
    file_dfs = {}
    total_rows = 0
    
    # Load from directory
    if directory_path:
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
            
        csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {directory_path}")
            
    # Load from list of files
    elif csv_files:
        if not csv_files:
            raise ValueError("Empty list of CSV files provided")
            
    # Load from single file
    elif single_file:
        if not os.path.exists(single_file):
            raise FileNotFoundError(f"File not found: {single_file}")
            
        csv_files = [single_file]
        
    else:
        raise ValueError("No data source specified. Provide directory_path, csv_files, or single_file.")
    
    # Load each file
    for i, file_path in enumerate(csv_files):
        try:
            file_id = f"file_{i+1}"
            df = load_csv(file_path)
            
            # Add file path as metadata
            df['SourceFile'] = os.path.basename(file_path)
            
            # Store in dictionary
            file_dfs[file_id] = df
            total_rows += len(df)
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
    
    if not file_dfs:
        raise ValueError("No data was successfully loaded")
    
    # Combine all files into a single DataFrame
    combined_df = pd.concat(
        [df.assign(FileID=file_id) for file_id, df in file_dfs.items()],
        ignore_index=True
    )
    
    logger.info(f"Loaded {len(file_dfs)} files with {total_rows} total rows")
    
    return file_dfs, combined_df 