#!/usr/bin/env python3
"""
Twitter Sentiment Analysis - Data Download Script
==================================================

This script downloads the Sentiment140 dataset from Kaggle and prepares it for training.

Authors: Team 3 Dude's
- Sarthak Singh (Project Lead & ML Engineer)
- Himanshu Majumdar (Data Scientist & NLP Specialist)
- Samit Singh Bag (ML Engineer & Data Analyst)
- Sahil Raghav (Software Developer & Visualization Expert)

Requirements:
1. Kaggle account with API access
2. kaggle.json file in the project root or ~/.kaggle/
3. kaggle package installed (pip install kaggle)

Usage:
    python scripts/download_data.py
"""

import os
import sys
import zipfile
import logging
import argparse
from pathlib import Path
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_kaggle_credentials():
    """Set up Kaggle API credentials."""
    kaggle_json_paths = [
        'kaggle.json',  # Project root
        os.path.expanduser('~/.kaggle/kaggle.json'),  # Default location
        os.path.join(os.path.dirname(__file__), '..', 'kaggle.json')  # Parent directory
    ]
    
    kaggle_json_found = False
    for path in kaggle_json_paths:
        if os.path.exists(path):
            kaggle_json_found = True
            logger.info(f"Found kaggle.json at: {path}")
            
            # Ensure ~/.kaggle directory exists
            kaggle_dir = os.path.expanduser('~/.kaggle')
            os.makedirs(kaggle_dir, exist_ok=True)
            
            # Copy kaggle.json to ~/.kaggle/ if not already there
            default_kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
            if not os.path.exists(default_kaggle_json):
                import shutil
                shutil.copy2(path, default_kaggle_json)
                os.chmod(default_kaggle_json, 0o600)  # Set proper permissions
                logger.info(f"Copied kaggle.json to {default_kaggle_json}")
            break
    
    if not kaggle_json_found:
        logger.error("kaggle.json not found!")
        logger.info("Please download your kaggle.json from https://www.kaggle.com/settings")
        logger.info("and place it in one of these locations:")
        for path in kaggle_json_paths:
            logger.info(f"  - {path}")
        sys.exit(1)
    
    return True


def download_sentiment140_dataset(data_dir: str = "data"):
    """
    Download the Sentiment140 dataset from Kaggle.
    
    Args:
        data_dir (str): Directory to save the dataset
    """
    try:
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "raw"), exist_ok=True)
        
        # Check if dataset already exists
        dataset_file = os.path.join(data_dir, "training.1600000.processed.noemoticon.csv")
        if os.path.exists(dataset_file):
            logger.info(f"Dataset already exists at: {dataset_file}")
            return dataset_file
        
        logger.info("Downloading Sentiment140 dataset from Kaggle...")
        
        # Use subprocess to run kaggle command
        cmd = [
            "kaggle", "datasets", "download", 
            "-d", "kazanova/sentiment140",
            "-p", data_dir,
            "--unzip"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Kaggle download failed: {result.stderr}")
            # Try alternative approach
            logger.info("Trying alternative download method...")
            return download_with_python_api(data_dir)
        else:
            logger.info("Dataset downloaded successfully!")
            logger.info(f"Output: {result.stdout}")
        
        # Verify the file exists
        if os.path.exists(dataset_file):
            file_size = os.path.getsize(dataset_file) / (1024 * 1024)  # Size in MB
            logger.info(f"Dataset file: {dataset_file}")
            logger.info(f"File size: {file_size:.1f} MB")
            return dataset_file
        else:
            logger.error("Dataset file not found after download")
            return None
            
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        return None


def download_with_python_api(data_dir: str):
    """Download using Python Kaggle API."""
    try:
        import kaggle
        
        logger.info("Using Kaggle Python API...")
        kaggle.api.dataset_download_files(
            'kazanova/sentiment140',
            path=data_dir,
            unzip=True
        )
        
        dataset_file = os.path.join(data_dir, "training.1600000.processed.noemoticon.csv")
        if os.path.exists(dataset_file):
            logger.info("Download completed using Python API!")
            return dataset_file
        else:
            logger.error("Dataset file not found after Python API download")
            return None
            
    except ImportError:
        logger.error("Kaggle package not installed. Please run: pip install kaggle")
        return None
    except Exception as e:
        logger.error(f"Python API download failed: {str(e)}")
        return None


def extract_zip_file(zip_path: str, extract_to: str):
    """Extract ZIP file if needed."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"Extracted {zip_path} to {extract_to}")
        
        # Remove zip file after extraction
        os.remove(zip_path)
        logger.info(f"Removed zip file: {zip_path}")
        
    except Exception as e:
        logger.error(f"Error extracting zip file: {str(e)}")


def verify_dataset(dataset_path: str):
    """Verify the downloaded dataset."""
    try:
        import pandas as pd
        
        logger.info("Verifying dataset...")
        
        # Try to load the first few rows
        df_sample = pd.read_csv(
            dataset_path,
            names=['target', 'id', 'date', 'flag', 'user', 'text'],
            encoding='ISO-8859-1',
            nrows=1000
        )
        
        logger.info(f"Dataset shape (first 1000 rows): {df_sample.shape}")
        logger.info(f"Columns: {list(df_sample.columns)}")
        logger.info(f"Target distribution: {df_sample['target'].value_counts().to_dict()}")
        
        # Check for null values
        null_counts = df_sample.isnull().sum()
        if null_counts.sum() > 0:
            logger.warning(f"Found null values: {null_counts.to_dict()}")
        else:
            logger.info("No null values found in sample")
        
        logger.info("Dataset verification completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error verifying dataset: {str(e)}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Download Sentiment140 dataset from Kaggle')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Directory to save the dataset (default: data)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify the dataset after download'
    )
    
    args = parser.parse_args()
    
    logger.info("Starting Sentiment140 dataset download...")
    logger.info(f"Data directory: {args.data_dir}")
    
    # Setup Kaggle credentials
    if not setup_kaggle_credentials():
        sys.exit(1)
    
    # Download dataset
    dataset_path = download_sentiment140_dataset(args.data_dir)
    
    if dataset_path:
        logger.info(f"Dataset downloaded successfully to: {dataset_path}")
        
        if args.verify:
            if verify_dataset(dataset_path):
                logger.info("Dataset verification passed!")
            else:
                logger.warning("Dataset verification failed, but file exists")
        
        logger.info("\n" + "="*50)
        logger.info("DOWNLOAD COMPLETED SUCCESSFULLY!")
        logger.info("="*50)
        logger.info(f"Dataset location: {dataset_path}")
        logger.info("You can now run the training script:")
        logger.info(f"python scripts/train_model.py --data_path {dataset_path}")
        
    else:
        logger.error("Failed to download dataset")
        sys.exit(1)


if __name__ == "__main__":
    main()
