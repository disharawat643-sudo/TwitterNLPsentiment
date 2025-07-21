#!/usr/bin/env python3
"""
Twitter Sentiment Analysis - Model Training Script
==================================================

This script handles the complete training pipeline for the sentiment analysis model.

Authors: Team 3 Dude's
- Sarthak Singh (Project Lead & ML Engineer)
- Himanshu Majumdar (Data Scientist & NLP Specialist)
- Samit Singh Bag (ML Engineer & Data Analyst)
- Sahil Raghav (Software Developer & Visualization Expert)

Usage:
    python scripts/train_model.py --data_path data/training.1600000.processed.noemoticon.csv
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sentiment_analyzer import SentimentAnalyzer
import nltk

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
except:
    print("Warning: Could not download NLTK stopwords")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Twitter Sentiment Analysis Model')
    parser.add_argument(
        '--data_path', 
        type=str, 
        default='data/training.1600000.processed.noemoticon.csv',
        help='Path to the training dataset CSV file'
    )
    parser.add_argument(
        '--model_output', 
        type=str, 
        default='models/sentiment_model.pkl',
        help='Path to save the trained model'
    )
    parser.add_argument(
        '--test_size', 
        type=float, 
        default=0.2,
        help='Proportion of data to use for testing (default: 0.2)'
    )
    parser.add_argument(
        '--random_state', 
        type=int, 
        default=42,
        help='Random state for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        logger.info("Please run 'python scripts/download_data.py' first to download the dataset.")
        sys.exit(1)
    
    logger.info("Starting Twitter Sentiment Analysis Model Training")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Model output: {args.model_output}")
    logger.info(f"Test size: {args.test_size}")
    logger.info(f"Random state: {args.random_state}")
    
    try:
        # Initialize the sentiment analyzer
        analyzer = SentimentAnalyzer()
        
        # Load and preprocess the data
        logger.info("Loading and preprocessing data...")
        df = analyzer.load_data(args.data_path)
        
        # Prepare features and labels
        X, y = analyzer.prepare_features(df)
        
        logger.info(f"Dataset size: {len(X)} samples")
        logger.info(f"Positive samples: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
        logger.info(f"Negative samples: {len(y) - sum(y)} ({(len(y) - sum(y))/len(y)*100:.1f}%)")
        
        # Train the model
        logger.info("Training model...")
        results = analyzer.train_model(
            X, y, 
            test_size=args.test_size, 
            random_state=args.random_state
        )
        
        # Print results
        logger.info("\n" + "="*50)
        logger.info("TRAINING RESULTS")
        logger.info("="*50)
        logger.info(f"Training Accuracy: {results['train_accuracy']:.4f}")
        logger.info(f"Test Accuracy: {results['test_accuracy']:.4f}")
        logger.info(f"Training Set Size: {results['train_size']}")
        logger.info(f"Test Set Size: {results['test_size']}")
        
        # Print classification report
        logger.info("\nClassification Report:")
        logger.info("-" * 30)
        report = results['classification_report']
        for class_name, metrics in report.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                logger.info(f"{class_name:>10}: Precision={metrics['precision']:.3f}, "
                          f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        logger.info(f"Overall Accuracy: {report['accuracy']:.4f}")
        
        # Print confusion matrix
        logger.info("\nConfusion Matrix:")
        logger.info("-" * 20)
        cm = results['confusion_matrix']
        logger.info(f"True Negatives: {cm[0][0]}")
        logger.info(f"False Positives: {cm[0][1]}")
        logger.info(f"False Negatives: {cm[1][0]}")
        logger.info(f"True Positives: {cm[1][1]}")
        
        # Save the model
        logger.info(f"\nSaving model to {args.model_output}")
        analyzer.save_model(args.model_output)
        
        # Test with sample tweets
        logger.info("\nTesting with sample tweets:")
        logger.info("-" * 30)
        sample_tweets = [
            "I love this new product!",
            "This is terrible, worst experience ever",
            "Pretty good, could be better",
            "Absolutely fantastic! Highly recommend",
            "Not sure how I feel about this"
        ]
        
        for tweet in sample_tweets:
            prediction = analyzer.predict(tweet)
            probabilities = analyzer.predict_proba(tweet)
            sentiment_label = analyzer.get_sentiment_label(prediction)
            confidence = max(probabilities)
            
            logger.info(f"'{tweet[:50]}...' → {sentiment_label} "
                       f"(confidence: {confidence:.3f})")
        
        logger.info("\n" + "="*50)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*50)
        logger.info(f"Model saved to: {args.model_output}")
        logger.info("You can now use the trained model for predictions.")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
