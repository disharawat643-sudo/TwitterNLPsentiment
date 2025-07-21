"""
Twitter Sentiment Analysis - Main Sentiment Analyzer Module
============================================================

A comprehensive sentiment analysis system for Twitter data using
Natural Language Processing and Machine Learning techniques.

Authors: Team 3 Dude's
- Sarthak Singh (Project Lead & ML Engineer)
- Himanshu Majumdar (Data Scientist & NLP Specialist)
- Samit Singh Bag (ML Engineer & Data Analyst)  
- Sahil Raghav (Software Developer & Visualization Expert)

License: MIT
"""

import os
import pickle
import numpy as np
import pandas as pd
import re
from typing import Union, List, Tuple, Optional
import logging
from pathlib import Path

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    A comprehensive sentiment analysis system for Twitter data.
    
    This class handles the complete pipeline from data preprocessing
    to model training and prediction for sentiment classification.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the SentimentAnalyzer.
        
        Args:
            model_path (str, optional): Path to pre-trained model file
        """
        self.model = None
        self.vectorizer = None
        self.stemmer = PorterStemmer()
        self.stopwords = set(stopwords.words('english'))
        self.pattern = re.compile('[^a-zA-Z]')
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text data for sentiment analysis.
        
        This method performs:
        - Special character removal
        - Lowercase conversion
        - Tokenization
        - Stopword removal
        - Stemming
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            return ""
            
        # Remove special characters and convert to lowercase
        processed_text = re.sub(self.pattern, ' ', text)
        processed_text = processed_text.lower()
        
        # Tokenize and process
        words = processed_text.split()
        
        # Remove stopwords and apply stemming
        processed_words = [
            self.stemmer.stem(word) 
            for word in words 
            if word not in self.stopwords
        ]
        
        return ' '.join(processed_words)
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and preprocess the Twitter sentiment dataset.
        
        Args:
            file_path (str): Path to the CSV dataset file
            
        Returns:
            pd.DataFrame: Loaded and preprocessed dataset
        """
        try:
            # Define column names for Sentiment140 dataset
            columns = ['target', 'id', 'date', 'flag', 'user', 'text']
            
            # Load dataset
            df = pd.read_csv(file_path, names=columns, encoding='ISO-8859-1')
            
            # Convert target 4 to 1 (positive sentiment)
            df.replace({'target': {4: 1}}, inplace=True)
            
            # Preprocess text data
            logger.info("Preprocessing text data...")
            df['processed_text'] = df['text'].apply(self.preprocess_text)
            
            logger.info(f"Dataset loaded successfully with {len(df)} samples")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels for model training.
        
        Args:
            df (pd.DataFrame): Preprocessed dataset
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features (X) and labels (y)
        """
        X = df['processed_text'].values
        y = df['target'].values
        
        return X, y
    
    def train_model(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.2, random_state: int = 42) -> dict:
        """
        Train the sentiment analysis model.
        
        Args:
            X (np.ndarray): Feature data
            y (np.ndarray): Target labels
            test_size (float): Proportion of test data
            random_state (int): Random state for reproducibility
            
        Returns:
            dict: Training results and metrics
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=10000,  # Limit features for efficiency
            lowercase=True,
            ngram_range=(1, 2)   # Use unigrams and bigrams
        )
        
        # Transform text data to numerical features
        logger.info("Vectorizing text data...")
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        # Initialize and train the model
        logger.info("Training logistic regression model...")
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight='balanced'  # Handle class imbalance
        )
        
        self.model.fit(X_train_vectorized, y_train)
        
        # Evaluate the model
        train_predictions = self.model.predict(X_train_vectorized)
        test_predictions = self.model.predict(X_test_vectorized)
        
        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        # Generate detailed metrics
        classification_rep = classification_report(
            y_test, test_predictions, 
            target_names=['Negative', 'Positive'],
            output_dict=True
        )
        
        confusion_mat = confusion_matrix(y_test, test_predictions)
        
        results = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'classification_report': classification_rep,
            'confusion_matrix': confusion_mat,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        logger.info(f"Training completed!")
        logger.info(f"Training Accuracy: {train_accuracy:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        
        return results
    
    def predict(self, text: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Predict sentiment for given text(s).
        
        Args:
            text (Union[str, List[str]]): Text or list of texts to analyze
            
        Returns:
            Union[int, List[int]]: Predicted sentiment(s) (0=Negative, 1=Positive)
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained or loaded. Please train or load a model first.")
        
        # Handle single string input
        if isinstance(text, str):
            processed_text = self.preprocess_text(text)
            vectorized_text = self.vectorizer.transform([processed_text])
            prediction = self.model.predict(vectorized_text)
            return int(prediction[0])
        
        # Handle list of strings
        elif isinstance(text, list):
            processed_texts = [self.preprocess_text(t) for t in text]
            vectorized_texts = self.vectorizer.transform(processed_texts)
            predictions = self.model.predict(vectorized_texts)
            return [int(p) for p in predictions]
        
        else:
            raise TypeError("Input must be a string or list of strings")
    
    def predict_proba(self, text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Predict sentiment probabilities for given text(s).
        
        Args:
            text (Union[str, List[str]]): Text or list of texts to analyze
            
        Returns:
            Union[np.ndarray, List[np.ndarray]]: Prediction probabilities
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained or loaded. Please train or load a model first.")
        
        # Handle single string input
        if isinstance(text, str):
            processed_text = self.preprocess_text(text)
            vectorized_text = self.vectorizer.transform([processed_text])
            probabilities = self.model.predict_proba(vectorized_text)
            return probabilities[0]
        
        # Handle list of strings
        elif isinstance(text, list):
            processed_texts = [self.preprocess_text(t) for t in text]
            vectorized_texts = self.vectorizer.transform(processed_texts)
            probabilities = self.model.predict_proba(vectorized_texts)
            return [prob for prob in probabilities]
        
        else:
            raise TypeError("Input must be a string or list of strings")
    
    def save_model(self, model_path: str, vectorizer_path: Optional[str] = None) -> None:
        """
        Save the trained model and vectorizer to disk.
        
        Args:
            model_path (str): Path to save the model
            vectorizer_path (str, optional): Path to save the vectorizer
        """
        if self.model is None:
            raise ValueError("No model to save. Please train a model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save vectorizer
        if vectorizer_path is None:
            vectorizer_path = model_path.replace('.pkl', '_vectorizer.pkl')
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Vectorizer saved to {vectorizer_path}")
    
    def load_model(self, model_path: str, vectorizer_path: Optional[str] = None) -> None:
        """
        Load a pre-trained model and vectorizer from disk.
        
        Args:
            model_path (str): Path to the saved model
            vectorizer_path (str, optional): Path to the saved vectorizer
        """
        # Load model
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load vectorizer
        if vectorizer_path is None:
            vectorizer_path = model_path.replace('.pkl', '_vectorizer.pkl')
        
        if os.path.exists(vectorizer_path):
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
        else:
            logger.warning(f"Vectorizer file not found: {vectorizer_path}")
        
        logger.info("Model and vectorizer loaded successfully")
    
    def get_sentiment_label(self, prediction: int) -> str:
        """
        Convert numerical prediction to sentiment label.
        
        Args:
            prediction (int): Numerical prediction (0 or 1)
            
        Returns:
            str: Sentiment label ('Negative' or 'Positive')
        """
        return 'Positive' if prediction == 1 else 'Negative'


# Example usage and testing
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Example predictions (requires pre-trained model)
    sample_tweets = [
        "I love this new product!",
        "This is terrible, worst experience ever",
        "Pretty good, could be better",
        "Absolutely fantastic! Highly recommend"
    ]
    
    try:
        # This would work if model is already trained/loaded
        for tweet in sample_tweets:
            sentiment = analyzer.predict(tweet)
            label = analyzer.get_sentiment_label(sentiment)
            print(f"'{tweet}' → {label}")
    except ValueError as e:
        print(f"Model not loaded: {e}")
        print("Please train or load a model first.")
