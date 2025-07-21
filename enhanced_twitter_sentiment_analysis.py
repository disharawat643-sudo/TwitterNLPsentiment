#!/usr/bin/env python3
"""
Enhanced Twitter Sentiment Analysis
===================================

This is an enhanced version of the original twitter_sentiment_analysis.py
that properly saves both the model and vectorizer for web interface use.

Based on the original notebook but adapted for proper model persistence.

Authors: Team 3 Dude's
- Sarthak Singh (Project Lead & ML Engineer)  
- Himanshu Majumdar (Data Scientist & NLP Specialist)
- Samit Singh Bag (ML Engineer & Data Analyst)
- Sahil Raghav (Software Developer & Visualization Expert)
"""

import numpy as np
import pandas as pd
import re
import pickle
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk

# Download NLTK data
nltk.download('stopwords', quiet=True)

class EnhancedTwitterSentimentAnalyzer:
    """
    Enhanced version of the original Twitter sentiment analyzer
    that properly saves and loads both model and vectorizer.
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        # Original preprocessing components
        self.pattern = re.compile('[^a-zA-Z]')
        self.english_stopwords = stopwords.words('english')
        self.port_stemmer = PorterStemmer()
        
        # Model components
        self.model = None
        self.vectorizer = None
        
    def stemming(self, content):
        """
        Original stemming function from the notebook.
        
        Args:
            content (str): Text content to stem
            
        Returns:
            str: Stemmed content
        """
        if not isinstance(content, str):
            return ""
            
        stemmed_content = re.sub(self.pattern, ' ', content)
        stemmed_content = stemmed_content.lower()
        stemmed_content = stemmed_content.split()
        stemmed_content = [self.port_stemmer.stem(word) for word in stemmed_content if not word in self.english_stopwords]
        stemmed_content = ' '.join(stemmed_content)
        
        return stemmed_content
    
    def create_sample_dataset(self, size=1000):
        """
        Create a sample dataset for testing when the full dataset is not available.
        
        Args:
            size (int): Number of samples to create
            
        Returns:
            pd.DataFrame: Sample dataset
        """
        print("📝 Creating sample dataset for testing...")
        
        # Sample positive tweets
        positive_samples = [
            "I love this product so much!",
            "This is absolutely amazing!",
            "Best purchase I've ever made!",
            "So happy with this service!",
            "Fantastic quality and great value!",
            "Exceeded my expectations completely!",
            "Would definitely recommend to others!",
            "Outstanding customer service!",
            "This made my day so much better!",
            "Perfect exactly what I needed!",
            "Great experience from start to finish!",
            "Love the design and functionality!",
            "This is incredible and worth every penny!",
            "Such a wonderful surprise!",
            "Absolutely thrilled with my purchase!",
        ] * (size // 30)
        
        # Sample negative tweets  
        negative_samples = [
            "I hate this product so much!",
            "This is absolutely terrible!",
            "Worst purchase I've ever made!",
            "So disappointed with this service!",
            "Poor quality and bad value!",
            "Did not meet my expectations at all!",
            "Would never recommend to others!",
            "Terrible customer service!",
            "This ruined my day completely!",
            "Useless not what I needed!",
            "Bad experience from start to finish!",
            "Hate the design and functionality!",
            "This is awful and not worth it!",
            "Such a disappointing experience!",
            "Completely unsatisfied with my purchase!",
        ] * (size // 30)
        
        # Ensure we have exactly the right number of samples
        pos_count = size // 2
        neg_count = size // 2
        
        # Extend samples if needed
        while len(positive_samples) < pos_count:
            positive_samples.extend(positive_samples[:min(15, pos_count - len(positive_samples))])
        while len(negative_samples) < neg_count:
            negative_samples.extend(negative_samples[:min(15, neg_count - len(negative_samples))])
        
        # Create DataFrame
        texts = positive_samples[:pos_count] + negative_samples[:neg_count]
        targets = [1] * pos_count + [0] * neg_count
        
        # Create additional columns to match original format
        data = {
            'target': targets,
            'id': range(len(texts)),
            'date': ['2025-01-01'] * len(texts),
            'flag': ['NO_QUERY'] * len(texts),
            'user': ['sample_user'] * len(texts),
            'text': texts
        }
        
        df = pd.DataFrame(data)
        
        # Shuffle the dataset
        df = df.sample(frac=1).reset_index(drop=True)
        
        print(f"✅ Created sample dataset with {len(df)} samples")
        print(f"   - Positive samples: {sum(df['target'] == 1)}")
        print(f"   - Negative samples: {sum(df['target'] == 0)}")
        
        return df
    
    def load_or_create_dataset(self, csv_path=None):
        """
        Load the original dataset or create a sample one if not available.
        
        Args:
            csv_path (str): Path to the original training.1600000.processed.noemoticon.csv
            
        Returns:
            pd.DataFrame: Dataset for training
        """
        if csv_path and os.path.exists(csv_path):
            print(f"📂 Loading original dataset from {csv_path}")
            columns = ['target', 'id', 'date', 'flag', 'user', 'text']
            df = pd.read_csv(csv_path, names=columns, encoding='ISO-8859-1')
            
            # Convert target 4 to 1 (positive sentiment)
            df.replace({'target': {4: 1}}, inplace=True)
            
            print(f"✅ Loaded original dataset with {len(df)} samples")
            return df
        else:
            print("⚠️  Original dataset not found. Using sample dataset.")
            return self.create_sample_dataset(10000)  # Create larger sample
    
    def train_model(self, csv_path=None, test_size=0.2, random_state=2):
        """
        Train the sentiment analysis model using the original methodology.
        
        Args:
            csv_path (str): Path to the training CSV file
            test_size (float): Test split ratio
            random_state (int): Random state for reproducibility
            
        Returns:
            dict: Training results and metrics
        """
        print("🚀 Starting Enhanced Twitter Sentiment Analysis Training")
        print("=" * 60)
        
        # Load dataset
        twitter_data = self.load_or_create_dataset(csv_path)
        
        print(f"📊 Dataset Info:")
        print(f"   - Total samples: {len(twitter_data)}")
        print(f"   - Positive samples: {sum(twitter_data['target'] == 1)}")
        print(f"   - Negative samples: {sum(twitter_data['target'] == 0)}")
        
        # Apply stemming (original preprocessing)
        print("🔄 Applying stemming preprocessing...")
        twitter_data['stemmed_content'] = twitter_data['text'].apply(self.stemming)
        
        # Separating data and labels
        X = twitter_data['stemmed_content'].values
        Y = twitter_data['target'].values
        
        print(f"✅ Preprocessing complete")
        
        # Splitting data
        print("🔄 Splitting data into training and testing sets...")
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, stratify=Y, random_state=random_state
        )
        
        print(f"   - Training samples: {len(X_train)}")
        print(f"   - Testing samples: {len(X_test)}")
        
        # Vectorization
        print("🔄 Creating TF-IDF vectors...")
        self.vectorizer = TfidfVectorizer(max_features=10000, lowercase=True)
        
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        print(f"✅ Vectorization complete")
        print(f"   - Feature dimensions: {X_train_vectorized.shape[1]}")
        
        # Training the model
        print("🔄 Training Logistic Regression model...")
        self.model = LogisticRegression(max_iter=1000, random_state=random_state)
        self.model.fit(X_train_vectorized, Y_train)
        
        print("✅ Model training complete!")
        
        # Model evaluation
        print("📈 Evaluating model performance...")
        
        # Training accuracy
        X_train_prediction = self.model.predict(X_train_vectorized)
        training_accuracy = accuracy_score(Y_train, X_train_prediction)
        
        # Test accuracy  
        X_test_prediction = self.model.predict(X_test_vectorized)
        test_accuracy = accuracy_score(Y_test, X_test_prediction)
        
        print(f"✅ Training accuracy: {training_accuracy:.4f}")
        print(f"✅ Test accuracy: {test_accuracy:.4f}")
        
        # Detailed metrics
        classification_rep = classification_report(
            Y_test, X_test_prediction,
            target_names=['Negative', 'Positive'],
            output_dict=True
        )
        
        confusion_mat = confusion_matrix(Y_test, X_test_prediction)
        
        print("\n📊 Detailed Classification Report:")
        print(classification_report(Y_test, X_test_prediction, target_names=['Negative', 'Positive']))
        
        print("\n📊 Confusion Matrix:")
        print("     Predicted")
        print("     Neg  Pos")
        print(f"Neg  {confusion_mat[0][0]:4d} {confusion_mat[0][1]:4d}")
        print(f"Pos  {confusion_mat[1][0]:4d} {confusion_mat[1][1]:4d}")
        
        results = {
            'training_accuracy': training_accuracy,
            'test_accuracy': test_accuracy,
            'classification_report': classification_rep,
            'confusion_matrix': confusion_mat,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        return results
    
    def save_complete_model(self, model_path='models/twitter_sentiment_model.pkl', 
                           vectorizer_path='models/twitter_sentiment_vectorizer.pkl'):
        """
        Save both the model and vectorizer for web interface use.
        
        Args:
            model_path (str): Path to save the model
            vectorizer_path (str): Path to save the vectorizer
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model and vectorizer must be trained first!")
        
        # Create models directory
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✅ Model saved to {model_path}")
        
        # Save vectorizer
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print(f"✅ Vectorizer saved to {vectorizer_path}")
        
        # Also save in original format for compatibility
        original_model_path = 'trained_model.sav'
        with open(original_model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✅ Model also saved as {original_model_path} (original format)")
    
    def load_complete_model(self, model_path='models/twitter_sentiment_model.pkl', 
                           vectorizer_path='models/twitter_sentiment_vectorizer.pkl'):
        """
        Load both the model and vectorizer.
        
        Args:
            model_path (str): Path to the model file
            vectorizer_path (str): Path to the vectorizer file
        """
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"✅ Model loaded from {model_path}")
        
        # Load vectorizer
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        print(f"✅ Vectorizer loaded from {vectorizer_path}")
    
    def predict(self, text):
        """
        Predict sentiment for given text.
        
        Args:
            text (str or list): Text to analyze
            
        Returns:
            int or list: Sentiment prediction(s) (0=Negative, 1=Positive)
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model and vectorizer must be loaded first!")
        
        if isinstance(text, str):
            processed = self.stemming(text)
            vectorized = self.vectorizer.transform([processed])
            prediction = self.model.predict(vectorized)
            return int(prediction[0])
        else:
            processed = [self.stemming(t) for t in text]
            vectorized = self.vectorizer.transform(processed)
            predictions = self.model.predict(vectorized)
            return [int(p) for p in predictions]
    
    def predict_proba(self, text):
        """
        Predict sentiment probabilities for given text.
        
        Args:
            text (str or list): Text to analyze
            
        Returns:
            array or list: Sentiment probabilities
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model and vectorizer must be loaded first!")
        
        if isinstance(text, str):
            processed = self.stemming(text)
            vectorized = self.vectorizer.transform([processed])
            probabilities = self.model.predict_proba(vectorized)
            return probabilities[0]
        else:
            processed = [self.stemming(t) for t in text]
            vectorized = self.vectorizer.transform(processed)
            probabilities = self.model.predict_proba(vectorized)
            return [prob for prob in probabilities]
    
    def get_sentiment_label(self, prediction: int) -> str:
        """
        Convert numerical prediction to sentiment label.
        
        Args:
            prediction (int): Numerical prediction (0 or 1)
            
        Returns:
            str: Sentiment label ('Negative' or 'Positive')
        """
        return 'Positive' if prediction == 1 else 'Negative'
    
    def preprocess_text(self, text: str) -> str:
        """
        Alias for stemming method to maintain API compatibility.
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        return self.stemming(text)
    
    def test_predictions(self):
        """Test the model with sample predictions."""
        if self.model is None or self.vectorizer is None:
            print("❌ Model not loaded. Please train or load model first.")
            return
        
        test_texts = [
            "I love this product so much!",
            "This is terrible and awful",
            "Amazing quality, highly recommend!",
            "Worst experience ever, hate it",
            "Pretty good, could be better",
            "Absolutely fantastic service!"
        ]
        
        print("\n🧪 Testing Model Predictions:")
        print("=" * 50)
        
        for text in test_texts:
            prediction = self.predict(text)
            probabilities = self.predict_proba(text)
            label = "Positive" if prediction == 1 else "Negative"
            confidence = max(probabilities)
            
            print(f"Text: '{text}'")
            print(f"   → {label} (confidence: {confidence:.3f})")
            print(f"   → Probabilities: [Neg: {probabilities[0]:.3f}, Pos: {probabilities[1]:.3f}]")
            print()


def main():
    """Main function to train and save the enhanced model."""
    # Initialize analyzer
    analyzer = EnhancedTwitterSentimentAnalyzer()
    
    # Look for original dataset
    dataset_paths = [
        'training.1600000.processed.noemoticon.csv',
        'data/training.1600000.processed.noemoticon.csv',
        '../data/training.1600000.processed.noemoticon.csv'
    ]
    
    dataset_path = None
    for path in dataset_paths:
        if os.path.exists(path):
            dataset_path = path
            break
    
    # Train the model
    results = analyzer.train_model(csv_path=dataset_path)
    
    # Save the complete model (both model and vectorizer)
    analyzer.save_complete_model()
    
    # Test predictions
    analyzer.test_predictions()
    
    print("\n🎉 Enhanced Twitter Sentiment Analysis Complete!")
    print("✅ Model and vectorizer saved for web interface use")
    print("💡 You can now use the web interface with the properly trained model")


if __name__ == "__main__":
    main()
