# Twitter Sentiment Analysis 🐦📊

A comprehensive machine learning project for analyzing sentiment in Twitter data using Natural Language Processing techniques and Logistic Regression, complete with an interactive web interface.

## 🎯 Project Overview

This project implements a robust sentiment analysis system that can classify tweets as positive or negative. Using the Sentiment140 dataset containing 1.6 million tweets, we've built a complete machine learning pipeline with preprocessing, training, and deployment capabilities including a modern web interface for real-time sentiment analysis.

### Key Features

- ✅ **Interactive Web Interface**: Modern web app with real-time sentiment analysis
- ✅ **Enhanced Training Pipeline**: Improved model training with proper preprocessing
- ✅ **Large-scale Dataset**: Utilizes Sentiment140 dataset with 1.6M tweets
- ✅ **Advanced Text Preprocessing**: Implements stemming, stopword removal, and text normalization
- ✅ **TF-IDF Vectorization**: Efficient feature extraction for text data
- ✅ **Machine Learning Model**: Logistic Regression with optimized parameters
- ✅ **Model Persistence**: Proper saving/loading of models and vectorizers
- ✅ **FastAPI Backend**: Modern API framework for web services
- ✅ **Comprehensive Testing**: Thorough model validation and accuracy testing

## 👥 Team Credits

This project was developed by **Team CYAN**:

- **[Sarthak Singh](https://github.com/sarthaksingh02-sudo)** - Project Lead & ML Engineer
- **Himanshu Majumdar** - Data Scientist & NLP Specialist  
- **Samit Singh Bag** - ML Engineer & Data Analyst
- **Disha Rawat** - Software Developer & Visualization Expert

## 📁 Project Structure

```
Twitter-Sentiment-Analysis/
├── web/                           # Web application
│   ├── app.py                     # FastAPI backend server
│   ├── templates/                 # HTML templates
│   │   ├── index.html             # Main web interface
│   │   └── index_standalone.html  # Standalone version
│   └── static/                    # Static assets
│       ├── css/style.css          # Styling
│       └── js/app.js              # Frontend JavaScript
├── src/
│   └── sentiment_analyzer.py     # Core sentiment analysis module
├── models/
│   ├── twitter_sentiment_model.pkl      # Trained logistic regression model
│   └── twitter_sentiment_vectorizer.pkl # TF-IDF vectorizer
├── notebooks/
│   └── twitter_sentiment_analysis.py    # Original training script
├── scripts/
│   ├── download_data.py           # Dataset download utility
│   └── train_model.py             # Model training pipeline
├── data/
│   └── test_sample.csv            # Sample test data
├── tests/                         # Test directory
├── docs/                          # Documentation
├── enhanced_twitter_sentiment_analysis.py  # Enhanced training script
├── launch_web.py                  # Quick web app launcher
├── requirements.txt               # Core dependencies
├── requirements-web.txt           # Web app dependencies
├── setup.py                       # Package setup
├── LICENSE                        # MIT License
└── README.md                      # Project documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- pip package manager
- Kaggle account (for dataset access, optional for web app)

### Option 1: Web Application (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sarthaksingh02-sudo/Twitter-Sentiment-Analysis.git
   cd Twitter-Sentiment-Analysis
   ```

2. **Install web dependencies:**
   ```bash
   pip install -r requirements-web.txt
   ```

3. **Launch the web application:**
   ```bash
   python launch_web.py
   ```
   
4. **Open your browser:**
   - Navigate to `http://localhost:8000`
   - Start analyzing sentiment in real-time!

### Option 2: Command Line Training

1. **Install full dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Kaggle API (for new training):**
   - Download your `kaggle.json` from Kaggle account settings
   - Place it in the project root directory
   ```bash
   python scripts/download_data.py
   ```

3. **Train a new model:**
   ```bash
   python enhanced_twitter_sentiment_analysis.py
   ```

### Quick Usage

```python
from src.sentiment_analyzer import SentimentAnalyzer

# Initialize the analyzer
analyzer = SentimentAnalyzer()

# Load pre-trained model
analyzer.load_model('models/trained_model.sav')

# Analyze sentiment
tweet = "I love this new product!"
sentiment = analyzer.predict(tweet)
print(f"Sentiment: {'Positive' if sentiment == 1 else 'Negative'}")
```

## 📊 Dataset Information

**Sentiment140 Dataset**
- **Size**: 1.6 million tweets
- **Classes**: Binary (0 = Negative, 1 = Positive)
- **Features**: 
  - `target`: Sentiment label (0/1)
  - `id`: Tweet ID
  - `date`: Tweet timestamp
  - `flag`: Query flag
  - `user`: Username
  - `text`: Tweet content
- **Source**: [Kaggle Sentiment140](https://www.kaggle.com/kazanova/sentiment140)

## 🔧 Methodology

### 1. Data Preprocessing
- **Text Cleaning**: Remove special characters, URLs, mentions
- **Normalization**: Convert to lowercase
- **Tokenization**: Split text into individual words
- **Stopword Removal**: Remove common English stopwords
- **Stemming**: Reduce words to their root form using Porter Stemmer

### 2. Feature Engineering
- **TF-IDF Vectorization**: Convert text to numerical features
- **Feature Selection**: Optimal feature dimensionality

### 3. Model Training
- **Algorithm**: Logistic Regression
- **Training Split**: 80% training, 20% testing
- **Stratification**: Maintain class balance in splits
- **Optimization**: Maximum iterations = 1000

### 4. Model Evaluation
- **Metrics**: Accuracy Score
- **Validation**: Train/Test performance comparison
- **Visualization**: Sentiment distribution analysis

## 📈 Results

Our model achieves strong performance on the Sentiment140 dataset:

- **Training Accuracy**: ~77-80%
- **Test Accuracy**: ~77-80%
- **Model Generalization**: Good performance on unseen data
- **Processing Speed**: Efficient inference for real-time applications

### Sample Results Visualization

The model provides insights into sentiment distribution:
- Positive sentiment percentage
- Negative sentiment percentage  
- Visual bar charts for easy interpretation

## 🛠️ Technical Stack

- **Programming Language**: Python 3.7+
- **ML Libraries**: scikit-learn, numpy, pandas
- **NLP Libraries**: NLTK, re (regex)
- **Visualization**: matplotlib
- **Data Handling**: pandas, zipfile
- **Model Persistence**: pickle
- **Development**: Jupyter Notebooks, Google Colab

## 📝 Usage Examples

### Basic Sentiment Analysis
```python
import pickle
from src.sentiment_analyzer import SentimentAnalyzer

# Load trained model
model = pickle.load(open('models/trained_model.sav', 'rb'))

# Analyze custom text
def analyze_sentiment(text):
    # Preprocess text (implement stemming function)
    processed_text = preprocess_text(text)
    
    # Vectorize
    vectorized_text = vectorizer.transform([processed_text])
    
    # Predict
    prediction = model.predict(vectorized_text)
    
    return "Positive" if prediction[0] == 1 else "Negative"

# Example usage
print(analyze_sentiment("I love machine learning!"))  # Positive
print(analyze_sentiment("This is terrible"))          # Negative
```

### Batch Processing
```python
tweets = [
    "Amazing product, highly recommend!",
    "Worst experience ever",
    "Pretty good, could be better",
    "Absolutely fantastic!"
]

for tweet in tweets:
    sentiment = analyze_sentiment(tweet)
    print(f"'{tweet}' → {sentiment}")
```

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
```

## 🚀 Deployment Options

### Local Flask API
```python
from flask import Flask, request, jsonify
from src.sentiment_analyzer import SentimentAnalyzer

app = Flask(__name__)
analyzer = SentimentAnalyzer()

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    sentiment = analyzer.predict(text)
    return jsonify({'sentiment': 'positive' if sentiment == 1 else 'negative'})

if __name__ == '__main__':
    app.run(debug=True)
```

### Docker Deployment
```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

## 📚 Documentation

- [Methodology Documentation](docs/methodology.md)
- [API Reference](docs/api_reference.md)
- [Contributing Guidelines](CONTRIBUTING.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Sentiment140 Dataset**: Thanks to Stanford University for providing the dataset
- **NLTK**: Natural Language Toolkit for text processing utilities
- **scikit-learn**: Machine learning library for model implementation
- **Kaggle**: Platform for dataset hosting and competition
- **Google Colab**: Development environment for initial prototyping

## 📞 Contact

- **Project Maintainer**: DISHA RAWAT
- **Team Email**: [disharawat643@gmail.com]

## 🔮 Future Improvements

- [ ] **Deep Learning Models**: Implement LSTM/GRU networks
- [ ] **Real-time Processing**: Stream processing for live Twitter data
- [ ] **Multi-class Classification**: Extend to emotion detection (joy, anger, fear, etc.)
- [ ] **Web Interface**: Interactive dashboard for sentiment analysis
- [ ] **API Development**: RESTful API for integration with other applications
- [ ] **Performance Optimization**: GPU acceleration and model compression
- [ ] **Advanced NLP**: Implement BERT/transformer models
- [ ] **Multilingual Support**: Support for multiple languages

---

**Made with ❤️ by Team Cyan**

⭐ **If you found this project helpful, please give it a star!** ⭐
