#!/usr/bin/env python3
"""
Twitter Sentiment Analysis - Web Application
==============================================

A beautiful web interface for the Twitter Sentiment Analysis system.

Authors: Team 3 Dude's
- Sarthak Singh (Project Lead & ML Engineer)
- Himanshu Majumdar (Data Scientist & NLP Specialist)
- Samit Singh Bag (ML Engineer & Data Analyst)
- Sahil Raghav (Software Developer & Visualization Expert)

License: MIT
"""

import os
import sys
import json
import time
from typing import List, Dict, Any
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from sentiment_analyzer import SentimentAnalyzer

# Initialize FastAPI app
app = FastAPI(
    title="Twitter Sentiment Analysis",
    description="Beautiful web interface for sentiment analysis powered by Team 3 Dude's",
    version="1.0.0"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global analyzer instance
analyzer = None

# Pydantic models for API requests
class SentimentRequest(BaseModel):
    text: str

class BatchSentimentRequest(BaseModel):
    texts: List[str]

class AnalysisResult(BaseModel):
    text: str
    sentiment: str
    confidence: float
    processed_text: str
    emoji: str
    timestamp: str

# Initialize the sentiment analyzer
def initialize_analyzer():
    """Initialize the sentiment analyzer with the trained model."""
    global analyzer
    try:
        # Try to load the enhanced model first (our properly trained model)
        enhanced_model_path = os.path.join('..', 'models', 'twitter_sentiment_model.pkl')
        enhanced_vectorizer_path = os.path.join('..', 'models', 'twitter_sentiment_vectorizer.pkl')
        
        if os.path.exists(enhanced_model_path) and os.path.exists(enhanced_vectorizer_path):
            # Import the enhanced analyzer
            sys.path.append('..')
            from enhanced_twitter_sentiment_analysis import EnhancedTwitterSentimentAnalyzer
            
            analyzer = EnhancedTwitterSentimentAnalyzer()
            analyzer.load_complete_model(enhanced_model_path, enhanced_vectorizer_path)
            print(f"✅ Enhanced model loaded successfully!")
            print(f"   - Model: {enhanced_model_path}")
            print(f"   - Vectorizer: {enhanced_vectorizer_path}")
            return True
        else:
            # Fallback to old model (with bias issues)
            print("⚠️  Enhanced model not found. Using fallback model...")
            analyzer = SentimentAnalyzer()
            model_path = os.path.join('..', 'models', 'demo_model.pkl')
            if not os.path.exists(model_path):
                model_path = os.path.join('..', 'models', 'test_model.pkl')
            
            if os.path.exists(model_path):
                analyzer.load_model(model_path)
                print(f"⚠️  Fallback model loaded from {model_path} (may have bias issues)")
                return True
            else:
                print("❌ No trained model found. Please train a model first.")
                return False
                
    except Exception as e:
        print(f"❌ Error initializing analyzer: {str(e)}")
        return False

# Initialize analyzer on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    print("🚀 Starting Twitter Sentiment Analysis Web App...")
    print("👥 Developed by Team 3 Dude's")
    success = initialize_analyzer()
    if not success:
        print("⚠️  Warning: Running without trained model. Please train a model for full functionality.")

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse("index_standalone.html", {"request": request})

@app.get("/original", response_class=HTMLResponse)
async def original_page(request: Request):
    """Serve the original web interface with external files."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/test", response_class=HTMLResponse)
async def test_page(request: Request):
    """Serve test page for debugging static files."""
    with open("test.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "analyzer_loaded": analyzer is not None,
        "timestamp": datetime.now().isoformat(),
        "team": "Team 3 Dude's"
    }

@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze_sentiment(request: SentimentRequest):
    """Analyze sentiment for a single text."""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Sentiment analyzer not initialized")
    
    try:
        # Get prediction
        sentiment_score = analyzer.predict(request.text)
        sentiment_label = analyzer.get_sentiment_label(sentiment_score)
        probabilities = analyzer.predict_proba(request.text)
        confidence = float(max(probabilities))
        processed_text = analyzer.preprocess_text(request.text)
        
        # Add emoji based on sentiment
        emoji = "😊" if sentiment_score == 1 else "😞"
        
        return AnalysisResult(
            text=request.text,
            sentiment=sentiment_label,
            confidence=confidence,
            processed_text=processed_text,
            emoji=emoji,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.post("/api/analyze-batch")
async def analyze_batch_sentiment(request: BatchSentimentRequest):
    """Analyze sentiment for multiple texts."""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Sentiment analyzer not initialized")
    
    if len(request.texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 texts allowed per batch")
    
    try:
        results = []
        
        # Batch predictions
        sentiment_scores = analyzer.predict(request.texts)
        probabilities = analyzer.predict_proba(request.texts)
        
        for i, text in enumerate(request.texts):
            sentiment_score = sentiment_scores[i]
            sentiment_label = analyzer.get_sentiment_label(sentiment_score)
            confidence = float(max(probabilities[i]))
            processed_text = analyzer.preprocess_text(text)
            emoji = "😊" if sentiment_score == 1 else "😞"
            
            results.append(AnalysisResult(
                text=text,
                sentiment=sentiment_label,
                confidence=confidence,
                processed_text=processed_text,
                emoji=emoji,
                timestamp=datetime.now().isoformat()
            ))
        
        return {
            "results": results,
            "summary": {
                "total": len(results),
                "positive": sum(1 for r in results if r.sentiment == "Positive"),
                "negative": sum(1 for r in results if r.sentiment == "Negative"),
                "avg_confidence": sum(r.confidence for r in results) / len(results)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis error: {str(e)}")

@app.get("/api/stats")
async def get_stats():
    """Get application statistics."""
    return {
        "model_info": {
            "status": "loaded" if analyzer else "not_loaded",
            "model_type": "Logistic Regression with TF-IDF",
            "preprocessing": "Stemming + Stopword Removal"
        },
        "team": {
            "name": "Team 3 Dude's",
            "members": [
                "Sarthak Singh - Project Lead & ML Engineer",
                "Himanshu Majumdar - Data Scientist & NLP Specialist",
                "Samit Singh Bag - ML Engineer & Data Analyst",
                "Sahil Raghav - Software Developer & Visualization Expert"
            ]
        },
        "features": [
            "Real-time sentiment analysis",
            "Batch text processing",
            "Advanced text preprocessing",
            "Confidence scoring",
            "Beautiful web interface"
        ]
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return templates.TemplateResponse("404.html", {"request": request}, status_code=404)

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )

# Main function to run the app
def main():
    """Run the web application."""
    print("🌟 Twitter Sentiment Analysis Web App")
    print("👥 By Team 3 Dude's")
    print("🚀 Starting server...")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
