/**
 * Twitter Sentiment Analysis - JavaScript Application
 * ===================================================
 * 
 * Interactive web interface for sentiment analysis.
 * 
 * Authors: Team 3 Dude's
 * - Sarthak Singh (Project Lead & ML Engineer)
 * - Himanshu Majumdar (Data Scientist & NLP Specialist)
 * - Samit Singh Bag (ML Engineer & Data Analyst)
 * - Sahil Raghav (Software Developer & Visualization Expert)
 * 
 * License: MIT
 */

class SentimentAnalyzer {
    constructor() {
        this.initializeElements();
        this.attachEventListeners();
        this.stats = {
            totalAnalyzed: 0,
            totalConfidence: 0
        };
    }

    initializeElements() {
        // Single analysis elements
        this.singleTextInput = document.getElementById('singleTextInput');
        this.analyzeSingleBtn = document.getElementById('analyzeSingleBtn');
        this.singleResult = document.getElementById('singleResult');
        this.charCount = document.getElementById('charCount');

        // Batch analysis elements
        this.batchTextInput = document.getElementById('batchTextInput');
        this.analyzeBatchBtn = document.getElementById('analyzeBatchBtn');
        this.batchResult = document.getElementById('batchResult');
        this.lineCount = document.getElementById('lineCount');

        // UI elements
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.toastContainer = document.getElementById('toastContainer');
        
        // Stats elements
        this.totalAnalyzed = document.getElementById('totalAnalyzed');
        this.avgConfidence = document.getElementById('avgConfidence');
    }

    attachEventListeners() {
        // Single text analysis
        this.analyzeSingleBtn.addEventListener('click', () => this.analyzeSingle());
        this.singleTextInput.addEventListener('input', () => this.updateCharCount());
        this.singleTextInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                this.analyzeSingle();
            }
        });

        // Batch analysis
        this.analyzeBatchBtn.addEventListener('click', () => this.analyzeBatch());
        this.batchTextInput.addEventListener('input', () => this.updateLineCount());

        // Initialize counters
        this.updateCharCount();
        this.updateLineCount();
    }

    updateCharCount() {
        const text = this.singleTextInput.value;
        const count = text.length;
        this.charCount.textContent = count;
        
        // Visual feedback for character limit
        if (count > 250) {
            this.charCount.style.color = '#f59e0b';
        } else {
            this.charCount.style.color = '#657786';
        }
    }

    updateLineCount() {
        const text = this.batchTextInput.value;
        const lines = text.split('\n').filter(line => line.trim().length > 0);
        this.lineCount.textContent = `${lines.length} lines`;
        
        // Visual feedback for line limit
        if (lines.length > 100) {
            this.lineCount.style.color = '#f59e0b';
            this.analyzeBatchBtn.disabled = true;
        } else {
            this.lineCount.style.color = '#657786';
            this.analyzeBatchBtn.disabled = false;
        }
    }

    showLoading() {
        this.loadingOverlay.style.display = 'flex';
        document.body.style.overflow = 'hidden';
    }

    hideLoading() {
        this.loadingOverlay.style.display = 'none';
        document.body.style.overflow = 'auto';
    }

    showToast(message, type = 'info', duration = 3000) {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <div class="toast-content">
                <i class="fas ${this.getToastIcon(type)}"></i>
                <span>${message}</span>
            </div>
        `;

        this.toastContainer.appendChild(toast);

        // Auto remove toast
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, duration);
    }

    getToastIcon(type) {
        const icons = {
            success: 'fa-check-circle',
            error: 'fa-exclamation-circle',
            warning: 'fa-exclamation-triangle',
            info: 'fa-info-circle'
        };
        return icons[type] || icons.info;
    }

    async analyzeSingle() {
        const text = this.singleTextInput.value.trim();
        
        if (!text) {
            this.showToast('Please enter some text to analyze', 'warning');
            return;
        }

        if (text.length > 280) {
            this.showToast('Text is too long. Maximum 280 characters allowed.', 'warning');
            return;
        }

        this.showLoading();
        this.analyzeSingleBtn.disabled = true;

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            this.displaySingleResult(result);
            this.updateStats(1, result.confidence);
            this.showToast('Analysis completed successfully!', 'success');

        } catch (error) {
            console.error('Analysis error:', error);
            this.showToast(`Analysis failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
            this.analyzeSingleBtn.disabled = false;
        }
    }

    async analyzeBatch() {
        const text = this.batchTextInput.value.trim();
        
        if (!text) {
            this.showToast('Please enter some texts to analyze', 'warning');
            return;
        }

        const texts = text.split('\n').filter(line => line.trim().length > 0);
        
        if (texts.length === 0) {
            this.showToast('No valid texts found', 'warning');
            return;
        }

        if (texts.length > 100) {
            this.showToast('Too many texts. Maximum 100 allowed per batch.', 'warning');
            return;
        }

        this.showLoading();
        this.analyzeBatchBtn.disabled = true;

        try {
            const response = await fetch('/api/analyze-batch', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ texts: texts })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            this.displayBatchResults(result);
            this.updateStats(texts.length, result.summary.avg_confidence);
            this.showToast(`Successfully analyzed ${texts.length} texts!`, 'success');

        } catch (error) {
            console.error('Batch analysis error:', error);
            this.showToast(`Batch analysis failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
            this.analyzeBatchBtn.disabled = false;
        }
    }

    displaySingleResult(result) {
        // Show result section
        this.singleResult.style.display = 'block';
        
        // Animate scroll to result
        this.singleResult.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'nearest' 
        });

        // Update sentiment display
        document.getElementById('sentimentEmoji').textContent = result.emoji;
        
        const sentimentLabel = document.getElementById('sentimentLabel');
        sentimentLabel.textContent = result.sentiment;
        sentimentLabel.className = `sentiment-label ${result.sentiment.toLowerCase()}`;
        
        // Update confidence bar
        const confidenceBar = document.getElementById('confidenceBar');
        const confidencePercent = (result.confidence * 100);
        
        // Animate confidence bar
        setTimeout(() => {
            confidenceBar.style.width = `${confidencePercent}%`;
        }, 100);
        
        document.getElementById('confidenceText').textContent = 
            `${confidencePercent.toFixed(1)}% confidence`;

        // Update result details
        document.getElementById('originalText').textContent = result.text;
        document.getElementById('processedText').textContent = result.processed_text;
        document.getElementById('timestamp').textContent = 
            new Date(result.timestamp).toLocaleString();
    }

    displayBatchResults(result) {
        // Show result section
        this.batchResult.style.display = 'block';
        
        // Animate scroll to result
        this.batchResult.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'nearest' 
        });

        // Update summary stats
        document.getElementById('positiveCount').textContent = result.summary.positive;
        document.getElementById('negativeCount').textContent = result.summary.negative;

        // Create individual result items
        const batchResults = document.getElementById('batchResults');
        batchResults.innerHTML = '';

        result.results.forEach((item, index) => {
            const resultItem = document.createElement('div');
            resultItem.className = 'batch-item';
            
            const truncatedText = item.text.length > 60 
                ? item.text.substring(0, 60) + '...' 
                : item.text;
            
            resultItem.innerHTML = `
                <div class="batch-item-text">${truncatedText}</div>
                <div class="batch-item-result">
                    <span>${item.emoji}</span>
                    <span class="${item.sentiment.toLowerCase()}">${item.sentiment}</span>
                    <span class="confidence">${(item.confidence * 100).toFixed(1)}%</span>
                </div>
            `;

            // Add click handler to show full text
            resultItem.addEventListener('click', () => {
                this.showFullResult(item);
            });

            batchResults.appendChild(resultItem);
        });
    }

    showFullResult(item) {
        const modal = document.createElement('div');
        modal.className = 'result-modal';
        modal.innerHTML = `
            <div class="modal-overlay">
                <div class="modal-content">
                    <div class="modal-header">
                        <h3>Detailed Analysis</h3>
                        <button class="modal-close">&times;</button>
                    </div>
                    <div class="modal-body">
                        <div class="sentiment-display">
                            <span class="sentiment-emoji">${item.emoji}</span>
                            <div class="sentiment-info">
                                <h4 class="sentiment-label ${item.sentiment.toLowerCase()}">${item.sentiment}</h4>
                                <div class="confidence-display">
                                    <div class="confidence-bar">
                                        <div class="confidence-fill" style="width: ${item.confidence * 100}%"></div>
                                    </div>
                                    <span class="confidence-text">${(item.confidence * 100).toFixed(1)}% confidence</span>
                                </div>
                            </div>
                        </div>
                        <div class="text-analysis">
                            <div class="text-item">
                                <label>Original Text:</label>
                                <p>${item.text}</p>
                            </div>
                            <div class="text-item">
                                <label>Processed Text:</label>
                                <p>${item.processed_text}</p>
                            </div>
                            <div class="text-item">
                                <label>Analysis Time:</label>
                                <p>${new Date(item.timestamp).toLocaleString()}</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);
        document.body.style.overflow = 'hidden';

        // Close modal handlers
        const closeModal = () => {
            document.body.removeChild(modal);
            document.body.style.overflow = 'auto';
        };

        modal.querySelector('.modal-close').addEventListener('click', closeModal);
        modal.querySelector('.modal-overlay').addEventListener('click', (e) => {
            if (e.target === modal.querySelector('.modal-overlay')) {
                closeModal();
            }
        });

        // Escape key handler
        const escapeHandler = (e) => {
            if (e.key === 'Escape') {
                closeModal();
                document.removeEventListener('keydown', escapeHandler);
            }
        };
        document.addEventListener('keydown', escapeHandler);
    }

    updateStats(count, confidence) {
        this.stats.totalAnalyzed += count;
        this.stats.totalConfidence = (this.stats.totalConfidence + confidence) / 2; // Simple average

        // Animate counter updates
        this.animateNumber(this.totalAnalyzed, this.stats.totalAnalyzed);
        this.animateNumber(this.avgConfidence, this.stats.totalConfidence * 100, '%');
    }

    animateNumber(element, targetValue, suffix = '') {
        const startValue = parseInt(element.textContent) || 0;
        const increment = (targetValue - startValue) / 30;
        let currentValue = startValue;
        
        const animate = () => {
            currentValue += increment;
            if (
                (increment > 0 && currentValue >= targetValue) ||
                (increment < 0 && currentValue <= targetValue)
            ) {
                currentValue = targetValue;
            }
            
            element.textContent = Math.round(currentValue) + suffix;
            
            if (currentValue !== targetValue) {
                requestAnimationFrame(animate);
            }
        };
        
        requestAnimationFrame(animate);
    }

    // Initialize sample texts for demo
    addSampleTexts() {
        const sampleTexts = [
            "I absolutely love this new product! Best purchase ever!",
            "This service is terrible. Worst experience of my life.",
            "The weather is nice today, perfect for a walk in the park.",
            "Can't believe how amazing this restaurant is! Highly recommended!",
            "Stuck in traffic again... this commute is getting frustrating."
        ];

        // Add sample text buttons
        const singleCard = document.querySelector('.analysis-card');
        const sampleContainer = document.createElement('div');
        sampleContainer.className = 'sample-texts';
        sampleContainer.innerHTML = `
            <p><strong>Try these examples:</strong></p>
            <div class="sample-buttons">
                ${sampleTexts.map((text, index) => 
                    `<button class="sample-btn" data-text="${text}">Sample ${index + 1}</button>`
                ).join('')}
            </div>
        `;

        // Insert before input section
        const inputSection = singleCard.querySelector('.input-section');
        inputSection.parentNode.insertBefore(sampleContainer, inputSection);

        // Add event listeners for sample buttons
        sampleContainer.addEventListener('click', (e) => {
            if (e.target.classList.contains('sample-btn')) {
                this.singleTextInput.value = e.target.dataset.text;
                this.updateCharCount();
            }
        });
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const app = new SentimentAnalyzer();
    app.addSampleTexts();
    
    console.log('🚀 Twitter Sentiment Analysis Web App Initialized');
    console.log('👥 Developed by Team 3 Dude\'s');
});

// Add modal styles dynamically
const modalStyles = `
.result-modal {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 2000;
}

.modal-overlay {
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.8);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
}

.modal-content {
    background: white;
    border-radius: 12px;
    max-width: 600px;
    width: 100%;
    max-height: 90vh;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    animation: slideInUp 0.3s ease-out;
}

.modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px;
    border-bottom: 2px solid #e2e8f0;
}

.modal-header h3 {
    margin: 0;
    color: #14171A;
}

.modal-close {
    background: none;
    border: none;
    font-size: 24px;
    cursor: pointer;
    color: #657786;
    width: 30px;
    height: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    transition: all 0.2s ease;
}

.modal-close:hover {
    background: #f7f9fa;
    color: #14171A;
}

.modal-body {
    padding: 20px;
    max-height: calc(90vh - 80px);
    overflow-y: auto;
}

.sentiment-display {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 24px;
    padding: 16px;
    background: linear-gradient(135deg, #f8fafc, #ffffff);
    border-radius: 12px;
    border: 2px solid #e2e8f0;
}

.sentiment-emoji {
    font-size: 48px;
}

.sentiment-info {
    flex: 1;
}

.sentiment-info h4 {
    margin: 0 0 8px 0;
    font-size: 20px;
}

.sentiment-info h4.positive {
    color: #10b981;
}

.sentiment-info h4.negative {
    color: #f59e0b;
}

.confidence-display {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.confidence-bar {
    width: 200px;
    height: 8px;
    background: #e2e8f0;
    border-radius: 4px;
    overflow: hidden;
}

.confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, #1DA1F2, #1D9BF0);
    border-radius: 4px;
    transition: width 1s ease-out;
}

.confidence-text {
    font-size: 14px;
    color: #657786;
    font-weight: 500;
}

.text-analysis {
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.text-item {
    padding: 16px;
    background: #f7f9fa;
    border-radius: 8px;
}

.text-item label {
    display: block;
    font-weight: 600;
    color: #657786;
    margin-bottom: 8px;
    font-size: 14px;
}

.text-item p {
    margin: 0;
    color: #14171A;
    line-height: 1.5;
    word-break: break-word;
}

.sample-texts {
    margin-bottom: 24px;
    padding: 16px;
    background: #f7f9fa;
    border-radius: 8px;
}

.sample-texts p {
    margin: 0 0 12px 0;
    color: #657786;
    font-size: 14px;
}

.sample-buttons {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

.sample-btn {
    background: #1DA1F2;
    color: white;
    border: none;
    padding: 6px 12px;
    border-radius: 16px;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.2s ease;
}

.sample-btn:hover {
    background: #0d8bd9;
    transform: translateY(-1px);
}

@media (max-width: 768px) {
    .modal-overlay {
        padding: 10px;
    }
    
    .sentiment-display {
        flex-direction: column;
        text-align: center;
    }
    
    .confidence-bar {
        width: 100%;
    }
}
`;

// Add styles to head
const styleSheet = document.createElement('style');
styleSheet.textContent = modalStyles;
document.head.appendChild(styleSheet);
