# Spam Email Detection Using NLP and Logistic Regression

## Project Overview
This project implements an end-to-end spam email detection system using Natural Language Processing (NLP) and Machine Learning. The model classifies emails as spam or ham (non-spam) by learning patterns from real-world email data.

The system processes raw email files, cleans and transforms text data, extracts meaningful features using TF-IDF, trains a logistic regression classifier, and evaluates performance using standard classification metrics.

---

## Dataset
The project uses the **SpamAssassin public email corpus**, which contains real spam and non-spam (ham) emails.

Data sources:
- Easy Ham emails
- Spam emails

Each email is labeled as:
- 0 → Ham (Non-spam)
- 1 → Spam

---

## Tools and Technologies
- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- BeautifulSoup
- Matplotlib
- Seaborn

---

## Machine Learning & NLP Workflow

### 1. Data Collection
- Download and extract SpamAssassin dataset.
- Parse raw email files.
- Extract text from plain text and HTML emails.

### 2. Text Preprocessing
- Convert text to lowercase.
- Remove URLs, email addresses, and punctuation.
- Remove stopwords using NLTK.
- Clean and normalize email content.

### 3. Feature Extraction
- Convert cleaned text into numerical features using TF-IDF Vectorizer.
- Limit vocabulary size for efficiency.

### 4. Model Training
- Split dataset into training and testing sets.
- Train Logistic Regression classifier on TF-IDF features.

### 5. Model Evaluation
- Evaluate model using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Visualize results using a confusion matrix.

### 6. Real-Time Testing
- Test model on new unseen email examples.
- Predict spam/ham labels with probability scores.

---

## Results
- Successfully classified spam and ham emails.
- Achieved strong accuracy on test data.
- Model generalizes well to unseen email samples.

---

## How to Run
1. Clone the repository.

2. Install dependencies:
pip install numpy pandas scikit-learn nltk beautifulsoup4 matplotlib seaborn

3. Run the script:
python spam_detection.py

NLTK stopwords will be downloaded automatically.

---

## Future Improvements
- Use advanced NLP models (Naive Bayes, SVM).
- Apply deep learning (LSTM, Transformers).
- Improve text normalization.
- Deploy as a web or API service.

---

## Author
Manya Sondhi  
AI & ML Engineering Student  
