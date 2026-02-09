SMS Spam Detection (Machine Learning + Flask API)

This project is a machine learning–based SMS spam detection system built using TF-IDF text vectorization and Logistic Regression, deployed as a REST API using Flask.

Multiple machine learning models were evaluated in metrics.ipynb to determine the best-performing approach. Logistic Regression was selected due to its high recall for spam messages, strong overall performance, and computational efficiency.

Features
- Text preprocessing (cleaning & normalization)
- TF-IDF vectorization
- Logistic Regression with class balancing
- API for predictions
- JSON-based communication
- Model evaluation and comparison

Model
- Algorithm: Logistic Regression
- Vectorization: TF-IDF
- Class imbalance handled when training model

Performance
- Accuracy: ~97%
- Recall (Spam): ~85%
- F1-score: ~97%

Project Structure
SMS SPAM
- backend/
    - spam.py          # Model training script
    - spam_app.py      # Flask API
    - test.py          # API test script

- dataset/
    - spam.csv 

- model/
    - spam_detector.pkl
    - spam_vectorizer.pkl

- metrics.ipynb        # Model evaluation & comparison
- requirements.txt
- README.md

How To Run Locally
1. Install Dependencies
pip install -r requirement.txt

2. Train Model
   backend
   -spam.py

3. Run Flask App
   -spam_app.py

4. Test App:
   Api runs on http://127.0.0.1:5000
   Run test.py


SMS Spam Collection Dataset
Source: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Author
Awwal Ajao