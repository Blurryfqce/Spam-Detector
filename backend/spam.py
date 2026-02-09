#Importing necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import joblib

#Function to remove punctuation, numbers, and convert to lowercase
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    return text

#Loading dataset
cols = ['label', 'text']
df = pd.read_csv(r'dataset\spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = cols
df['label']= df['label'].map({'ham': 0, 'spam': 1})
df['text'] = df['text'].apply(clean_text)

#Vectorizing text data  
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])

x = X
y = df['label']

#Training logistic regression model
model = LogisticRegression( class_weight='balanced')
model = model.fit(x, y)

#Saving the vectorizer and model
joblib.dump(vectorizer, r"model\spam_vectorizer.pkl")
joblib.dump(model, r"model\spam_detector.pkl")
print("Model and vectorizer trained and saved successfully.")