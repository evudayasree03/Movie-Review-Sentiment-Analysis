import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os

nltk.download('stopwords', download_dir='nltk_data')
nltk.download('punkt', download_dir='nltk_data')
nltk.download('wordnet', download_dir='nltk_data')
nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))


st.header("🎬 Movie Review Sentiement Analysis")

text = st.text_input("Enter the movie review: ")

#Preprocessing the text review
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()                 #Converting the review in lowercase
    text = re.sub(r'<.*?>', '', text)   #Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  #Remove special Characters
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df = pd.read_csv('IMDB Dataset.csv')
df = df.sample(500, random_state=42)

df['cleaned_review'] = df['review'].apply(preprocess_text)  #We are using apply fn, it will take the review and passed to preprocess text fn 
print(df[['review', 'cleaned_review']].head(1).values[0])

vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['sentiment'].apply(lambda x :1 if x =='positive' else 0)
print("TF-IDF Matrix Shape:", X.shape)
print("Sample TD-IDF Row: ", X[0][:10])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print ("Training Set Size:", X_train.shape)
# print ("Testing Set Size:", X_test.shape)

model = LogisticRegression()
model.fit(X_train, y_train)


#preprocess the input
cleaned = preprocess_text(text)
#convert to TD-IDF
vectorized = vectorizer.transform([cleaned])
prediction = model.predict(vectorized)[0]
sentiment = "Positive ☺️" if prediction == 1 else "Negative 😔"

if st.button("Predict Sentiment:"):
    st.success(f"Predicted Sentiment: {sentiment}")