import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the SVM model and TF-IDF vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Define function to preprocess input text
def preprocess_text(text):
    
    # Tokenize the text
    tokens = word_tokenize(text)

    # Lowercase all words
    tokens = [word.lower() for word in tokens]

    # Remove punctuation and special characters
    table = str.maketrans('', '', string.punctuation)
    stripped = [word.translate(table) for word in tokens]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in stripped if word not in stop_words]

    return ' '.join(words)

# Define function to make predictions
def predict(text):
    # Preprocess the input text
    preprocessed_text = preprocess_text(text)
    
    # Convert preprocessed text to TF-IDF vector
    tfidf_vector = vectorizer.transform([preprocessed_text])
    vector = tfidf_vector.toarray()
    
    # Make prediction using the loaded SVM model
    prediction = model.predict(vector)
    
    return prediction

# Streamlit app code
st.title('Spam Detection App')
text_input = st.text_input('Enter a message:')

if st.button('Predict'):
    prediction = predict(text_input)
    if prediction==0:
        st.write('Prediction:', "Not spam")
    else:
        st.write('Prediction:', "Spam")
    
