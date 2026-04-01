# importing libraries 
import numpy as np
import tensorflow as ts
from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model
import streamlit as st

# Load the IMDB dataset word index 
word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}

# Load the pre trained model 
model = load_model('simple_rnn.h5')

# function to decode review 
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# function to preprocess user input 
def preprocess_text(text): 
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+ 3 for word in words]
    padded_review = pad_sequences([encoded_review],maxlen = 500)
    return padded_review

def predict_sentiment(review): 
    preprossed_input = preprocess_text(review)
    prediction = model.predict(preprossed_input)
    sentiment = 'Postive' if prediction[0][0] > .5 else 'Negative'
    return prediction[0][0],sentiment

#streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.title('Enter a movie review to classify it as positve or negative')

#User input 
user_input = st.text_area('Movie Review')

if st.button('Classify'): 
    prediction,sentiment = predict_sentiment(user_input)

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction}')
else:
    st.write('Please enter a movie review.')
