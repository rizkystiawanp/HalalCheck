import nltk
import pickle
import streamlit as st
import re
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM
from nltk.tokenize import RegexpTokenizer

st.set_page_config(layout='wide',
                   page_title='Sentiment Analysis Application',
                   )
def run():
    # gambar dengan st.image
    st.image('https://d2vuyvo9qdtgo9.cloudfront.net/assets/img/bg/img-misi.jpg')

    st.markdown('''
    ## Background of the Problem
    In the fast-food industry, understanding customer satisfaction is a major challenge for large restaurants like McDonald's.
    Customer reviews can provide valuable insights into their dining experience, including food quality, service, restaurant cleanliness, and price.
    However, with thousands of reviews posted across multiple platforms, manually analyzing customer sentiment is difficult and inefficient.
    Without a clear understanding of customer opinions, McDonald's faces several challenges, such as:

    - Inability to quickly identify customer satisfaction trends.
    - Difficulty in objectively assessing service quality across multiple branches.
    - Lack of transparency in systematically responding to customer complaints.
    - Potential loss of customers due to service or food quality that does not meet expectations.
                    ''')


