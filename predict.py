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

def run():
    # Load model dengan custom object scope buat handle parameter LSTM
    with tf.keras.utils.custom_object_scope({'LSTM': LSTM}):
        model = load_model('sentiment_analysis_model.h5')

    # Download resource nltk secara lokal biar nggak error kalau dijalankan di sistem baru
    nltk.data.path.append('./nltk_data')
    nltk.download('stopwords', download_dir='./nltk_data')
    nltk.download('punkt', download_dir='./nltk_data')

    # Load model yang udah dilatih sebelumnya
    model = tf.keras.models.load_model('sentiment_analysis_model.h5')

    # Load tokenizer yang udah dipakai waktu training
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # Load stopwords, tapi tetap biarin kata-kata kasar biar nggak hilang dalam analisis sentimen
    stop_words = set(stopwords.words('english')) - {'fuck', 'shit', 'damn'}

    # Fungsi buat preprocessing teks biar sesuai sama format input model
    def text_preprocessing(text):
        text = text.lower()  # Biar semua teks jadi huruf kecil
        text = re.sub("@[A-Za-z0-9_]+", " ", text)  # Hapus mention (@username)
        text = re.sub("#[A-Za-z0-9_]+", " ", text)  # Hapus hashtag
        text = re.sub(r"http\S+", " ", text)  # Hapus link atau URL
        text = re.sub(r"www.\S+", " ", text)  # Hapus link atau URL
        text = re.sub("[^a-z\s']", " ", text)  # Hapus karakter selain huruf
        text = text.strip()  # Hapus spasi berlebih di awal dan akhir teks

        # Pakai RegexpTokenizer biar tokenisasi lebih rapi tanpa perlu punkt dari nltk
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(text)

        # Hapus stopwords kecuali kata-kata kasar
        tokens = [word for word in tokens if word not in stop_words]

        return ' '.join(tokens)

    # Interface pakai Streamlit
    st.title('Sentiment Analysis Application')

    # Input dari user
    user_input = st.text_area("Enter the text to be analyzed:")

    if st.button('Analysis'):
        if user_input:
            # Preprocessing teks input dari user
            processed_text = text_preprocessing(user_input)
            st.write(f"Text After Preprocessing: {processed_text}")
            
            # Cek apakah ada kata-kata kasar di teks yang diproses
            negative_words = ['fuck', 'shit', 'damn']
            if any(neg_word in processed_text.split() for neg_word in negative_words):
                sentiment = 'Negatif'
            else:
                # Tokenisasi dan padding teks agar sesuai dengan model
                sequence = tokenizer.texts_to_sequences([processed_text])
                padded_sequence = pad_sequences(sequence, maxlen=100, padding='post')
                
                # Prediksi menggunakan model
                prediction = model.predict(padded_sequence)
                predicted_class = prediction.argmax(axis=-1)[0]
                
                # Mapping hasil prediksi ke label sentimen
                label_mapping = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
                sentiment = label_mapping[predicted_class]
            
            # Tampilkan hasil analisis sentimen
            st.write(f"Sentimen: **{sentiment}**")
        else:
            st.write("Please Input Text.")


# # Additional Debug Button to Check if 'fuck' is in Tokenizer
# if st.button('Check if "fuck" is in tokenizer'):
#     word_index = tokenizer.word_index
#     if 'fuck' in word_index:
#         st.write(f"'fuck' is in the tokenizer with index: {word_index['fuck']}")
#     else:
#         st.write("'fuck' is NOT in the tokenizer. It might be treated as OOV (Out of Vocabulary).")