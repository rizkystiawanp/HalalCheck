import os
import json
import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import re
import pandas as pd
import psycopg2
from tensorflow.keras.preprocessing.sequence import pad_sequences
from psycopg2 import sql
from google.cloud import vision
from google.oauth2 import service_account
from PIL import Image
import nltk

# 🔹 Download NLTK Resources
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# ===================== 1️⃣ KONFIGURASI GOOGLE CLOUD & POSTGRESQL ===================== #

# 🔹 Load Google Cloud Credentials
if "GCP_KEY" in st.secrets:
    gcp_credentials = json.loads(st.secrets["GCP_KEY"]["value"])
    credentials = service_account.Credentials.from_service_account_info(gcp_credentials)
    client = vision.ImageAnnotatorClient(credentials=credentials)
else:
    st.error("⚠ GCP_KEY tidak ditemukan! Gunakan `os.environ` jika menjalankan secara lokal.")
    st.stop()

# 🔹 Konfigurasi Koneksi PostgreSQL
DB_NAME = "halal_check"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"

def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn
    except Exception as e:
        st.error(f"❌ Gagal terhubung ke PostgreSQL: {e}")
        return None

# ===================== 2️⃣ LOAD MODEL, TOKENIZER & DATA ===================== #

@st.cache_resource
def load_model_and_tokenizer():
    try:
        model = tf.keras.models.load_model('halal_haram_lstm_finetuned_model.h5')
        with open('tokenizer.pkl', 'rb') as file:
            tokenizer = pickle.load(file)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None, None

model, tokenizer = load_model_and_tokenizer()

if model is None or tokenizer is None:
    st.error("⚠ Model atau Tokenizer gagal dimuat. Periksa kembali file yang diperlukan.")
    st.stop()

@st.cache_data
def load_kode_e_list():
    try:
        df_kode_e = pd.read_csv("kode_e_halal_check.csv")
        if 'Nama Bahan' not in df_kode_e.columns:
            st.error("⚠ CSV tidak memiliki kolom 'Nama Bahan'. Periksa kembali format file!")
            return set()
        return set(df_kode_e['Nama Bahan'].dropna().str.lower().str.strip())
    except Exception as e:
        st.error(f"⚠ Gagal memuat daftar bahan meragukan: {e}")
        return set()

kode_e_list = load_kode_e_list()

# ===================== 3️⃣ PROSES TEKS (NLP) ===================== #

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
irrelevant_words = ["CUSTOMER SER","Mail Box", "Hotline Email", "PO BOXER", "+62-295","customer service", "mailbox", "contact", "website", "email", "barcode",
                    "phone", "nutrition", "produced in", "for more info", "www", "distributor",
                    "net weight", "shelf life", "tel", "fax", "consumer inquiries"]

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    words = [word for word in words if word not in irrelevant_words]
    return ' '.join(words)

def check_suspicious_ingredients(text):
    text = text.lower()
    found_ingredients = []

    # 🔹 Bersihkan teks untuk pencocokan bahan
    cleaned_text = re.sub(r'[^\w\s]', '', text)

    # 🔹 Cari bahan langsung dalam teks OCR
    for bahan in kode_e_list:
        if bahan in cleaned_text:
            found_ingredients.append(bahan)

    return list(set(found_ingredients))  # Hapus duplikasi

# ===================== 4️⃣ PREDIKSI HALAL / HARAM ===================== #

def predict_label(text):
    text_clean = clean_text(text)
    sequence = tokenizer.texts_to_sequences([text_clean])

    if not sequence or len(sequence[0]) == 0:
        st.error("⚠ Tokenisasi gagal! Cek apakah teks kosong atau tidak sesuai format.")
        return "Error", 0, []

    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post')
    prediction = model.predict(padded_sequence)[0][0]
    confidence = round(prediction * 100, 2)

    label = "Halal" if confidence >= 50 else "Haram"
    warning_ingredients = check_suspicious_ingredients(text)
    return label, confidence, warning_ingredients

# ===================== 5️⃣ EXTRACT TEKS DARI GAMBAR (DIPERBAIKI) ===================== #

def extract_text_from_image(image):
    content = image.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if not texts:
        return None

    full_text = texts[0].description

    # 🔹 Regex untuk menangkap bagian Ingredients / Komposisi dengan lebih baik
    pattern = r"(?:Ingredients|Komposisi)[:\s]?(.*)"
    match = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)

    if match:
        extracted_text = match.group(1).strip()

        # 🔹 **Pisahkan teks jika ada kata pemisah seperti 'Contains', 'Allergen Information', dll.**
        stop_words = ["contains", "allergen information", "nutrition facts", "distributed by", "manufactured by"]
        for stop_word in stop_words:
            extracted_text = re.split(r"\b{}\b".format(stop_word), extracted_text, flags=re.IGNORECASE)[0]

        # 🔹 **Ambil hanya 3-6 baris pertama agar tidak berlebihan**
        extracted_lines = extracted_text.split("\n")[:6]
        extracted_text = " ".join(extracted_lines).strip()

        return extracted_text

    return None


# ===================== 6️⃣ SIMPAN KE DATABASE ===================== #

def save_to_database(nama_produk, teks_ocr, prediksi, confidence, bahan_meragukan):
    conn = get_db_connection()
    if conn is None:
        return

    try:
        cursor = conn.cursor()
        insert_query = sql.SQL("""
            INSERT INTO hasil_ocr (nama_produk, teks_ocr, prediksi, confidence, bahan_meragukan)
            VALUES (%s, %s, %s, %s, %s)
        """)
        cursor.execute(insert_query, (nama_produk, teks_ocr, prediksi, confidence, ", ".join(bahan_meragukan)))
        conn.commit()
        cursor.close()
        st.success("✅ Data berhasil disimpan ke database PostgreSQL!")
    except Exception as e:
        st.error(f"❌ Gagal menyimpan data ke database: {e}")
    finally:
        conn.close()

# ===================== 7️⃣ STREAMLIT UI ===================== #

st.title("Halal Check 🍽️")
option = st.radio("Pilih metode input:", ["Input Teks Manual", "Upload Gambar"])

if option == "Input Teks Manual":
    user_input = st.text_area("Masukkan teks komposisi makanan:", "")
    if st.button("Prediksi") and user_input:
        label, confidence, warning_ingredients = predict_label(user_input)
        st.success(f"**Prediksi: {label}**")
        st.info(f"**Confidence: {confidence}%**")
        if warning_ingredients:
            st.warning(f"⚠ Produk ini mengandung bahan yang perlu diwaspadai: {', '.join(warning_ingredients)}")
        save_to_database("Manual Input", user_input, label, confidence, warning_ingredients)

elif option == "Upload Gambar":
    uploaded_image = st.file_uploader("Upload gambar (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        st.image(uploaded_image, caption="Gambar yang Diupload", use_column_width=True)
        extracted_text = extract_text_from_image(uploaded_image)
        if extracted_text:
            st.write("📄 **Teks yang diekstrak:**", extracted_text)
            label, confidence, warning_ingredients = predict_label(extracted_text)
            st.success(f"**Prediksi: {label}**")
            st.info(f"**Confidence: {confidence}%**")
            if warning_ingredients:
                st.warning(f"⚠ Produk ini mengandung bahan yang perlu diwaspadai: {', '.join(warning_ingredients)}")
            save_to_database("Gambar Upload", extracted_text, label, confidence, warning_ingredients)
