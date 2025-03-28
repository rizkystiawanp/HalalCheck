# 🍽️ HalalCheck – Automatic Product Halal Status Checker
HalalCheck is a Deep Learning-based system that enables users to check the halal or haram status of a food product by analyzing the list of ingredients obtained from the packaging label. It uses a combination of OCR (Google Vision API) for text extraction, LSTM with Multi-Head Attention for NLP-based classification, and Streamlit as the user interface.

## Problem Background
As awareness of halal food increases among Muslim communities, many products on the market still lack official halal labels or have questionable ingredient compositions.  
Many consumers struggle to verify the halal/haram status of products independently, especially when ingredients are listed using technical terms or E-codes.  
Therefore, this project was developed to provide a quick and automated solution for checking the halal/haram status of food products based on the ingredient list printed on the packaging.

## Project Objectives
✅ Develop an OCR and NLP-based system to identify the halal or haram status of a food item.  
✅ Use Google Cloud Vision API to automatically extract text from food labels.  
✅ Use an LSTM + Multi-Head Attention model to analyze ingredient lists and determine the product's halal status.  
✅ Provide automatic alerts if suspicious ingredients are found in the food composition.  
✅ Store analysis results in a PostgreSQL database for future reference.  
✅ Provide an interactive interface that allows users to input text manually or upload images for analysis.

## Dataset
The dataset used in this project includes lists of halal, haram, and doubtful food ingredients collected from various trusted sources.  
The dataset has been processed to be suitable for training an LSTM-based NLP model, with tokenizer and word embedding to enhance the model's understanding of food ingredient lists.

## Technologies Used
🔹 Python → Primary programming language.  
🔹 TensorFlow & Keras → Build a **Deep Learning model based on LSTM + Multi-Head Attention**.  
🔹 Google Cloud Vision API → Text extraction from food labels using **automatic OCR**.  
🔹 Natural Language Processing (NLP) → Text preprocessing to **clean data and analyze ingredient compositions**.  
🔹 PostgreSQL → Store analysis results for future access.  
🔹 Streamlit → Web-based app deployment on **Hugging Face Spaces**.  
🔹 Matplotlib & Seaborn → Data visualization and model performance analysis.

## Model Evaluation
The LSTM model has been tested and shows **excellent results** in classifying the **halal or haram** status of food ingredient lists.

- Training Accuracy: 98.36%
- Validation Accuracy: 98.57%
- Test Accuracy: 99.10%

### Confusion Matrix Analysis:
- Very low classification errors, indicating the model distinguishes classes well.
- Misclassifications occurred only on 17 Halal samples and 38 Haram samples out of 3,979 test data.

### Classification Report

| **Class** | **Precision** | **Recall** | **F1-Score** | **Support** |
|----------|---------------|------------|--------------|-------------|
| **Halal** | 99%          | 99%        | 99%          | 1,796       |
| **Haram** | 99%          | 99%        | 99%          | 2,183       |

### Conclusion:
- High precision → The model rarely misclassifies Halal/Haram products.
- High recall → The model successfully identifies almost all products correctly.
- The model’s accuracy is excellent and ready for real-world halal/haram food classification.

## Key Features
✔ Automatic text extraction from food labels using Google Vision API (OCR).  
✔ Ingredient classification using an LSTM + Multi-Head Attention model.  
✔ Automatic alerts if doubtful or haram ingredients are detected.  
✔ Supports manual text input & image uploads for flexible use.  
✔ Stores analysis results in PostgreSQL for future reference.  
✔ Interactive Streamlit-based interface, deployed on Hugging Face Spaces.

## How to Use the App
![Demo GIF](Demo.gif)

Try the app at <a href="https://huggingface.co/spaces/rizkystiawanp/HalalCheck">Huggingface</a>  

1️⃣ Upload a food label image or manually input the ingredient list.  
2️⃣ OCR will extract the text from the product label (if using an image).  
3️⃣ The LSTM + Multi-Head Attention NLP model will analyze the ingredient list.  
4️⃣ The system will display the classification result and warn if any suspicious ingredients are found.  
5️⃣ The analysis result will be saved in a PostgreSQL database for future reference.

## Future Development
🔹 Integration with official halal databases to improve analysis accuracy.  
🔹 Development of a mobile application for easier use anywhere.  
🔹 Barcode scanner feature for instant halal status checking.  
🔹 Expanded dataset and enhanced NLP model for better classification accuracy.  
🔹 Multilingual support for global usability.  
🔹 Save more comprehensive analysis results in the database for user access.

## ⚠ Disclaimer
🔸 This app is not a substitute for official halal certification and is only intended to provide preliminary information to users.  
🔸 Classification results are not absolute, as halal regulations may vary across countries.  
🔸 Users are advised to still refer to official halal certification from relevant authorities before consuming a product.

## Developed By
**[Rizky SP](https://github.com/rizkystiawanp)** : Data Scientist  
**[Dzaki A.](https://github.com/DzakiAF)** : Data Scientist  
**[Dwi H.](https://github.com/dwihst)** : Data Analyst  
**[Aditya W.](https://github.com/bUtekwijay)** : Data Engineer  
**[Rizki R.](https://github.com/rizkeyyy)** : Data Engineer

[See the presentation as PDF](presentation.pdf)
