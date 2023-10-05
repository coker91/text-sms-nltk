import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

#load save model
model_fraud = pickle.load(open('model_fraud.sav', 'rb'))

tfidf = TfidfVectorizer

loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("new_selected_features_tf-idf.sav", "rb"))))

#judul halaman
st.title('Prediksi SMS Penipuan')

clean_text = st.text_input('Masukkan Teks SMS')

fraud_detection = ' '

if st.button('Hasil Prediksi'):
    predict_fraud = model_fraud.predict(loaded_vec.fit_transform([clean_text]))
    
    if (predict_fraud == 0):
        fraud_detection = 'SMS Normal'
    elif (predict_fraud == 1):
        fraud_detection = 'SMS Spam'
    else:
        fraud_detection = 'SMS Promo'
        
st.success(fraud_detection)
