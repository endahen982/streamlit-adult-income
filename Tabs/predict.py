import numpy as np
from web_functions import predict
from web_functions import load_data
import streamlit as st


def app(df, x, y):
   st.title('Prediksi Pendapatan Orang Dewasa')
   st.image('img8.png')
   st.write('Masukkan fitur berikut untuk memprediksi pendapatan orang dewasa')

   workclass_dict = {'Private': 0, 'Local-gov': 1, 'Self-emp-not-inc': 2, 'Federal-gov': 3, 'State-gov': 4, 'Self-emp-inc': 5, 'Without-pay': 6, 'Never-worked': 7}
   education_dict = {'11th': 0, 'HS-grad': 1, 'Assoc-acdm': 2, 'Some-college': 3, '10th': 4, 'Prof-school': 5, '7th-8th': 6, 'Bachelors': 7, 'Masters': 8, 'Doctorate': 9, '5th-6th': 10, 'Assoc-voc': 11, '9th': 12, '12th': 13, '1st-4th': 14, 'Preschool': 15}
   marital_status_dict = {'Never-married': 0, 'Married-civ-spouse': 1, 'Widowed': 2, 'Divorced': 3, 'Separated': 4, 'Married-spouse-absent': 5, 'Married-AF-spouse': 6}
   occupation_dict = {'Machine-op-inspct': 0, 'Farming-fishing': 1, 'Protective-serv': 2, '?': 3, 'Other-service': 4, 'Prof-specialty': 5, 'Craft-repair': 6, 'Adm-clerical': 7, 'Exec-managerial': 8, 'Tech-support': 9, 'Sales': 10, 'Priv-house-serv': 11, 'Transport-moving': 12, 'Handlers-cleaners': 13, 'Armed-Forces': 14}
   relationship_dict = {'Own-child': 0, 'Husband': 1, 'Not-in-family': 2, 'Unmarried': 3, 'Wife': 4, 'Other-relative': 5}
   race_dict = {'Black': 0, 'White': 1, 'Asian-Pac-Islander': 2, 'Other': 3, 'Amer-Indian-Eskimo': 4}
   native_country_dict = {'United-States': 0, '?': 1, 'Peru': 2, 'Guatemala': 3, 'Mexico': 4, 'Dominican-Republic': 5, 'Ireland': 6, 'Germany': 7, 'Philippines': 8, 'Thailand': 9, 'Haiti': 10, 'El-Salvador': 11, 'Puerto-Rico': 12, 'Vietnam': 13, 'South': 14, 'Columbia': 15, 'Japan': 16, 'India': 17, 'Cambodia': 18, 'Poland': 19, 'Laos': 20, 'England': 21, 'Cuba': 22, 'Taiwan': 23, 'Italy': 24, 'Canada': 25, 'Portugal': 26, 'Ecuador': 27, 'Yugoslavia': 28, 'Hungary': 29, 'Hong': 30, 'Greece': 31, 'Trinadad&Tobago': 32, 'Outlying-US(Guam-USVI-etc)': 33, 'France': 34, 'Holand-Netherlands': 35}
   gender_dict = {'Male': 1, 'Female': 0}

   col1, col2 = st.columns(2)

   with col1:
        workclass = st.selectbox('Kelas Kerja', ('Private','Local-gov', 'Self-emp-not-inc', 'Federal-gov','State-gov', 'Self-emp-inc', 'Without-pay', 'Never-worked'))
   with col2:
        education = st.selectbox('Pendidikan', ('11th', 'HS-grad', 'Assoc-acdm', 'Some-college', '10th','Prof-school', '7th-8th', 'Bachelors', 'Masters', 'Doctorate','5th-6th', 'Assoc-voc', '9th', '12th', '1st-4th', 'Preschool'))
   with col1:
        marital_status = st.selectbox('Status Pernikahan', ('Never-married', 'Married-civ-spouse', 'Widowed', 'Divorced','Separated', 'Married-spouse-absent', 'Married-AF-spouse'))
   with col2:
        occupation = st.selectbox('Pekerjaan', ('Machine-op-inspct', 'Farming-fishing', 'Protective-serv', '?','Other-service', 'Prof-specialty', 'Craft-repair', 'Adm-clerical','Exec-managerial', 'Tech-support', 'Sales', 'Priv-house-serv','Transport-moving', 'Handlers-cleaners', 'Armed-Forces'))
   with col1:
        relationship = st.selectbox('Hubungan', ('Own-child', 'Husband', 'Not-in-family', 'Unmarried', 'Wife','Other-relative'))
   with col2:
        race = st.selectbox('Ras', ('Black', 'White', 'Asian-Pac-Islander', 'Other','Amer-Indian-Eskimo'))
   with col1:
        native_country = st.selectbox('Negara Asal', ('United-States', '?', 'Peru', 'Guatemala', 'Mexico','Dominican-Republic', 'Ireland', 'Germany', 'Philippines','Thailand', 'Haiti', 'El-Salvador', 'Puerto-Rico', 'Vietnam','South', 'Columbia', 'Japan', 'India', 'Cambodia', 'Poland','Laos', 'England', 'Cuba', 'Taiwan', 'Italy', 'Canada', 'Portugal','Ecuador', 'Yugoslavia', 'Hungary', 'Hong', 'Greece','Trinadad&Tobago', 'Outlying-US(Guam-USVI-etc)', 'France','Holand-Netherlands'))
   with col2:
        gender = st.selectbox('Jenis Kelamin', ('Male', 'Female'))

   prediction = None

   if st.button('Prediksi Pendapatan Orang Dewasa'):
      df, x, y = load_data()
      workclass_val = workclass_dict[workclass]
      education_val = education_dict[education]
      marital_status_val = marital_status_dict[marital_status]
      occupation_val = occupation_dict[occupation]
      relationship_val = relationship_dict[relationship]
      race_val = race_dict[race]
      native_country_val = native_country_dict[native_country]
      gender_val = gender_dict[gender]

      st.info("Prediksi Sukses...")

      features = [education_val, workclass_val, occupation_val, marital_status_val, native_country_val, gender_val, race_val, relationship_val]

      prediction, score = predict(x, y, features)

   if prediction is not None:
      if prediction == 0:
         st.warning("Pendapatan <=50K")
      else:
         st.success("Pendapatan >50K")

    # Menambahkan informasi akurasi model jika tersedia
      if 'score' in locals():
         st.write("Model yang digunakan memiliki tingkat akurasi", round(score * 100, 2), "%")
      else:
         st.warning("Silakan tekan tombol untuk melakukan prediksi.")

