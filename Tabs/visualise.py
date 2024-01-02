# visualise.py
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from mlxtend.plotting import plot_decision_regions
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from web_functions import train_model

def app(df, x, y):

    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title("Halaman Visualisasi Prediksi Pendapatan Orang Dewasa")

    # Contoh data x_train, x_test, y_train, y_test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Training model KNN
    classifier = KNeighborsClassifier(n_neighbors=8, metric='minkowski', p=2)
    classifier.fit(x_train, y_train)

    # Misalnya, y_pred_prob adalah probabilitas hasil prediksi model
    y_pred_prob = classifier.predict_proba(x_test)[:, 1]

    # Menghitung nilai False Positive Rate (FPR) dan True Positive Rate (TPR)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    # Menghitung Area Under Curve (AUC)
    roc_auc = auc(fpr, tpr)

    # Menampilkan ROC Curve menggunakan Streamlit
    st.subheader('Visualisasi ROC Curve')
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()

    # Tampilkan grafik menggunakan Streamlit
    st.pyplot()

    # Visualisasi KNN
    st.subheader('Visualisasi K-Nearest Neighbors (KNN)')

    # Buat dataset dengan 2 pusat
    centers = 2
    n_samples = 200

    x, y = make_blobs(n_samples=n_samples, centers=centers, random_state=42, cluster_std=2.5)
    df = pd.DataFrame(data=x, columns=['workclass', 'education'])
    df['income'] = y

    # Pisahkan fitur dan target
    x = df[['workclass', 'education']]
    y = df['income']

    # Pisahkan data menjadi data latih dan data uji
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Skalakan fitur
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Latih model k-NN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train_scaled, y_train)

    # Visualisasi hasil prediksi
    plt.figure(figsize=(10, 6))

    # Visualisasi data latih
    sns.scatterplot(x='workclass', y='education', hue='income', data=df, palette='viridis', s=100)

    # Visualisasi batas keputusan
    plot_decision_regions(x_train_scaled, np.array(y_train), clf=knn, legend=2)

    plt.title('k-NN Decision Boundary')
    plt.xlabel('workclass')
    plt.ylabel('education')
    plt.title('Decision Boundary for K-Nearest Neighbors (KNN)')
    st.pyplot()
