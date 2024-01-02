# Import Modul
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

@st.cache_data()
def load_data():
    # Load Dataset
    df = pd.read_csv('adult-income.csv')

    # Features and Target
    x = df[['education','workclass','occupation','marital-status','native-country','gender','race','relationship']]
    y = df[['income']]

    return df, x, y

@st.cache_data()
def train_model(x, y):
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = KNeighborsClassifier(n_neighbors=8)
    model.fit(x_train, y_train)

    # Evaluate the model
    score = model.score(x_test, y_test)

    return model, score

def predict(x, y, features):
    model, score = train_model(x, y)

    # Make predictions
    prediction = model.predict(np.array(features).reshape(1, -1))
    return prediction, score