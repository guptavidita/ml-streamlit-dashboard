import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

st.title("🤖 Train Machine Learning Model")

if not os.path.exists("data/uploaded_data.csv"):
    st.warning("Please upload data first.")
    st.stop()

df = pd.read_csv("data/uploaded_data.csv")
target = st.selectbox("Select Target Variable", df.columns)

if st.button("Train Model"):
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    with open("data/model.pkl", "wb") as f:
        pickle.dump((model, X_train, X_test, y_train, y_test), f)

    st.success("✅ Model trained and saved!")
