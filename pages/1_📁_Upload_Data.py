import streamlit as st
import pandas as pd
import os

st.title("📁 Upload Your Dataset")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/uploaded_data.csv", index=False)
    st.success("✅ File uploaded and saved!")
    st.write("### Preview of Data", df.head())
    st.write("Shape:", df.shape)
else:
    st.warning("👈 Please upload a CSV file.")
