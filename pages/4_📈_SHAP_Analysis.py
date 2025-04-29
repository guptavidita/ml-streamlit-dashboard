import streamlit as st
import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt

st.title("📈 SHAP Feature Importance")

try:
    with open("data/model.pkl", "rb") as f:
        model, X_train, X_test, y_train, y_test = pickle.load(f)
except:
    st.warning("Please train the model first.")
    st.stop()

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

st.subheader("💡 Summary Plot")
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
st.pyplot(fig)

st.subheader("🔬 Detailed SHAP for a Row")
idx = st.number_input("Select Row Index", 0, len(X_test)-1, 0)
fig2, ax2 = plt.subplots()
shap.force_plot(explainer.expected_value[1], shap_values[1][idx], X_test.iloc[idx], matplotlib=True, show=False)
st.pyplot(fig2)
