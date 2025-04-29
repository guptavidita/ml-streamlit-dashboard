import streamlit as st
import pandas as pd
import pickle
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

st.title("📊 Evidently AI Reports")

try:
    with open("data/model.pkl", "rb") as f:
        model, X_train, X_test, y_train, y_test = pickle.load(f)
except:
    st.warning("Please train the model first.")
    st.stop()

df_ref = X_train.copy()
df_cur = X_test.copy()

st.subheader("📈 Data Drift Report")
data_report = Report(metrics=[DataDriftPreset()])
data_report.run(reference_data=df_ref, current_data=df_cur)
data_report.show(mode='inline')

st.subheader("🔍 Classification Performance")
preds = model.predict(X_test)
perf_report = Report(metrics=[ClassificationPreset()])
perf_report.run(reference_data=pd.DataFrame({"target": y_train}),
                current_data=pd.DataFrame({"target": y_test, "prediction": preds}))
perf_report.show(mode='inline')
