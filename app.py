import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Dropout Predictor", layout="wide")

@st.cache_resource
def load_model():
    model    = joblib.load('models/best_model_xgb.pkl')
    features = joblib.load('models/selected_features.pkl')
    return model, features

model, features = load_model()

st.title("Student Dropout Prediction")
st.markdown("Predict whether a student will **dropout**, stay **enrolled**, or **graduate**.")

st.sidebar.header("Student Profile")

grade_s1 = st.sidebar.slider("Grade S1 (0–20)", 0.0, 20.0, 12.0, 0.1)
grade_s2 = st.sidebar.slider("Grade S2 (0–20)", 0.0, 20.0, 12.0, 0.1)
approved_s1 = st.sidebar.slider("Units approved S1", 0, 10, 5)
approved_s2 = st.sidebar.slider("Units approved S2", 0, 10, 5)
enrolled_s1 = st.sidebar.slider("Units enrolled S1", 0, 10, 6)
enrolled_s2 = st.sidebar.slider("Units enrolled S2", 0, 10, 6)
age = st.sidebar.slider("Age at enrollment", 17, 60, 20)
scholarship = st.sidebar.selectbox("Scholarship holder", [0, 1], format_func=lambda x: "Yes" if x else "No")
debtor = st.sidebar.selectbox("Debtor", [0, 1], format_func=lambda x: "Yes" if x else "No")
tuition_ok = st.sidebar.selectbox("Tuition fees up to date", [0, 1], format_func=lambda x: "Yes" if x else "No")

input_dict = {
    'Curricular units 1st sem (grade)':    grade_s1,
    'Curricular units 2nd sem (grade)':    grade_s2,
    'Curricular units 1st sem (approved)': approved_s1,
    'Curricular units 2nd sem (approved)': approved_s2,
    'Curricular units 1st sem (enrolled)': enrolled_s1,
    'Curricular units 2nd sem (enrolled)': enrolled_s2,
    'Age at enrollment':                   age,
    'Scholarship holder':                  scholarship,
    'Debtor':                              debtor,
    'Tuition fees up to date':             tuition_ok,
    'Grade_improvement': grade_s2 - grade_s1,
    'Avg_grade':         (grade_s1 + grade_s2) / 2,
    'Total_approved':    approved_s1 + approved_s2,
    'Approval_rate_sem1': approved_s1 / max(enrolled_s1, 1),
    'Approval_rate_sem2': approved_s2 / max(enrolled_s2, 1),
}

input_df = pd.DataFrame([input_dict])
for feat in features:
    if feat not in input_df.columns:
        input_df[feat] = 0
input_df = input_df[features]

pred  = model.predict(input_df)[0]
proba = model.predict_proba(input_df)[0]
labels = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
colors = {0: '#E74C3C', 1: '#F39C12', 2: '#2ECC71'}

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Prediction", labels[pred])
with col2:
    st.metric("Confidence", f"{proba[pred]*100:.1f}%")
with col3:
    st.metric("Risk level", "High" if pred == 0 else ("Medium" if pred == 1 else "Low"))

st.subheader("Probability breakdown")
prob_df = pd.DataFrame({
    'Class': [labels[i] for i in range(3)],
    'Probability': [round(p*100, 1) for p in proba]
})
st.bar_chart(prob_df.set_index('Class'))

st.subheader("What drives this prediction")
explainer = shap.TreeExplainer(model)
shap_out  = explainer(input_df)
raw = shap_out.values
if raw.ndim == 3:
    sv = raw[:, :, pred]
else:
    sv = raw
fig, ax = plt.subplots()
shap.waterfall_plot(
    shap.Explanation(values=sv[0], base_values=shap_out.base_values[0][pred] if shap_out.base_values.ndim > 1 else shap_out.base_values[0],
                     data=input_df.iloc[0].values, feature_names=features),
    max_display=10, show=False
)
st.pyplot(fig)
plt.close()