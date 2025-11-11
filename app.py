# import os
# import pickle
# import pandas as pd
# import streamlit as st
# import joblib

# # =======================
# # Folder paths
# # =======================
# ARTIFACTS_FOLDER = r"D:\MLOPs\Dynamic_Price_Predication\Dynamic_Pricing_Model_(using_Mercari_Price_Suggestion_Dataset)\artifacts"
# MODEL_TRAINER_FOLDER = os.path.join(ARTIFACTS_FOLDER, "3_model_trainer")
# DATA_TRANSFORMATION_ARTIFACTS = os.path.join(ARTIFACTS_FOLDER, "2_data_transformation")

# # =======================
# # Load the best model
# # =======================
# summary_file = os.path.join(MODEL_TRAINER_FOLDER, "model_training_summary.csv")
# summary_df = pd.read_csv(summary_file)
# best_model_name = summary_df.sort_values(by="accuracy", ascending=False).iloc[0]["model"]
# best_model_path = os.path.join(MODEL_TRAINER_FOLDER, f"{best_model_name}_model.pkl")
# best_model = joblib.load(best_model_path)

# # Load label encoders
# encoder_file = os.path.join(DATA_TRANSFORMATION_ARTIFACTS, "label_encoders.pkl")
# with open(encoder_file, "rb") as f:
#     label_encoders = pickle.load(f)

# # =======================
# # Streamlit UI
# # =======================
# st.title("📊 Telco Customer Churn Prediction")
# st.subheader("Input Customer Details")

# # =======================
# # Input fields
# # =======================
# gender = st.selectbox("Gender", ["Female", "Male"])
# SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
# Partner = st.selectbox("Partner", ["Yes", "No"])
# Dependents = st.selectbox("Dependents", ["Yes", "No"])
# tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
# PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
# MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
# InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
# OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
# OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
# DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
# TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
# StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
# StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
# Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
# PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
# PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check",
#                                                 "Bank transfer (automatic)", "Credit card (automatic)"])
# MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=0.0)
# TotalCharges = st.number_input("Total Charges", min_value=0.0, value=0.0)

# # =======================
# # Prepare input dataframe
# # =======================
# input_data = {
#     "gender": gender,
#     "SeniorCitizen": SeniorCitizen,
#     "Partner": Partner,
#     "Dependents": Dependents,
#     "tenure": tenure,
#     "PhoneService": PhoneService,
#     "MultipleLines": MultipleLines,
#     "InternetService": InternetService,
#     "OnlineSecurity": OnlineSecurity,
#     "OnlineBackup": OnlineBackup,
#     "DeviceProtection": DeviceProtection,
#     "TechSupport": TechSupport,
#     "StreamingTV": StreamingTV,
#     "StreamingMovies": StreamingMovies,
#     "Contract": Contract,
#     "PaperlessBilling": PaperlessBilling,
#     "PaymentMethod": PaymentMethod,
#     "MonthlyCharges": MonthlyCharges,
#     "TotalCharges": TotalCharges
# }

# input_df = pd.DataFrame([input_data])

# # Encode categorical features using saved label encoders
# for col, le in label_encoders.items():
#     if col != "Churn" and col in input_df.columns:
#         input_df[col] = le.transform(input_df[col].astype(str))

# # Ensure column order matches training data
# feature_order = best_model.feature_names_in_
# input_df = input_df[feature_order]

# # =======================
# # Prediction
# # =======================
# if st.button("Predict"):
#     try:
#         # Predict encoded value
#         prediction = best_model.predict(input_df)[0]

#         # Convert numeric prediction to Yes/No
#         prediction_label = "Yes" if prediction == 1 else "No"

#         st.success(f"✅ Churn Prediction: **{prediction_label}**")
#     except Exception as e:
#         st.error(f"Prediction failed: {e}")



import os
import pickle
import pandas as pd
import streamlit as st
import joblib

# =======================
# Folder paths
# =======================
ARTIFACTS_FOLDER = r"D:\MLOPs\Dynamic_Price_Predication\Dynamic_Pricing_Model_(using_Mercari_Price_Suggestion_Dataset)\artifacts"
MODEL_TRAINER_FOLDER = os.path.join(ARTIFACTS_FOLDER, "3_model_trainer")
DATA_TRANSFORMATION_ARTIFACTS = os.path.join(ARTIFACTS_FOLDER, "2_data_transformation")

# =======================
# Load the best model
# =======================
summary_file = os.path.join(MODEL_TRAINER_FOLDER, "model_training_summary.csv")
summary_df = pd.read_csv(summary_file)

# Select model with highest accuracy
best_model_name = summary_df.sort_values(by="accuracy", ascending=False).iloc[0]["model"]
best_model_path = os.path.join(MODEL_TRAINER_FOLDER, f"{best_model_name}_model.pkl")
best_model = joblib.load(best_model_path)

# Load label encoders
encoder_file = os.path.join(DATA_TRANSFORMATION_ARTIFACTS, "label_encoders.pkl")
with open(encoder_file, "rb") as f:
    label_encoders = pickle.load(f)

# =======================
# Streamlit UI
# =======================
st.title("📊 Telco Customer Churn Prediction")
st.subheader(f"Model selected for inference: **{best_model_name}**")

st.subheader("Input Customer Details")

# =======================
# Input fields
# =======================
gender = st.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])  # user-friendly
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check",
                                                "Bank transfer (automatic)", "Credit card (automatic)"])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=0.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, value=0.0)

# =======================
# Prepare input dataframe
# =======================
# Convert SeniorCitizen from Yes/No to 1/0
senior_map = {"No": 0, "Yes": 1}
SeniorCitizen_encoded = senior_map[SeniorCitizen]

input_data = {
    "gender": gender,
    "SeniorCitizen": SeniorCitizen_encoded,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}

input_df = pd.DataFrame([input_data])

# Encode categorical features using saved label encoders
for col, le in label_encoders.items():
    if col != "Churn" and col in input_df.columns:
        input_df[col] = le.transform(input_df[col].astype(str))

# Ensure column order matches training data
feature_order = best_model.feature_names_in_
input_df = input_df[feature_order]

# =======================
# Prediction
# =======================
if st.button("Predict"):
    try:
        prediction = best_model.predict(input_df)[0]

        # Convert numeric prediction to Yes/No
        prediction_label = "Yes" if prediction == 1 else "No"

        st.success(f"✅ Churn Prediction: **{prediction_label}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
