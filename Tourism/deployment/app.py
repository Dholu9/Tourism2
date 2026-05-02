import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="Amidho/Tourism", filename="best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Buyer Prediction
st.title("Tourism Package Buyer Prediction App")
st.write("""
This application predicts the likelihood of a customer to buy the tourism package.
Please enter the details below to get a prediction.
""")

# User input
TypeofContact = st.selectbox("TypeofContact", ["Self Enquiry", "Company Invited"])
Occupation = st.selectbox("Occupation", ["Free Lancer", "Salaried", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Female", "Male"])
ProductPitched = st.selectbox("ProductPitched", ["Deluxe", "Basic","Standard","Super Deluxe", "King"])
MaritalStatus = st.selectbox("MaritalStatus", ["Single", "Unmarried", "Married", "Divorced"])
Designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "VP", "AVP"])
OwnCar = st.selectbox("OwnCar", ["Yes", "No"])
Passport = st.selectbox("Passport", ["Yes", "No"])
MonthlyIncome = st.number_input("MonthlyIncome")
NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=100, step=1)
PitchSatisfactionScore = st.number_input("PitchSatisfactionScore")
NumberOfTrips = st.number_input("NumberOfTrips")
PreferredPropertyStar = st.number_input("PreferredPropertyStar", min_value=0, max_value=5, step=1)
NumberOfFollowups = st.number_input("NumberOfFollowups")
NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", min_value=0, max_value=300, step=1)
DurationOfPitch = st.number_input("DurationOfPitch")
Age = st.number_input("Age", min_value=1, max_value=100, step=1)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'ProductPitched': ProductPitched,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation,
    'OwnCar': OwnCar,
    'Passport': Passport,
    'MonthlyIncome': MonthlyIncome,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'NumberOfTrips': NumberOfTrips,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfFollowups': NumberOfFollowups,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'DurationOfPitch': DurationOfPitch,
    'Age': Age
}])


if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Buyer" if prediction == 1 else "Non-Buyer"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
