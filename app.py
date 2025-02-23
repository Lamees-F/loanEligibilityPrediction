import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


st.set_page_config(page_title='Loan Eligibility Prediction', page_icon='💰')
# Load the model
def load_model():
    model = joblib.load('model (2).joblib')
    return model

model = load_model()

# Title of the app
st.title('💰Loan Eligibility Prediction')
st.write("""
This app predicts loan eligibility based on user input. 
Please fill in the details below and click **Predict**.
""")

# Divide the input form into columns
col1, col2 = st.columns(2)

# Personal Details
with col1:
    st.subheader("Personal Details")
    gender = st.selectbox(
        'Gender',
        ['Male', 'Female'],
        help="Select your gender."
    )
    married = st.selectbox(
        'Married',
        ['Yes', 'No'],
        help="Are you married?"
    )
    dependents = st.selectbox(
        'Dependents',
        ['0', '1', '2', '3+'],
        help="Number of dependents."
    )
    education = st.selectbox(
        'Education',
        ['Graduate', 'Not Graduate'],
        help="Select your education level."
    )
    self_employed = st.selectbox(
        'Self Employed',
        ['Yes', 'No'],
        help="Are you self-employed?"
    )

# Financial Details
with col2:
    st.subheader("Financial Details")
    applicant_income = st.number_input(
        'Applicant Income (USD)',
        min_value=0,
        help="Your monthly income."
    )
    coapplicant_income = st.number_input(
        'Coapplicant Income (USD)',
        min_value=0,
        help="Coapplicant's monthly income (if applicable)."
    )
    loan_amount = st.number_input(
        'Loan Amount (USD)',
        min_value=0,
        help="The loan amount you are requesting."
    )
    loan_amount_term = st.number_input(
        'Loan Amount Term (Months)',
        min_value=0,
        help="The loan repayment term in months."
    )
    credit_history = st.selectbox(
        'Credit History',
        ['0', '1'],
        help="Your credit history (0 = Bad, 1 = Good)."
    )
    property_area = st.selectbox(
        'Property Area',
        ['Urban', 'Rural', 'Semiurban'],
        help="The location of the property."
    )

# Create a DataFrame from user input
data = {
    'Gender': gender,
    'Married': married,
    'Dependents': dependents,
    'Education': education,
    'Self_Employed': self_employed,
    'ApplicantIncome': applicant_income,
    'CoapplicantIncome': coapplicant_income,
    'LoanAmount': loan_amount,
    'Loan_Amount_Term': loan_amount_term,
    'Credit_History': credit_history,
    'Property_Area': property_area
}

input_df = pd.DataFrame(data, index=[0])

# Display the user input features
st.subheader('User Input Features')
st.write(input_df)

# Preprocess the input data
def preprocess_input(input_df):
    # Apply Label Encoding to categorical features
    categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    for column in categorical_columns:
        le = LabelEncoder()
        input_df[column] = le.fit_transform(input_df[column])
    
    # Apply MinMax Scaling to numeric features
    numeric_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    scaler = MinMaxScaler()
    input_df[numeric_columns] = scaler.fit_transform(input_df[numeric_columns])
    
    return input_df

# Preprocess the input
processed_input = preprocess_input(input_df)

# Predict
if st.button('Predict'):
    prediction = model.predict(processed_input)
    st.subheader('Prediction')
    st.write('Eligible' if prediction[0] == 1 else 'Not Eligible')