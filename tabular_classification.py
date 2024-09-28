import streamlit as st
import pandas as pd
import joblib

model = joblib.load('model/classification_model.joblib')

def make_prediction(input_data):
    input_df = pd.DataFrame([input_data])

    input_df = pd.get_dummies(input_df, columns=['Geography', 'Gender'], drop_first=False)

    expected_columns = ['CreditScore', 'Geography_France', 'Geography_Spain', 'Geography_Germany',
                        'Gender_Male', 'Gender_Female', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                        'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0 

    input_df = input_df[expected_columns]

    prediction = model.predict(input_df)
    return prediction[0]

def run():
    st.title("Tabular Classification- Churn Prediction")

    st.header("Input Customer Details")
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, step=1)
    geography = st.selectbox("Geography", ['France', 'Spain', 'Germany'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    age = st.number_input("Age", min_value=18, max_value=100, step=1)
    tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, step=1)
    balance = st.number_input("Balance", min_value=0.0, format="%.2f")
    num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, step=1)
    has_cr_card = st.selectbox("Has Credit Card?", ['Yes', 'No'])
    is_active_member = st.selectbox("Is Active Member?", ['Yes', 'No'])
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, format="%.2f")

    has_cr_card = 1 if has_cr_card == 'Yes' else 0
    is_active_member = 1 if is_active_member == 'Yes' else 0

    input_data = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary
    }

    if st.button("Predict Churn"):
        prediction = make_prediction(input_data)
        if prediction == 1:
            st.error("This customer is likely to churn.")
        else:
            st.success("This customer is not likely to churn.")

    if st.button("Back to Homepage"):
        st.session_state.page = "homepage"

if __name__ == "__main__":
    run() 