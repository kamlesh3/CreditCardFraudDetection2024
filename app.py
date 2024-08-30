import streamlit as st
import numpy as np
import joblib
Loaded_model = joblib.load("best_credit_card_model.pkl")

st.title("Credit Card Fraud Detection Using RandomForestClassifier Model")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

# create input fields for user to enter feature values
input_df = st.text_input('Input All features')
input_df_lst = input_df.split(',')
# create a button to submit input and get prediction
submit = st.button("Submit")

if submit:
    # get input feature values
    features = np.array(input_df_lst, dtype=np.float64)
    # make prediction
    prediction = Loaded_model.predict(features.reshape(1,-1))
    # display result
    if prediction[0] == 0:
        st.title("Normal transaction")
    else:
        st.title("Fraudulent transaction")
