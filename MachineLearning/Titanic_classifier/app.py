import pickle
import streamlit as st

# Load the pre-trained model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please make sure 'model.pkl' exists in the current directory.")

# Define function for prediction
def predict(age, fare, sex, pclass, embarked, sibsp, parch):
    # Preprocess input data (categorical encoding)
    try:
        sex_encoded = {'Male': 1, 'Female': 0}[sex]
        embarked_encoded = {'C': 0, 'S': 1, 'Q': 2}[embarked]

        # Create the input data as an array
        input_data = [[age, fare, sex_encoded, pclass, embarked_encoded, sibsp, parch]]

        # Make prediction using the model
        prediction = model.predict(input_data)[0]
        return prediction
    except KeyError:
        st.error("Invalid input values. Please check your input.")

# Set app title and header
st.title('Passenger Survival Prediction App')
st.header('Enter Passenger Details:')

# Create columns for layout
left_column, right_column = st.columns(2)

# Collect user input with labels and appropriate data types
with left_column:
    age = st.number_input('Age:', min_value=0)
    fare = st.number_input('Fare:', min_value=0.0)
    sex = st.selectbox('Sex:', ['Male', 'Female'])
    pclass = st.selectbox('Pclass:', [1, 2, 3])

with right_column:
    embarked = st.selectbox('Embarked:', ['C', 'S', 'Q'])
    sibsp = st.number_input('Number of Siblings/Spouses:', min_value=0)
    parch = st.number_input('Number of Parents/Children:', min_value=0)


import random

# Define phrases for positive and negative predictions
positive_phrases = ["You'll live another day", "Fortune smiles upon you", "Survivor in the making"]
negative_phrases = ["Get ready to sleep in your casket", "Prepare for the worst", "Life's journey ends here"]

# Call prediction function if the user clicks the button
if st.button('Predict'):
    # Make the prediction
    prediction = predict(age, fare, sex, pclass, embarked, sibsp, parch)

    # Display prediction result
    if prediction == 1:
        st.success(random.choice(positive_phrases))
    else:
        st.error(random.choice(negative_phrases))
        
