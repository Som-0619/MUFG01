import streamlit as st
import pandas as pd
import pickle

# Load the trained model
# Make sure the pickle file is in the same directory as your Streamlit app or provide the full path
try:
    with open('logistic_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please make sure 'logistic_regression_model.pkl' is in the correct directory.")
    st.stop()


st.title('Heart Disease Prediction')

st.write("""
Enter the patient's information to predict the likelihood of heart disease.
""")

# Add input fields for features
# You will need to customize these based on your DataFrame's columns and their data types
age = st.slider('Age', 18, 100, 50)
sex = st.selectbox('Sex', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
chest_pain_type = st.selectbox('Chest Pain Type', [0, 1, 2, 3])
resting_blood_pressure = st.slider('Resting Blood Pressure', 80, 200, 120)
cholesterol = st.slider('Cholesterol', 100, 600, 200)
fasting_blood_sugar = st.selectbox('Fasting Blood Sugar > 120 mg/dL', [0, 1])
resting_ecg = st.selectbox('Resting ECG', [0, 1, 2])
max_heart_rate = st.slider('Maximum Heart Rate Achieved', 60, 220, 150)
exercise_induced_angina = st.selectbox('Exercise Induced Angina', [0, 1])
st_depression = st.slider('ST Depression induced by exercise relative to rest', 0.0, 6.2, 1.0)
st_slope = st.selectbox('Slope of the peak exercise ST segment', [0, 1, 2])
num_major_vessels = st.selectbox('Number of major vessels (0-3) colored by fluoroscopy', [0, 1, 2, 3])
thalassemia = st.selectbox('Thalassemia', [0, 1, 2, 3])


# Create a DataFrame from the input values
input_data = pd.DataFrame([[age, sex, chest_pain_type, resting_blood_pressure, cholesterol,
                            fasting_blood_sugar, resting_ecg, max_heart_rate, exercise_induced_angina,
                            st_depression, st_slope, num_major_vessels, thalassemia]],
                          columns=['age', 'sex', 'chest_pain_type', 'resting_blood_pressure',
                                   'cholesterol', 'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate',
                                   'exercise_induced_angina', 'st_depression', 'st_slope',
                                   'num_major_vessels', 'thalassemia'])

# Make prediction
if st.button('Predict'):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error('Prediction: High likelihood of Heart Disease')
    else:
        st.success('Prediction: Low likelihood of Heart Disease')
