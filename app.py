import streamlit as st
import pickle
import numpy as np

# 1. Load the trained model
with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('🩺 Diabetes Prediction App')
st.write("Enter the patient's medical details below to predict the likelihood of diabetes.")

# 2. Create input fields for all 8 features
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0)
    glucose = st.number_input('Glucose Level', min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input('Insulin Level', min_value=0, max_value=900, value=79)
    bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=32.0)
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.47)
    age = st.number_input('Age', min_value=21, max_value=120, value=33)

# 3. Create a predict button
if st.button('Predict Diabetes Risk'):
    # Gather the inputs into a numpy array matching the training data format
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # Make the prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[0][1] # Get probability of class 1

    # Display the results
    st.markdown("---")
    if prediction[0] == 1:
        st.error(f'⚠️ **High Risk:** The model predicts the patient has diabetes. (Probability: {prediction_proba:.2%})')
    else:
        st.success(f'✅ **Low Risk:** The model predicts the patient does not have diabetes. (Probability: {prediction_proba:.2%})')
