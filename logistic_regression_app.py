import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('logistic_regression_model_updated.joblib')
pipeline = joblib.load('logistic_regression_pipeline.joblib')

st.image("engage_jooust_branding.png",caption="Engage brands")
st.markdown("[ENGAGE Program](https://engage.uonbi.ac.ke)")

# Streamlit app
st.title('Child Participation Prediction App')

# Collect user input
diarrhea_status = st.selectbox('Diarrhea Status', options=['No', 'Yes'])
weight = st.number_input('Weight (kg)', value=12.5)
height = st.number_input('Height (cm)', value=95.0)
residence = st.selectbox('Residence', options=['Urban', 'Rural'])
sex = st.selectbox('Sex', options=['Female', 'Male'])
age = st.number_input('Age (months)', value=30)
age_category = st.selectbox('Age Category', options=['0-11', '12-23', '24-35', '36-47', '48-59'])
education = st.selectbox('Mother\'s Education Level', options=['No Education', 'Primary', 'Secondary', 'Higher'])
wealth_index = st.selectbox('Wealth Index', options=['Poorest', 'Fourth', 'Middle', 'Second', 'Richest'])

# Create a DataFrame for the new data
new_data = pd.DataFrame({
    ' diarrhea_status': pd.Categorical([diarrhea_status], categories=['No', 'Yes'], ordered=False),
    'weight': [weight],
    'height': [height],
    'residence': pd.Categorical([residence], categories=['Urban', 'Rural'], ordered=False),
    'Sex': pd.Categorical([sex], categories=['Female', 'Male'], ordered=False),
    'age': [age],
    'age_category': pd.Categorical([age_category], categories=['0-11', '12-23', '24-35', '36-47', '48-59'], ordered=False),
    'education': pd.Categorical([education], categories=['No Education', 'Primary', 'Secondary', 'Higher'], ordered=False),
    'wealth_index': pd.Categorical([wealth_index], categories=['Poorest', 'Fourth', 'Middle', 'Second', 'Richest'], ordered=False)
})

df_dummies = pd.get_dummies(new_data, columns=[' diarrhea_status','age_category','education', 'wealth_index','Sex', 'residence'], drop_first=True, dtype=int)

# Button to make predictions
if st.button('Predict'):
    predictions = pipeline.predict(df_dummies)
    #st.write(predictions[0])
    if predictions[0] == 'Yes':
        st.write(f'Predictions: The child participated in the Polio campaign')
    else:
        st.write("Predictions: The child did not participate in the Polio campaign")
