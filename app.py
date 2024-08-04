#! C:\Users\ODUTOPX\OneDrive - AbbVie Inc (O365)\Desktop\Project Heart Attack\myenv\Scripts\python.exe


import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns



# Load the dataset and train the model
def load_and_train_model():
    # Load the dataset
    file_path = 'heart_attack.csv'
    heart_data = pd.read_csv(file_path)
    
    # Data preprocessing
    X = heart_data.drop('HeartDiseaseorAttack', axis=1)
    y = heart_data['HeartDiseaseorAttack']
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)
    
    return model, scaler, X.columns

# Correctly unpack all returned values from the function
model, scaler, feature_names = load_and_train_model()

# Streamlit app layout
st.title('Heart Attack Risk Predictor')

st.sidebar.header('Input Health Information')

# Input sliders with unique keys
highbp = st.sidebar.selectbox('High Blood Pressure', [0, 1], key='highbp_selectbox')
highchol = st.sidebar.selectbox('High Cholesterol', [0, 1], key='highchol_selectbox')
cholcheck = st.sidebar.selectbox('Cholesterol Check', [0, 1], key='cholcheck_selectbox')
bmi = st.sidebar.slider('BMI', 10.0, 50.0, 25.0, key='bmi_slider')
smoker = st.sidebar.selectbox('Smoker', [0, 1], key='smoker_selectbox')
stroke = st.sidebar.selectbox('Stroke', [0, 1], key='stroke_selectbox')
diabetes = st.sidebar.selectbox('Diabetes', [0, 1], key='diabetes_selectbox')
physactivity = st.sidebar.selectbox('Physical Activity', [0, 1], key='physactivity_selectbox')
fruits = st.sidebar.selectbox('Fruits Consumption', [0, 1], key='fruits_selectbox')
veggies = st.sidebar.selectbox('Vegetables Consumption', [0, 1], key='veggies_selectbox')
hvyalcohol = st.sidebar.selectbox('Heavy Alcohol Consumption', [0, 1], key='hvyalcohol_selectbox')
anyhealthcare = st.sidebar.selectbox('Any Healthcare', [0, 1], key='anyhealthcare_selectbox')

nodocbcost = st.sidebar.selectbox('No Doctor Due to Cost', [0, 1], key='nodocbcost_selectbox')
genhlth = st.sidebar.slider('General Health', 1, 5, 3, key='genhlth_slider')
menthlth = st.sidebar.slider('Mental Health', 0, 30, 0, key='menthlth_slider')
physhlth = st.sidebar.slider('Physical Health', 0, 30, 0, key='physhlth_slider')
diffwalk = st.sidebar.selectbox('Difficulty Walking', [0, 1], key='diffwalk_selectbox')
sex = st.sidebar.selectbox('Sex', [0, 1], key='sex_selectbox')
age = st.sidebar.slider('Age', 20, 100, 50, key='age_slider')
education = st.sidebar.selectbox('Education Level', [1, 2, 3, 4, 5], key='education_selectbox')
income = st.sidebar.slider('Income', 1, 10, 2, key='income_slider')


# Create a DataFrame for prediction including all features
input_data = pd.DataFrame({
'HighBP': [highbp],
'HighChol': [highchol],
'CholCheck': [cholcheck],
'BMI': [bmi],
'Smoker': [smoker],
'Stroke': [stroke],
'Diabetes': [diabetes],
'PhysActivity': [physactivity],
'Fruits': [fruits],
'Veggies': [veggies],
'HvyAlcoholConsump': [hvyalcohol],
'AnyHealthcare': [anyhealthcare],
'NoDocbcCost': [nodocbcost],
'GenHlth': [genhlth],
'MentHlth': [menthlth],
'PhysHlth': [physhlth],
'DiffWalk': [diffwalk],
'Sex': [sex],
'Age': [age],
'Education': [education],
'Income': [income]
})

# Debugging statements
#st.write("Feature names used in training:", feature_names)
#st.write("Input data columns:", input_data.columns)
#st.write("Input data:", input_data)

# Ensure the feature names used in training are in the input data
for feature in feature_names:
    if feature not in input_data.columns:
        st.error(f"Missing feature: {feature}")
        input_data[feature] = 0  # Add the missing feature with default value

# Scale the input data
try:
    input_data_scaled = scaler.transform(input_data)
    # Predict heart attack risk
    prediction = model.predict_proba(input_data_scaled)[0, 1]
    
    st.subheader('Prediction Result')
    st.write(f'Probability of Heart Attack: {prediction:.2f}')
except ValueError as e:
    st.error(f"Error scaling input data: {e}")

# Optionally, compare with normal ranges (example ranges)
st.subheader('Health Metrics Comparison')
st.write(f'Age: {age} years')
st.write(f'Sex: {"Male" if sex == 0 else "Female"}')
st.write(f'High Blood Pressure: {"Yes" if highbp == 1 else "No"}')
st.write(f'BMI: {bmi}')
st.write(f'Smoker: {"Yes" if smoker == 1 else "No"}')
st.write(f'Stroke: {"Yes" if stroke == 1 else "No"}')
st.write(f'Diabetes: {"Yes" if diabetes == 1 else "No"}')

# Data Visualization (EDA tab)
st.sidebar.header('Explore Underlying Data')

if st.sidebar.checkbox('Show Data Visualization'):
    heart_data = pd.read_csv('heart_attack.csv')
    
    st.subheader('Dataset Overview')
    st.write(heart_data.head())
    
    st.subheader('Correlation Heatmap')
    plt.figure(figsize=(10, 6))
    sns.heatmap(heart_data.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    fig = plt.gcf() 
    st.pyplot(fig)

    # st.subheader('BMI vs Age')
    # plt.figure(figsize=(10, 6))
    # sns.scatterplot(data=heart_data, x='Age', y='BMI', hue='HeartDiseaseorAttack', palette='coolwarm')
    # plt.xlabel('Age')
    # plt.ylabel('BMI')
    #st.pyplot()


