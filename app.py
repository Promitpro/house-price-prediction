import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Load the original dataset
@st.cache_data
def load_dataset():
    df = pd.read_csv("house _price_bd.csv")
    df['City'] = df['Address'].astype(str).str.split(',').str[-1].str.strip()
    df['Location'] = df['Address'].astype(str).str.rsplit(',', n=1).str[0].str.strip()
    return df

def load_dataset():
    df = pd.read_csv("house _price_bd.csv", encoding='cp1252')  # or try 'ISO-8859-1'
    df['City'] = df['Address'].astype(str).str.split(',').str[-1].str.strip()
    df['Location'] = df['Address'].astype(str).str.rsplit(',', n=1).str[0].str.strip()
    return df

df = load_dataset()

# Load trained model
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("real_state_house_price_bd_model.pkl")
    return model

model = load_model()

# Sidebar – User Input
st.sidebar.title("Input House Features")

# City
selected_city = st.sidebar.selectbox("Select City", sorted(df['City'].dropna().unique()))

# Filter locations by city
available_locations = df[df['City'] == selected_city]['Location'].dropna().unique()
selected_location = st.sidebar.selectbox("Select Location", sorted(available_locations))

# House type
selected_type = st.sidebar.selectbox("Select Property Type", sorted(df['Type'].dropna().unique()))

# Numerical Inputs
beds = st.sidebar.number_input("Number of Bedrooms", min_value=1, value=3)
bath = st.sidebar.number_input("Number of Bathrooms", min_value=1, value=2)
area = st.sidebar.number_input("Floor Area (sq ft)", min_value=100.0, value=1200.0, step=50.0)

# Main Title
st.title("🏠 House Price Prediction in Bangladesh")
st.image("img.webp", use_container_width=True,)
st.write("Use the sidebar to input house features and predict the expected price (BDT).")

# Prepare input
input_data = pd.DataFrame({
    'Beds': [beds],
    'Bath': [bath],
    'Area': [area],
    'Type': [selected_type],
    'City': [selected_city],
    'Location': [selected_location]
})

# Label encoding using training data
label_encoder_city = LabelEncoder()
label_encoder_location = LabelEncoder()
label_encoder_type = LabelEncoder()

label_encoder_city.fit(df['City'])
label_encoder_location.fit(df['Location'])
label_encoder_type.fit(df['Type'])

# Transform input data
try:
    input_data['City'] = label_encoder_city.transform(input_data['City'])
    input_data['Location'] = label_encoder_location.transform(input_data['Location'])
    input_data['Type'] = label_encoder_type.transform(input_data['Type'])
except ValueError as e:
    st.error(f"Encoding error: {e}")
    input_data['City'] = -1
    input_data['Location'] = -1
    input_data['Type'] = -1

# Prediction
if st.button("🔍 Predict House Price"):
    prediction = model.predict(input_data[['Beds', 'Bath', 'Area', 'Type', 'City', 'Location']])
    st.success(f"Estimated House Price: {prediction[0]:,.2f} BDT")
