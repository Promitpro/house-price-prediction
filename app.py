import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Load the original dataset
@st.cache_data
def load_dataset():
    df = pd.read_csv("property_listing_data_in_Bangladesh.csv")
    df['city'] = df['adress'].astype(str).str.split(',').str[-1].str.strip()
    df['location'] = df['adress'].astype(str).str.rsplit(',', n=1).str[0].str.strip()
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
selected_city = st.sidebar.selectbox("Select City", sorted(df['city'].dropna().unique()))

# Filter locations by city
available_locations = df[df['city'] == selected_city]['location'].dropna().unique()
selected_location = st.sidebar.selectbox("Select Location", sorted(available_locations))

# House type
selected_type = st.sidebar.selectbox("Select Property Type", sorted(df['type'].dropna().unique()))

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
    'beds': [beds],
    'bath': [bath],
    'area': [area],
    'type': [selected_type],
    'city': [selected_city],
    'location': [selected_location]
})

# Label encoding using training data
label_encoder_city = LabelEncoder()
label_encoder_location = LabelEncoder()
label_encoder_type = LabelEncoder()

label_encoder_city.fit(df['city'])
label_encoder_location.fit(df['location'])
label_encoder_type.fit(df['type'])

# Transform input data
try:
    input_data['city'] = label_encoder_city.transform(input_data['city'])
    input_data['location'] = label_encoder_location.transform(input_data['location'])
    input_data['type'] = label_encoder_type.transform(input_data['type'])
except ValueError as e:
    st.error(f"Encoding error: {e}")
    input_data['city'] = -1
    input_data['location'] = -1
    input_data['type'] = -1

# Prediction
if st.button("🔍 Predict Rent Price"):
    prediction = model.predict(input_data[['beds', 'bath', 'area', 'type', 'city', 'location']])
    st.success(f"Estimated Monthly Rent: {prediction[0]:,.2f} BDT")
