import streamlit as st
import pandas as pd
import pickle as pkl

# Load the model and preprocessor
with open("model_dtr.pkl", "rb") as f:
    model = pkl.load(f)
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pkl.load(f)

st.title("Crop Yield Prediction")
st.image(
    "https://images.nationalgeographic.org/image/upload/t_edhub_resource_key_image/v1638892233/EducationHub/photos/crops-growing-in-thailand.jpg",
    width=450,
)
st.write("Enter the following details to predict the crop yield.")

year = st.number_input("Year:", value=2023)
avg_rainfall = st.number_input("Average rainfall (in mm/year):", value=78.0)
pesticides = st.number_input("Pesticides (in tonnes):", value=10.0)
avg_temp = st.number_input("Average temperature (in °C):", value=25.0)
area = st.selectbox(
    "Area:",
    options=[
        "Albania",
        "Algeria",
        "Angola",
        "Argentina",
        "Armenia",
        "Australia",
        "Austria",
        "Azerbaijan",
        "Bahamas",
        "Bahrain",
        "Bangladesh",
        "Belarus",
        "Belgium",
        "Botswana",
        "Brazil",
        "Bulgaria",
        "Burkina Faso",
        "Burundi",
        "Cameroon",
        "Canada",
        "Central African Republic",
        "Chile",
        "Colombia",
        "Croatia",
        "Denmark",
        "Dominican Republic",
        "Ecuador",
        "Egypt",
        "El Salvador",
        "Eritrea",
        "Estonia",
        "Finland",
        "France",
        "Germany",
        "Ghana",
        "Greece",
        "Guatemala",
        "Guinea",
        "Guyana",
        "Haiti",
        "Honduras",
        "Hungary",
        "India",
        "Indonesia",
        "Iraq",
        "Ireland",
        "Italy",
        "Jamaica",
        "Japan",
        "Kazakhstan",
        "Kenya",
        "Latvia",
        "Lebanon",
        "Lesotho",
        "Libya",
        "Lithuania",
        "Madagascar",
        "Malawi",
        "Malaysia",
        "Mali",
        "Mauritania",
        "Mauritius",
        "Mexico",
        "Montenegro",
        "Morocco",
        "Mozambique",
        "Namibia",
        "Nepal",
        "Netherlands",
        "New Zealand",
        "Nicaragua",
        "Niger",
        "Norway",
        "Pakistan",
        "Papua New Guinea",
        "Peru",
        "Poland",
        "Portugal",
        "Qatar",
        "Romania",
        "Rwanda",
        "Saudi Arabia",
        "Senegal",
        "Slovenia",
        "South Africa",
        "Spain",
        "Sri Lanka",
        "Sudan",
        "Suriname",
        "Sweden",
        "Switzerland",
        "Tajikistan",
        "Thailand",
        "Tunisia",
        "Turkey",
        "Uganda",
        "Ukraine",
        "United Kingdom",
        "Uruguay",
        "Zambia",
        "Zimbabwe",
    ],
)
crop = st.selectbox(
    "Crop:",
    options=[
        "Maize",
        "Potatoes",
        "Rice, paddy",
        "Sorghum",
        "Soybeans",
        "Wheat",
        "Cassava",
        "Sweet potatoes",
        "Plantains and others",
        "Yams",
    ],
)

cols = [
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
    "Area",
    "Item",
]

data = pd.DataFrame(
    {
        "Year": [year],
        "average_rain_fall_mm_per_year": [avg_rainfall],
        "pesticides_tonnes": [pesticides],
        "avg_temp": [avg_temp],
        "Area": [area],
        "Item": [crop],
    }
)
data = data[cols]
data = preprocessor.transform(data)
prediction = model.predict(data)
if st.button("Predict"):
    result = f"Estimated Crop Yield:  {prediction[0]} hg/ha"
    st.success(result)
