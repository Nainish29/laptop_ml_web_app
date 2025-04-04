import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load the data
df = pd.read_csv("laptop_prices.csv")

# Encode categorical columns
le_brand = LabelEncoder()
le_processor = LabelEncoder()
le_touch = LabelEncoder()
le_os = LabelEncoder()

df["Brand"] = le_brand.fit_transform(df["Brand"])
df["Processor"] = le_processor.fit_transform(df["Processor"])
df["Touchscreen"] = le_touch.fit_transform(df["Touchscreen"])
df["OS"] = le_os.fit_transform(df["OS"])

# Features & Target
X = df.drop("Price", axis=1)
y = df["Price"]

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# UI
st.title("💻 Advanced Laptop Price Predictor")

brand = st.selectbox("Brand", le_brand.classes_)
processor = st.selectbox("Processor", le_processor.classes_)
ram = st.slider("RAM (GB)", 2, 64, 8)
ssd = st.slider("SSD (GB)", 128, 2048, 512)
hdd = st.slider("HDD (GB)", 0, 2000, 500)
screen_size = st.slider("Screen Size (inches)", 10.0, 18.0, 15.6)
touchscreen = st.selectbox("Touchscreen", le_touch.classes_)
os = st.selectbox("Operating System", le_os.classes_)

if st.button("Predict Price"):
    input_data = pd.DataFrame([[
        le_brand.transform([brand])[0],
        le_processor.transform([processor])[0],
        ram,
        ssd,
        hdd,
        screen_size,
        le_touch.transform([touchscreen])[0],
        le_os.transform([os])[0],
    ]], columns=["Brand", "Processor", "RAM", "SSD", "HDD", "ScreenSize", "Touchscreen", "OS"])

    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Laptop Price: ₹{int(prediction):,}")
