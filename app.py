import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("laptop_price_model.pkl")
le_brand = joblib.load("le_brand.pkl")
le_processor = joblib.load("le_processor.pkl")

st.title("💻 Laptop Price Predictor")

# User inputs
brand = st.selectbox("Select Brand", le_brand.classes_)
ram = st.slider("RAM (GB)", 2, 64, 8)
storage = st.slider("Storage (GB)", 128, 2048, 512, step=128)
processor = st.selectbox("Processor", le_processor.classes_)

# Predict button
if st.button("Predict Price"):
    brand_enc = le_brand.transform([brand])[0]
    processor_enc = le_processor.transform([processor])[0]
    input_data = pd.DataFrame([[brand_enc, ram, storage, processor_enc]], 
                              columns=["Brand", "RAM_GB", "Storage_GB", "Processor"])
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Laptop Price: ₹{int(prediction):,}")