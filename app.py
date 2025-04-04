import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("laptop_data.csv")

# Encode categorical features
le_brand = LabelEncoder()
le_processor = LabelEncoder()
df["Brand"] = le_brand.fit_transform(df["Brand"])
df["Processor"] = le_processor.fit_transform(df["Processor"])

# Split features and target
X = df.drop("Price", axis=1)
y = df["Price"]

# Train model inside the app
model = RandomForestRegressor()
model.fit(X, y)

# Streamlit UI
st.title("💻 Laptop Price Predictor")

brand = st.selectbox("Select Brand", le_brand.classes_)
ram = st.slider("RAM (GB)", 2, 64, 8)
storage = st.slider("Storage (GB)", 128, 2048, 512, step=128)
processor = st.selectbox("Processor", le_processor.classes_)

if st.button("Predict Price"):
    brand_encoded = le_brand.transform([brand])[0]
    processor_encoded = le_processor.transform([processor])[0]
    input_df = pd.DataFrame([[brand_encoded, ram, storage, processor_encoded]],
                            columns=["Brand", "RAM_GB", "Storage_GB", "Processor"])
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Laptop Price: ₹{int(prediction):,}")
