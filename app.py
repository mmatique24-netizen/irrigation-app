import streamlit as st
import pandas as pd
import joblib

# تحميل النموذج المدرب
model = joblib.load("best_ir_model.joblib")

st.title("💧 Smart Irrigation Prediction App")

st.write("أدخل القيم التالية للتنبؤ بكمية الري:")

# إدخال القيم
temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity    = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
rainfall    = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, max_value=500.0, value=10.0)
moisture    = st.number_input("🌱 Soil Moisture (%)", min_value=0.0, max_value=100.0, value=30.0)

# إنشاء DataFrame واحد يحتوي على المدخلات
X_new = pd.DataFrame([[temperature, humidity, rainfall, moisture]],
                     columns=["temperature", "humidity", "rainfall", "moisture"])

# زر التنبؤ
if st.button("🔍 Predict Irrigation"):
    prediction = model.predict(X_new)[0]
    st.success(f"🚰 كمية الري المتوقعة: {prediction:.2f} لتر/م²")
