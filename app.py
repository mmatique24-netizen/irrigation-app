import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

# تحميل النموذج المدرب
model = joblib.load("best_ir_model.joblib")

st.title("🚀 تطبيق التنبؤ بالري + محاكاة الدرون (Dataset & TSP)")

# ===== اختيار طريقة إدخال البيانات =====
mode = st.radio("اختر طريقة إدخال البيانات:", ["🔹 إدخال يدوي", "🔹 استعمال Dataset"])

if mode == "🔹 إدخال يدوي":
    temperature = st.number_input("🌡️ Temperature (°C)", -10.0, 50.0, 25.0)
    humidity = st.number_input("💧 Humidity (%)", 0.0, 100.0, 60.0)
    rainfall = st.number_input("🌧️ Rainfall (mm)", 0.0, 500.0, 10.0)
    moisture = st.number_input("🌱 Soil Moisture (%)", 0.0, 100.0, 30.0)

    input_data = pd.DataFrame([[temperature, humidity, rainfall, moisture]],
                               columns=["temperature", "humidity", "rainfall", "moisture"])

else:
    dataset = pd.read_csv("sample_data.csv")
    st.write("📂 البيانات المتاحة (جزء):")
    st.dataframe(dataset.head())

    row_index = st.slider("اختر دورة الدرون (صف من البيانات)", 0, len(dataset)-1, 0)
    input_data = dataset.iloc[[row_index]][["temperature", "humidity", "rainfall", "moisture"]]

    st.write("✅ القيم المجمعة بواسطة الدرون:")
    st.table(input_data)

# ===== معطيات الحقل =====
st.markdown("### 🏞️ معطيات الحقل")
field_area = st.number_input("📐 مساحة الحقل (m²)", 100.0, 100000.0, 1000.0, step=100.0)
num_sensors = st.number_input("📡 عدد المستشعرات", 2, 12, 5, step=1)
uav_cycles = st.number_input("✈️ عدد دورات الدرون في اليوم", 1, 50, 3, step=1)

# ===== عند الضغط على زر التنبؤ =====
if st.button("🔍 Predict Irrigation"):
    prediction = model.predict(input_data)[0]

    # إنشاء جدول شامل
    data = input_data.copy()
    data["Field Area (m²)"] = field_area
    data["Sensors"] = num_sensors
    data["UAV Cycles/Day"] = uav_cycles
    data["Predicted Irrigation (L/m²)"] = prediction

    st.subheader("📊 جدول القيم المدخلة + النتيجة")
    st.table(data)

    # ===== مواقع المستشعرات =====
    np.random.seed(42)
    x_coords = np.random.uniform(0, np.sqrt(field_area), num_sensors)
    y_coords = np.random.uniform(0, np.sqrt(field_area), num_sensors)
    sensors = list(range(num_sensors))

    # ===== دالة حساب المسافة =====
    def distance(p1, p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    # ===== حل TSP (Brute Force) =====
    coords = list(zip(x_coords, y_coords))
    best_path = None
    best_dist = float("inf")

    for perm in permutations(sensors):
        dist = 0
        for i in range(len(perm)-1):
            dist += distance(coords[perm[i]], coords[perm[i+1]])
        dist += distance(coords[perm[-1]], coords[perm[0]])  # رجوع للبداية

        if dist < best_dist:
            best_dist = dist
            best_path = perm

    # ===== رسم المسار =====
    st.subheader("✈️ المسار الأمثل للدرون (TSP)")
    path_x = [coords[i][0] for i in best_path] + [coords[best_path[0]][0]]
    path_y = [coords[i][1] for i in best_path] + [coords[best_path[0]][1]]

    plt.figure(figsize=(6, 6))
    plt.plot(path_x, path_y, marker="o", linestyle="-", color="blue", label="UAV Path")
    for i, (xx, yy) in enumerate(coords):
        plt.text(xx, yy, f"S{i}", fontsize=9, ha="right")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(f"Optimal UAV Path - {num_sensors} Sensors")
    plt.legend()
    st.pyplot(plt)

    st.info(f"🔵 طول المسار الأمثل: {best_dist:.2f} متر")

    # ===== كمية الري الإجمالية =====
    total_irrigation = prediction * field_area
    st.success(f"💧 كمية الري الإجمالية المتوقعة للحقل: {total_irrigation:.2f} لتر/اليوم")

