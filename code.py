import streamlit as st
import pickle
import numpy as np

st.title("ทำนายแนวโน้มราคาหุ้นด้วยโมเดลที่บันทึกไว้")

# ฟังก์ชันสำหรับใช้งานโมเดลทำนายแนวโน้มราคาหุ้น
def predict_stock_trend(input_data):
    with open("trained_model.pkl", "rb") as file:
        loaded_model = pickle.load(file)
    prediction = loaded_model.predict([input_data])
    return "Up" if prediction[0] == 1 else "Down"

st.subheader("อัปโหลดไฟล์โมเดล (.pkl)")
uploaded_file = st.file_uploader("เลือกไฟล์ trained_model.pkl", type="pkl")

if uploaded_file:
    try:
        loaded_model = pickle.load(uploaded_file)

        st.subheader("กรอกค่าฟีเจอร์สำหรับทำนาย")
        input_features = st.text_input("เช่น: 1000, 55.0, 70.2 (คั่นด้วย ,)")

        if input_features:
            try:
                input_data = np.array([float(x.strip()) for x in input_features.split(",")])
                prediction = loaded_model.predict([input_data])
                result = "Up" if prediction[0] == 1 else "Down"
                st.success(f"Predicted Trend: {result}")
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการแปลงข้อมูล: {e}")
    except Exception as e:
        st.error(f"ไม่สามารถโหลดโมเดลได้: {e}")
