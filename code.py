import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.title("ทำนายแนวโน้มราคาหุ้นด้วย Logistic Regression")

# ฟังก์ชันบันทึกโมเดลจาก LogisticRegression ที่กำหนดเอง
def save_model_from_input():
    # สร้างข้อมูลจำลอง
    df = pd.DataFrame({
        "MA20": [42.5],
        "MA50": [43.1],
        "MA100": [44.0],
        "RSI": [65.0],
        "Upper": [45.0],
        "Lower": [41.0],
    })
    X = df.values
    y = [1]  # ตัวอย่าง target = 1 (Up)

    model = LogisticRegression()
    model.fit(X, y)

    model_filename = "logistic_regression_stock.pkl"
    with open(model_filename, "wb") as file:
        pickle.dump(model, file)
    return model_filename

# สร้างโมเดลจำลองและบันทึก
model_filename = save_model_from_input()

# โหลดและใช้โมเดลทันที
st.subheader("ใช้โมเดลที่สร้างไว้ล่วงหน้า")
def predict_stock_trend(input_data):
    with open(model_filename, "rb") as file:
        loaded_model = pickle.load(file)
    prediction = loaded_model.predict([input_data])
    return "Up 📈" if prediction[0] == 1 else "Down 📉"

st.markdown("**ลำดับฟีเจอร์:** MA20, MA50, MA100, RSI, Upper, Lower")
input_str = st.text_input("ตัวอย่าง: 42.5, 43.1, 44.0, 65.0, 45.0, 41.0")

if input_str:
    try:
        input_data = np.array([float(x.strip()) for x in input_str.split(",")])
        if len(input_data) != 6:
            st.error("กรุณากรอกข้อมูลฟีเจอร์ให้ครบ 6 ค่า")
        else:
            result = predict_stock_trend(input_data)
            st.success(f"แนวโน้มที่คาดการณ์: {result}")
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการประมวลผลข้อมูล: {e}")

