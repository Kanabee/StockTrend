import streamlit as st
import numpy as np

st.title("ทำนายแนวโน้มราคาหุ้นด้วยโมเดลที่บันทึกไว้")

# ฟังก์ชันสำหรับใช้งานโมเดลทำนายแนวโน้มราคาหุ้น
def predict_stock_trend(input_data):
    with open("trained_model.pkl", "rb") as file:
        loaded_model = pickle.load(file)
    prediction = loaded_model.predict([input_data])
    return "Up" if prediction[0] == 1 else "Down"

