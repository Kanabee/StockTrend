import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.title("ทำนายแนวโน้มราคาหุ้นด้วย Logistic Regression")

# อินพุตชื่อหุ้น (ยังไม่ได้ใช้งานจริง แต่สำหรับ UI เตรียมพร้อม)
ticker = st.text_input("กรุณากรอกรหัสหุ้น (เช่น PTT.BK):", "PTT.BK")

def predict_stock_trend(input_data):
    with open(model_filename, "rb") as file:
        loaded_model = pickle.load(file)
    prediction = loaded_model.predict([input_data])
    return "Up" if prediction[0] == 1 else "Down"

