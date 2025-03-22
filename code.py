import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.title("ทำนายแนวโน้มราคาหุ้นด้วย Logistic Regression")

# อัปโหลดไฟล์โมเดล
st.subheader("อัปโหลดไฟล์โมเดล (logistic_regression_stock.pkl)")
uploaded_file = st.file_uploader("เลือกไฟล์ .pkl", type="pkl")

if uploaded_file:
    try:
        loaded_model = pickle.load(uploaded_file)

        st.subheader("กรอกค่าฟีเจอร์")
        st.markdown("**ลำดับฟีเจอร์:** MA20, MA50, MA100, RSI, Upper, Lower")
        input_str = st.text_input("ตัวอย่าง: 42.5, 43.1, 44.0, 65.0, 45.0, 41.0")

        if input_str:
            try:
                input_data = np.array([float(x.strip()) for x in input_str.split(",")])
                if len(input_data) != 6:
                    st.error("กรุณากรอกข้อมูลฟีเจอร์ให้ครบ 6 ค่า")
                else:
                    prediction = loaded_model.predict([input_data])
                    result = "Up 📈" if prediction[0] == 1 else "Down 📉"
                    st.success(f"แนวโน้มที่คาดการณ์: {result}")
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการประมวลผลข้อมูล: {e}")
    except Exception as e:
        st.error(f"ไม่สามารถโหลดโมเดลได้: {e}")
else:
    st.info("กรุณาอัปโหลดไฟล์โมเดลเพื่อเริ่มต้น")
