import streamlit as st
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression

st.title("ทำนายแนวโน้มราคาหุ้นด้วย Logistic Regression")

# อินพุตชื่อหุ้น
ticker = st.text_input("กรุณากรอกรหัสหุ้น (เช่น PTT.BK):", "PTT.BK")

# ฟังก์ชันดึงข้อมูลและสร้างโมเดลจากข้อมูลจริง
def save_model_from_yfinance(ticker):
    df = yf.Ticker(ticker).history(period="5y")[["Close"]]
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    df["MA100"] = df["Close"].rolling(window=100).mean()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["STD20"] = df["Close"].rolling(window=20).std()
    df["Upper"] = df["MA20"] + 2 * df["STD20"]
    df["Lower"] = df["MA20"] - 2 * df["STD20"]
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna()

    X = df[["MA20", "MA50", "MA100", "RSI", "Upper", "Lower"]]
    y = df["Target"]

    model = LogisticRegression()
    model.fit(X, y)

    model_filename = "logistic_regression_stock.pkl"
    with open(model_filename, "wb") as file:
        pickle.dump(model, file)
    return model_filename

# โหลดและใช้โมเดลทันที
def predict_stock_trend(input_data):
    with open(model_filename, "rb") as file:
        loaded_model = pickle.load(file)
    prediction = loaded_model.predict([input_data])
    return "Up 📈" if prediction[0] == 1 else "Down 📉"

# ปุ่มสร้างโมเดลจากข้อมูลจริง
ticker_valid = False
if st.button("🔄 ดึงข้อมูล & สร้างโมเดลจาก yfinance"):
    try:
        model_filename = save_model_from_yfinance(ticker)
        st.success("✅ สร้างโมเดลเรียบร้อยแล้วจากข้อมูลหุ้นจริง")
        ticker_valid = True
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดข้อมูลหรือฝึกโมเดล: {e}")


if st.button("📊 ทำนายแนวโน้มราคาหุ้น") and input_str:
    try:
        input_data = np.array([float(x.strip()) for x in input_str.split(",")])
        if len(input_data) != 6:
            st.error("กรุณากรอกข้อมูลฟีเจอร์ให้ครบ 6 ค่า")
        else:
            result = predict_stock_trend(input_data)
            st.success(f"แนวโน้มที่คาดการณ์สำหรับ {ticker}: {result}")
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการประมวลผลข้อมูล: {e}")

