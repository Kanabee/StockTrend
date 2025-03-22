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
@st.cache_data(show_spinner=False)
def load_data_and_train_model(ticker):
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

    latest_features = X.iloc[-1].values  # แถวล่าสุด
    return model, latest_features

model = None
latest_input = None

if st.button("🔄 ดึงข้อมูล & สร้างโมเดลจาก yfinance"):
    try:
        model, latest_input = load_data_and_train_model(ticker)
        st.success("✅ สร้างโมเดลและเตรียมข้อมูลล่าสุดเรียบร้อยแล้ว")
        st.write("**ฟีเจอร์ล่าสุดที่ใช้ในการทำนาย:**")
        st.write(dict(zip(["MA20", "MA50", "MA100", "RSI", "Upper", "Lower"], latest_input)))
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดข้อมูลหรือฝึกโมเดล: {e}")

if st.button("📊 ทำนายแนวโน้มราคาหุ้นจากข้อมูลล่าสุด"):
    if not model or latest_input is None:
        st.error("กรุณากดปุ่มด้านบนเพื่อโหลดข้อมูลและฝึกโมเดลก่อน")
    else:
        try:
            prediction = model.predict([latest_input])[0]
            result = "Up 📈" if prediction == 1 else "Down 📉"
            st.success(f"แนวโน้มที่คาดการณ์สำหรับ {ticker}: {result}")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}")
