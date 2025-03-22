import streamlit as st
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Stock Trend Prediction", layout="centered")
st.title("📈 Trend Forcasting")

# อินพุตชื่อหุ้น
ticker = st.text_input("Stock Name")

# ฟังก์ชันดึงข้อมูลและสร้างโมเดลจากข้อมูลจริง
@st.cache_data(show_spinner=False)
def load_data_and_train_model(ticker):
    df = yf.Ticker(ticker).history(period="5y")[["Close"]]
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA25"] = df["Close"].rolling(window=25).mean()
    df["MA75"] = df["Close"].rolling(window=75).mean()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["STD25"] = df["Close"].rolling(window=25).std()
    df["Upper"] = df["MA25"] + 2 * df["STD25"]
    df["Lower"] = df["MA25"] - 2 * df["STD25"]
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna()

    X = df[["MA5", "MA25", "MA75", "RSI", "Upper", "Lower"]]
    y = df["Target"]

    model = LogisticRegression()
    model.fit(X, y)

    latest_features = X.iloc[-1].values  # แถวล่าสุด
    return model, latest_features, df

# ปุ่มโหลดข้อมูลและฝึกโมเดล
if st.button("🔄 Gte Data from yfinance"):
    try:
        model, latest_input, df_plot = load_data_and_train_model(ticker)
        st.session_state.model = model
        st.session_state.latest_input = latest_input
        st.session_state.df_plot = df_plot
        st.success("✅ Model Complete ")
        st.write("**Feature for Prediction:**")
        st.write(dict(zip(["MA5", "MA25", "MA75", "RSI", "Upper", "Lower"], latest_input)))
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดข้อมูลหรือฝึกโมเดล: {e}")

# ปุ่มทำนาย
if st.button("📊 Prediction"):
    if "model" not in st.session_state or "latest_input" not in st.session_state:
        st.error("กรุณากดปุ่มด้านบนเพื่อโหลดข้อมูลและฝึกโมเดลก่อน")
    else:
        try:
            model = st.session_state.model
            latest_input = st.session_state.latest_input
            df_plot = st.session_state.df_plot
            prediction = model.predict([latest_input])[0]
            result = "Up 📈" if prediction == 1 else "Down 📉"
            st.success(f"Trend Forcase {ticker}: {result}")

            # แสดงกราฟราคาหุ้นย้อนหลัง
            st.subheader("📊 5 year Stock Price")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df_plot.index, df_plot["Close"], label="Close", linewidth=1)
            ax.plot(df_plot.index, df_plot["MA5"], label="MA5", linestyle="--")
            ax.plot(df_plot.index, df_plot["MA25"], label="MA25", linestyle="--")
            ax.plot(df_plot.index, df_plot["MA75"], label="MA75", linestyle="--")
            ax.set_title(f"Historical Close Price with MAs: {ticker}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}")
