import streamlit as st
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Stock Trend Prediction", layout="centered")
st.title("📈 ทำนายแนวโน้มราคาหุ้น: Logistic Regression vs ARIMA")

# อินพุตชื่อหุ้น
ticker = st.text_input("กรุณากรอกรหัสหุ้น (เช่น PTT.BK):", "PTT.BK")

# ฟังก์ชันดึงข้อมูลและสร้างโมเดลจากข้อมูลจริง
@st.cache_data(show_spinner=False)
def load_data_and_models(ticker):
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

    # Logistic Regression
    X = df[["MA20", "MA50", "MA100", "RSI", "Upper", "Lower"]]
    y = df["Target"]
    lr_model = LogisticRegression()
    lr_model.fit(X, y)
    latest_features = X.iloc[-1].values

    # ARIMA Forecasting
    close_series = df["Close"]
    arima_model = ARIMA(close_series, order=(5,1,0))
    arima_result = arima_model.fit()
    arima_forecast = arima_result.forecast(steps=7)

    return df, lr_model, latest_features, arima_forecast

# ปุ่มโหลดข้อมูลและฝึกโมเดล
if st.button("🚀 โหลดข้อมูลและสร้างโมเดลทั้งสองแบบ"):
    try:
        df_plot, lr_model, latest_input, arima_forecast = load_data_and_models(ticker)
        st.session_state.df_plot = df_plot
        st.session_state.lr_model = lr_model
        st.session_state.latest_input = latest_input
        st.session_state.arima_forecast = arima_forecast
        st.success("✅ โหลดข้อมูลและสร้างโมเดล Logistic + ARIMA เรียบร้อยแล้ว")

        st.subheader("📊 กราฟราคาปิดย้อนหลัง 5 ปี")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df_plot.index, df_plot["Close"], label="Close", linewidth=1)
        ax.plot(df_plot.index, df_plot["MA20"], label="MA20", linestyle="--")
        ax.plot(df_plot.index, df_plot["MA50"], label="MA50", linestyle="--")
        ax.plot(df_plot.index, df_plot["MA100"], label="MA100", linestyle="--")
        ax.set_title(f"Historical Close Price with MAs: {ticker}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # แสดงฟีเจอร์ล่าสุด
        st.subheader("📌 ฟีเจอร์ล่าสุด (สำหรับ Logistic Regression)")
        st.write(dict(zip(["MA20", "MA50", "MA100", "RSI", "Upper", "Lower"], latest_input)))

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดข้อมูลหรือฝึกโมเดล: {e}")

# ปุ่มทำนาย
if st.button("🔮 ทำนายแนวโน้มด้วย Logistic Regression และ ARIMA"):
    try:
        lr_model = st.session_state.lr_model
        latest_input = st.session_state.latest_input
        arima_forecast = st.session_state.arima_forecast

        lr_result = lr_model.predict([latest_input])[0]
        lr_trend = "Up 📈" if lr_result == 1 else "Down 📉"

        st.subheader("📈 ผลลัพธ์ Logistic Regression")
        st.success(f"แนวโน้มที่คาดการณ์: {lr_trend}")

        st.subheader("🧠 ผลลัพธ์ ARIMA (พยากรณ์ราคาปิด 7 วันถัดไป)")
        forecast_df = pd.DataFrame({"วันถัดไป": range(1,8), "ราคาที่คาดการณ์ (บาท)": arima_forecast})
        st.dataframe(forecast_df.set_index("วันถัดไป"))

        fig2, ax2 = plt.subplots()
        ax2.plot(forecast_df.index, forecast_df["ราคาที่คาดการณ์ (บาท)"], marker="o")
        ax2.set_title("ARIMA Forecast (7 วันถัดไป)")
        ax2.set_xlabel("วัน")
        ax2.set_ylabel("ราคาที่คาดการณ์")
        ax2.grid(True)
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการทำนาย: {e}")
