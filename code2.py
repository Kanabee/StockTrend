import pandas as pd
import matplotlib.pyplot as plt
import warnings
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
import streamlit as st

warnings.simplefilter('ignore', ConvergenceWarning)

st.title("📈 ARIMA Forecasting for Stocks")

# User input
ticker = st.text_input("Enter stock ticker (e.g. PTTGC.BK):", value='PTTGC.BK')

if st.button("📥 Load Stock Data"):
    df = yf.Ticker(ticker).history(period="5y")[["Close"]]
    series = df['Close'].astype(float)
    series.index = pd.to_datetime(df.index)
    series = series.dropna()

    st.success("✅ Stock data loaded successfully!")
    st.line_chart(series)

    if st.button("🚀 Run ARIMA Model"):
        # Split into train/test
        train_size = int(len(series) * 0.8)
        train, test = series[:train_size], series[train_size:]

        # Step 1: Grid search (p, d, q)
        best_aic = float('inf')
        best_order = None
        best_model = None

        for p in range(0, 5):
            for d in range(0, 2):
                for q in range(0, 5):
                    try:
                        model = ARIMA(train, order=(p, d, q))
                        model_fit = model.fit()
                        aic = model_fit.aic
                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)
                            best_model = model_fit
                    except:
                        continue

        st.success(f'Best ARIMA order: {best_order} with AIC: {best_aic:.2f}')

        # Step 2: Forecast for test set
        forecast_result = best_model.get_forecast(steps=len(test))
        forecast_mean = forecast_result.predicted_mean

        # Step 3: Accuracy metrics
        rmse = np.sqrt(mean_squared_error(test, forecast_mean))
        mape = mean_absolute_percentage_error(test, forecast_mean) * 100
        st.metric("RMSE", f"{rmse:.4f}")
        st.metric("MAPE", f"{mape:.2f}%")

        # Step 4: Plot
        st.subheader("Forecast vs Actual")
        import matplotlib
        matplotlib.use('Agg')
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(train[-60:], label='Train')
        ax.plot(test, label='Actual')
        ax.plot(forecast_mean, label='Forecast', linestyle='--')
        ax.set_title(f'{ticker} Forecast vs Actual (ARIMA{best_order})')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)


