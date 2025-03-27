
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
forecast_days = st.number_input("Enter number of days to forecast ahead:", min_value=1, max_value=365, value=30)

# Load data button
if st.button("📥 Load Stock Data"):
    df = yf.Ticker(ticker).history(period="5y")[["Close"]]
    series = df['Close'].astype(float)
    series.index = pd.to_datetime(df.index)
    series = series.dropna()

    st.session_state['series'] = series
    st.success("✅ Stock data loaded successfully!")
    st.line_chart(series)

# Run model button
if st.button("🚀 Run ARIMA Model"):
    if 'series' not in st.session_state:
        st.warning("⚠️ Please load stock data first.")
    else:
        series = st.session_state['series']

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

        # Step 2: Forecast future prices
        forecast_result = best_model.get_forecast(steps=forecast_days)
        forecast_mean = forecast_result.predicted_mean

        # Generate forecast dates
        last_date = series.index[-1]
        forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='B')
        forecast_series = pd.Series(forecast_mean.values, index=forecast_index)

        st.subheader("📊 Forecasted Prices")
        st.line_chart(forecast_series)

        # Step 3: Accuracy metrics (on test set)
        if len(test) >= forecast_days:
            test_subset = test[:forecast_days]
            rmse = np.sqrt(mean_squared_error(test_subset, forecast_mean))
            mape = mean_absolute_percentage_error(test_subset, forecast_mean) * 100
            st.metric("RMSE", f"{rmse:.4f}")
            st.metric("MAPE", f"{mape:.2f}%")

        # Step 4: Plot all
        st.subheader("Forecast vs Actual")
        import matplotlib
        matplotlib.use('Agg')
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(series[-60:], label='Historical')
        ax.plot(forecast_series, label='Forecast', linestyle='--')
        ax.set_title(f'{ticker} Price Forecast (ARIMA{best_order})')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
