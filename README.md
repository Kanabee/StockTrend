# 📈 Stock Trend Prediction

A machine learning web application for analyzing historical stock prices and predicting the **next-day price direction (Up / Down)** using technical indicators and Logistic Regression.

The application retrieves market data from Yahoo Finance and provides an interactive interface built with Streamlit.

## 🌐 Live Demo

**English Version**
https://stocktrend-dcc8wybbjikjdlgtg6j7pj.streamlit.app/

**Japanese Version / 日本語版**
https://stocktrend-ibgvicgbw3qbljc623ky3e.streamlit.app/

---

## Project Overview

Financial market data contains large amounts of historical price information that can be transformed into technical indicators for market analysis.

This project explores how historical stock prices and technical indicators can be combined with a machine learning classification model to estimate whether the **next trading day's closing price will move Up or Down**.

The project covers the complete workflow from:

**Market Data → Feature Engineering → Machine Learning → Prediction → Interactive Web Application**

---

## Project Objective

The main objectives of this project are:

* Retrieve historical stock market data automatically
* Generate technical indicators from historical prices
* Transform stock movement into a binary classification problem
* Train a machine learning model to identify price direction
* Provide users with an interactive interface for stock analysis
* Visualize historical stock prices and moving averages

---

## How It Works

The application follows the workflow below:

`Stock Ticker → Yahoo Finance → 5-Year Historical Data → Technical Indicators → Logistic Regression → Up / Down Prediction → Streamlit`

### 1. Stock Selection

Users enter a stock ticker symbol in the Streamlit application.

Examples:

`AAPL` — Apple
`MSFT` — Microsoft
`TSLA` — Tesla

The application then retrieves the stock's historical market data.

---

## Data Source

Historical stock price data is retrieved dynamically using the **yfinance** Python library.

The application uses approximately **5 years of historical closing-price data** for model training and analysis.

This allows the model to be rebuilt using data for the stock selected by the user.

---

## Feature Engineering

Historical closing prices are transformed into several technical indicators.

### Moving Averages

Three moving averages are calculated:

* **MA5** — 5-day moving average
* **MA25** — 25-day moving average
* **MA75** — 75-day moving average

Moving averages help represent short-, medium-, and longer-term price trends.

### Relative Strength Index (RSI)

A **14-day RSI** is calculated using average price gains and losses.

RSI provides information about recent price momentum.

### Bollinger Bands

Bollinger Bands are calculated using the 25-day moving average and standard deviation.

**Upper Band**

`MA25 + (2 × 25-day Standard Deviation)`

**Lower Band**

`MA25 - (2 × 25-day Standard Deviation)`

These features provide information about the position of the stock price relative to recent price volatility.

---

## Machine Learning Model

### Logistic Regression

The project uses **Logistic Regression** as a binary classification model.

The model uses six technical features:

| Feature | Description                    |
| ------- | ------------------------------ |
| MA5     | 5-day Moving Average           |
| MA25    | 25-day Moving Average          |
| MA75    | 75-day Moving Average          |
| RSI     | 14-day Relative Strength Index |
| Upper   | Upper Bollinger Band           |
| Lower   | Lower Bollinger Band           |

---

## Target Variable

The prediction target is created by comparing today's closing price with the next trading day's closing price.

Conceptually:

`Target = 1 → Next closing price > Current closing price`

`Target = 0 → Next closing price ≤ Current closing price`

Therefore, this project predicts **price direction rather than the exact future stock price**.

---

## Prediction

After downloading the data and training the model, the latest technical indicators are passed to the Logistic Regression model.

The application returns one of two predictions:

**Up 📈**

or

**Down 📉**

The purpose is to demonstrate how market data can be transformed into features and used within a machine learning classification pipeline.

---

## Stock Price Visualization

The application also displays approximately five years of historical stock prices.

The chart includes:

* Closing Price
* MA5
* MA25
* MA75

This allows users to visually compare historical price movements with short-, medium-, and longer-term moving averages.

---

## Streamlit Application

The machine learning workflow is deployed as an interactive **Streamlit web application**.

Users can:

1. Enter a stock ticker
2. Retrieve historical data from Yahoo Finance
3. Generate technical indicators
4. Train the Logistic Regression model
5. View the latest model features
6. Generate an Up / Down prediction
7. Explore historical stock prices and moving averages

Both **English and Japanese versions** are available.

---

## Technologies Used

* Python
* Pandas
* NumPy
* yfinance
* Scikit-learn
* Logistic Regression
* Matplotlib
* Streamlit
* GitHub

---

## Key Skills Demonstrated

This project demonstrates practical experience with:

* Financial market data
* Data collection
* Data preprocessing
* Feature engineering
* Technical indicators
* Machine learning classification
* Financial data visualization
* Interactive application development
* Model deployment with Streamlit

---

## Limitations

This project is designed as a machine learning and financial-data analysis demonstration.

The current model has several limitations:

* The model uses historical price-based technical indicators only.
* Fundamental company information is not included.
* Macroeconomic variables are not included.
* News and market sentiment are not considered.
* Transaction costs and market liquidity are not modeled.
* The current implementation does not include out-of-sample model evaluation or backtesting.
* Historical relationships do not guarantee future market performance.

Therefore, the prediction should **not be interpreted as an investment recommendation or trading signal**.

---

## Future Improvements

Potential improvements include:

* Adding chronological train/test evaluation
* Measuring Accuracy, Precision, Recall, F1-score and ROC-AUC
* Implementing walk-forward validation
* Building a backtesting framework
* Comparing Logistic Regression with Random Forest, XGBoost, and other models
* Adding additional technical indicators such as MACD
* Incorporating volume data
* Adding fundamental financial data
* Incorporating market sentiment and news data
* Displaying prediction probabilities
* Improving the Streamlit dashboard and visualization

---

## Conclusion

This project demonstrates how **financial market knowledge, data analysis, machine learning, and application development** can be combined into a complete analytical workflow.

Instead of working only with a static dataset, the application retrieves market data dynamically, performs feature engineering, trains a classification model, generates a prediction, and presents the result through an interactive web application.

The overall workflow is:

**Financial Data → Technical Analysis → Feature Engineering → Machine Learning → Prediction → Web Application**

---

# 🇯🇵 日本語概要

## 株価トレンド予測アプリ

本プロジェクトでは、株価の過去データとテクニカル指標を用いて、**翌営業日の株価方向（上昇・下降）を予測する機械学習Webアプリケーション**を開発しました。

ユーザーが銘柄コードを入力すると、**Yahoo Financeから過去5年間の株価データを取得**し、以下のテクニカル指標を作成します。

* 5日移動平均線（MA5）
* 25日移動平均線（MA25）
* 75日移動平均線（MA75）
* RSI
* ボリンジャーバンド

これらの特徴量を使用して**ロジスティック回帰（Logistic Regression）**モデルを構築し、翌営業日の終値が現在の終値より上昇するか下降するかを分類します。

さらに、Streamlitを使用してWebアプリケーションとして実装し、ユーザーが銘柄を入力してデータ取得、モデル構築、予測、株価チャートの確認まで行えるようにしました。

本プロジェクトを通して、

**金融データ取得 → データ加工 → テクニカル分析 → 特徴量作成 → 機械学習 → 予測 → Webアプリケーション**

という一連のデータ分析・機械学習プロセスを実装しました。

---

## Disclaimer

This project is intended for **educational and portfolio purposes only**.

The predictions generated by the application should not be considered financial advice or investment recommendations.
