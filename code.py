"""
Stock Trend Prediction — next-day direction classifier.

Key design decisions (see README):
  1. Features are scale-invariant ratios, not raw price levels.
  2. Train/test split is chronological, never random.
  3. Model is always reported against a majority-class baseline.
  4. Output is a probability, not just a binary label.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

TEST_SIZE = 0.2          # last 20% of the timeline is held out
LOOKBACK_YEARS = "5y"


# ----------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_prices(ticker: str) -> pd.DataFrame:
    df = yf.download(
        ticker, period=LOOKBACK_YEARS, auto_adjust=True, progress=False
    )
    if df.empty:
        return df
    # yfinance returns a MultiIndex column frame for some calls; flatten it.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


# ----------------------------------------------------------------------
# 2. Feature engineering
# ----------------------------------------------------------------------
def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder's smoothing
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Every feature is a ratio or a bounded index.

    Raw MA5 / MA25 / Upper / Lower are price *levels*: they are non-stationary
    (a stock at 50 USD in 2021 and 200 USD in 2026 is the same signal at a
    different scale) and near-perfectly collinear with each other. Feeding them
    to a linear model teaches it the price level instead of the direction.
    Converting to deviations and ratios removes both problems.
    """
    out = pd.DataFrame(index=df.index)
    close = df["Close"]

    ma5 = close.rolling(5).mean()
    ma25 = close.rolling(25).mean()
    ma75 = close.rolling(75).mean()
    std25 = close.rolling(25).std()

    upper = ma25 + 2 * std25
    lower = ma25 - 2 * std25

    # --- trend: where is price relative to its own averages (in %)
    out["dev_ma5"] = close / ma5 - 1
    out["dev_ma25"] = close / ma25 - 1
    out["dev_ma75"] = close / ma75 - 1

    # --- crossover: short-term vs medium/long-term trend
    out["ma5_vs_ma25"] = ma5 / ma25 - 1
    out["ma25_vs_ma75"] = ma25 / ma75 - 1

    # --- momentum: already bounded 0-100, rescaled to 0-1
    out["rsi14"] = rsi(close, 14) / 100

    # --- volatility position: %B is the standard Bollinger normalisation
    band_width = (upper - lower).replace(0, np.nan)
    out["pct_b"] = (close - lower) / band_width
    out["band_width"] = band_width / ma25          # relative volatility

    # --- short-horizon returns
    out["ret_1d"] = close.pct_change(1)
    out["ret_5d"] = close.pct_change(5)
    out["vol_20d"] = close.pct_change().rolling(20).std()

    return out


def build_target(df: pd.DataFrame) -> pd.Series:
    """1 if the NEXT close is above today's close, else 0."""
    return (df["Close"].shift(-1) > df["Close"]).astype(int)


def assemble(df: pd.DataFrame):
    X = build_features(df)
    y = build_target(df)

    data = X.copy()
    data["target"] = y
    # The final row has no next-day close -> its target is undefined.
    data = data.iloc[:-1].replace([np.inf, -np.inf], np.nan).dropna()

    return data.drop(columns="target"), data["target"], X


# ----------------------------------------------------------------------
# 3. Model + honest evaluation
# ----------------------------------------------------------------------
def make_model() -> Pipeline:
    # Scaling matters: without it, features on different numeric ranges get
    # arbitrarily different effective regularisation.
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ]
    )


def evaluate(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Chronological hold-out. A random split would let the model see the future,
    which is meaningless for a time-ordered series.
    """
    split = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = make_model().fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    # Baseline: always predict whichever class was more common in training.
    # Equity markets rise slightly more often than they fall, so this is
    # typically 52-54% — any model must beat it to be worth anything.
    majority = int(y_train.mode()[0])
    baseline_pred = np.full(len(y_test), majority)

    return {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_start": X_test.index[0].date(),
        "test_end": X_test.index[-1].date(),
        "accuracy": accuracy_score(y_test, pred),
        "baseline_accuracy": accuracy_score(y_test, baseline_pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "confusion": confusion_matrix(y_test, pred),
        "up_rate_test": y_test.mean(),
    }


# ----------------------------------------------------------------------
# 4. Streamlit UI
# ----------------------------------------------------------------------
st.set_page_config(page_title="Stock Trend Prediction", page_icon="📈")
st.title("📈 Stock Trend Prediction")
st.caption(
    "Next-day direction classifier built on technical indicators. "
    "Educational demonstration — not investment advice."
)

ticker = st.text_input("Ticker symbol", value="AAPL").strip().upper()

if not ticker:
    st.stop()

prices = load_prices(ticker)
if prices.empty:
    st.error(f"No data returned for '{ticker}'. Check the symbol and try again.")
    st.stop()

X, y, X_full = assemble(prices)
if len(X) < 250:
    st.error("Not enough history to train and validate a model.")
    st.stop()

metrics = evaluate(X, y)
edge = metrics["accuracy"] - metrics["baseline_accuracy"]

# --- Evaluation first. The honest number goes above the prediction, not below.
st.subheader("Out-of-sample performance")
st.write(
    f"Trained on {metrics['n_train']} days, tested on {metrics['n_test']} "
    f"unseen days ({metrics['test_start']} → {metrics['test_end']}), "
    "split chronologically."
)

c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{metrics['accuracy']:.1%}", f"{edge:+.1%} vs baseline")
c2.metric("Baseline (always 'Up')", f"{metrics['baseline_accuracy']:.1%}")
c3.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

c4, c5, c6 = st.columns(3)
c4.metric("Precision", f"{metrics['precision']:.1%}")
c5.metric("Recall", f"{metrics['recall']:.1%}")
c6.metric("F1", f"{metrics['f1']:.3f}")

if edge <= 0:
    st.warning(
        "This model does not beat the naive baseline on the test period. "
        "Reporting that plainly is part of the analysis: a directional edge "
        "from price-based indicators alone is not something to expect."
    )

with st.expander("Confusion matrix"):
    cm = metrics["confusion"]
    st.dataframe(
        pd.DataFrame(
            cm,
            index=["Actual Down", "Actual Up"],
            columns=["Predicted Down", "Predicted Up"],
        )
    )

# --- Live prediction, refit on all available history
st.subheader("Next-day signal")
final_model = make_model().fit(X, y)

latest = X_full.replace([np.inf, -np.inf], np.nan).dropna().iloc[[-1]]
prob_up = float(final_model.predict_proba(latest[X.columns])[0, 1])

direction = "Up 📈" if prob_up >= 0.5 else "Down 📉"
st.metric(
    f"{ticker} — next trading day",
    direction,
    f"{max(prob_up, 1 - prob_up):.1%} confidence",
)
st.progress(prob_up)
st.caption(f"P(Up) = {prob_up:.3f}. Values near 0.50 carry no meaningful signal.")

# --- Feature weights: which indicators the model actually leans on
with st.expander("Model coefficients"):
    coefs = pd.Series(
        final_model.named_steps["clf"].coef_[0], index=X.columns
    ).sort_values(key=abs, ascending=False)
    st.dataframe(coefs.rename("weight (standardised)").to_frame())

# --- Price chart
st.subheader("Price history")
fig, ax = plt.subplots(figsize=(10, 4))
close = prices["Close"]
ax.plot(close.index, close, label="Close", linewidth=1)
ax.plot(close.index, close.rolling(5).mean(), label="MA5", linewidth=0.8)
ax.plot(close.index, close.rolling(25).mean(), label="MA25", linewidth=0.8)
ax.plot(close.index, close.rolling(75).mean(), label="MA75", linewidth=0.8)
ax.set_ylabel("Price")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
st.pyplot(fig)

st.divider()
st.caption(
    "Limitations: technical indicators only. No fundamentals, macro data, "
    "sentiment, transaction costs, or liquidity modelling. Not a trading signal."
)
