"""
Stock Trend Prediction — next-day direction classifier.

Key design decisions (see README):
  1. Features are scale-invariant ratios, not raw price levels.
  2. Train/test split is chronological, never random.
  3. Model is always reported against a majority-class baseline.
  4. Output is a probability, not just a binary label.
  5. Threshold analysis asks the real question: is there a subset of days
     where the signal is strong enough to be worth acting on?
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
MIN_SIGNALS = 20         # below this, a precision figure is not interpretable

COMPARISON_TICKERS = {
    "AAPL": "Apple (US)",
    "MSFT": "Microsoft (US)",
    "^GSPC": "S&P 500 (US index)",
    "7203.T": "Toyota (JP)",
    "6758.T": "Sony (JP)",
    "^N225": "Nikkei 225 (JP index)",
    "PTT.BK": "PTT (TH)",
    "^SET.BK": "SET Index (TH index)",
}


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
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Every feature is a ratio or a bounded index.

    Raw MA5 / MA25 / Upper / Lower are price *levels*: non-stationary and
    near-perfectly collinear. Feeding them to a linear model teaches it the
    price level instead of the direction. Ratios remove both problems.
    """
    out = pd.DataFrame(index=df.index)
    close = df["Close"]

    ma5 = close.rolling(5).mean()
    ma25 = close.rolling(25).mean()
    ma75 = close.rolling(75).mean()
    std25 = close.rolling(25).std()

    upper = ma25 + 2 * std25
    lower = ma25 - 2 * std25

    out["dev_ma5"] = close / ma5 - 1
    out["dev_ma25"] = close / ma25 - 1
    out["dev_ma75"] = close / ma75 - 1

    out["ma5_vs_ma25"] = ma5 / ma25 - 1
    out["ma25_vs_ma75"] = ma25 / ma75 - 1

    out["rsi14"] = rsi(close, 14) / 100

    band_width = (upper - lower).replace(0, np.nan)
    out["pct_b"] = (close - lower) / band_width
    out["band_width"] = band_width / ma25

    out["ret_1d"] = close.pct_change(1)
    out["ret_5d"] = close.pct_change(5)
    out["vol_20d"] = close.pct_change().rolling(20).std()

    return out


def build_target(df: pd.DataFrame) -> pd.Series:
    return (df["Close"].shift(-1) > df["Close"]).astype(int)


def assemble(df: pd.DataFrame):
    X = build_features(df)
    y = build_target(df)

    data = X.copy()
    data["target"] = y
    data = data.iloc[:-1].replace([np.inf, -np.inf], np.nan).dropna()

    return data.drop(columns="target"), data["target"], X


# ----------------------------------------------------------------------
# 3. Model + evaluation
# ----------------------------------------------------------------------
def make_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ]
    )


def evaluate(X: pd.DataFrame, y: pd.Series) -> dict:
    """Chronological hold-out. A random split would leak the future."""
    split = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = make_model().fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    majority = int(y_train.mode()[0])
    baseline_pred = np.full(len(y_test), majority)

    return {
         
        "majority": majority,
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
        "up_rate_test": float(y_test.mean()),
        "y_test": y_test,
        "proba": proba,
    }


# ----------------------------------------------------------------------
# 4. Threshold analysis
# ----------------------------------------------------------------------
def threshold_table(
    y_test: pd.Series, proba: np.ndarray, base_rate: float
) -> pd.DataFrame:
    """
    A classifier does not have to trade every day. Raising the decision
    threshold trades coverage for precision: fewer signals, each carrying
    more conviction. This table shows whether that trade is available at
    all — for price-only models it often is not.
    """
    rows = []
    y = y_test.to_numpy()

    for t in np.arange(0.50, 0.76, 0.025):
        mask = proba >= t
        n = int(mask.sum())
        precision = float(y[mask].mean()) if n else np.nan
        rows.append(
            {
                "Threshold": round(float(t), 3),
                "Signals": n,
                "Coverage": n / len(y),
                "Precision": precision,
                "Lift vs base rate": (
                    precision - base_rate if n >= MIN_SIGNALS else np.nan
                ),
                "Reliable": n >= MIN_SIGNALS,
            }
        )

    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def evaluate_ticker(ticker: str):
    """Compact evaluation used by the cross-market comparison."""
    prices = load_prices(ticker)
    if prices.empty:
        return None
    X, y, _ = assemble(prices)
    if len(X) < 250:
        return None
    m = evaluate(X, y)
    
    best_naive = max(m["up_rate_test"], 1 - m["up_rate_test"])
    return {
        "Accuracy": m["accuracy"],
        "Baseline": m["baseline_accuracy"],
        "Edge": m["accuracy"] - m["baseline_accuracy"],
        "Best naive": "{:.1%}",
        "Edge vs naive": "{:+.1%}",
        "ROC-AUC": m["roc_auc"],
        "Up rate": m["up_rate_test"],
        "Test days": m["n_test"],
    }

# ----------------------------------------------------------------------
# 5. Streamlit UI
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

# --- Evaluation first. The honest number goes above the prediction.
st.subheader("Out-of-sample performance")
st.write(
    f"Trained on {metrics['n_train']} days, tested on {metrics['n_test']} "
    f"unseen days ({metrics['test_start']} → {metrics['test_end']}), "
    "split chronologically."
)

c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{metrics['accuracy']:.1%}", f"{edge:+.1%} vs baseline")
majority_label = "Up" if metrics["majority"] == 1 else "Down"
always_up = metrics["up_rate_test"]
c2.metric(
    f"Baseline (always '{majority_label}')",
    f"{metrics['baseline_accuracy']:.1%}",
    help=(
        f"'{majority_label}' was the majority class in the training period. "
        f"For reference, always-'Up' would score {always_up:.1%} on this "
        f"test window, always-'Down' {1 - always_up:.1%}."
    ),
)
c3.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

c4, c5, c6 = st.columns(3)
c4.metric("Precision", f"{metrics['precision']:.1%}")
c5.metric("Recall", f"{metrics['recall']:.1%}")
c6.metric("F1", f"{metrics['f1']:.3f}")

pos_is_majority = always_up >= 0.5
st.caption(
    "F1 is shown for completeness but is hard to read in isolation: 'Up' is "
    f"the {'majority' if pos_is_majority else 'minority'} class here "
    f"({always_up:.1%} of test days), so a constant prediction can outscore "
    "the model on F1 without carrying any information. ROC-AUC and the edge "
    "over baseline are the figures to read."
)

best_naive = max(always_up, 1 - always_up)
naive_label = "Up" if always_up >= 0.5 else "Down"

if edge <= 0:
    st.warning(
        "This model does not beat the naive baseline on the test period. "
        "Reporting that plainly is part of the analysis: a directional edge "
        "from price-based indicators alone is not something to expect."
    )
elif metrics["accuracy"] < best_naive:
    st.warning(
        f"The positive edge above is measured against the **training-period** "
        f"majority ('{majority_label}'). The direction flipped in the test "
        f"window: always-'{naive_label}' would have scored {best_naive:.1%}, "
        f"above the model's {metrics['accuracy']:.1%}. The apparent edge "
        "reflects a regime change, not predictive skill."
    )

with st.expander("Confusion matrix"):
    st.dataframe(
        pd.DataFrame(
            metrics["confusion"],
            index=["Actual Down", "Actual Up"],
            columns=["Predicted Down", "Predicted Up"],
        )
    )

# --- Threshold analysis
st.subheader("Signal threshold analysis")
base_rate = metrics["up_rate_test"]
tbl = threshold_table(metrics["y_test"], metrics["proba"], base_rate)

st.write(
    "The default 0.50 cut-off forces a call on every trading day. Raising it "
    "means acting only on high-conviction days. The question is whether "
    f"precision rises meaningfully above the {base_rate:.1%} base rate of "
    "up-days while enough signals remain to be meaningful."
)

st.dataframe(
    tbl.style.format(
        {
            "Coverage": "{:.1%}",
            "Precision": "{:.1%}",
            "Lift vs base rate": "{:+.1%}",
        },
        na_rep="—",
    ),
    hide_index=True,
    use_container_width=True,
)

reliable = tbl[tbl["Reliable"]].dropna(subset=["Precision"])
if not reliable.empty:
    fig_t, ax_t = plt.subplots(figsize=(9, 4))
    ax_t.plot(
        reliable["Threshold"],
        reliable["Precision"],
        marker="o",
        label="Precision",
    )
    ax_t.axhline(
        base_rate,
        linestyle="--",
        linewidth=1,
        color="grey",
        label=f"Base rate ({base_rate:.1%})",
    )
    ax_t.set_xlabel("Decision threshold — P(Up)")
    ax_t.set_ylabel("Precision")
    ax_t.grid(alpha=0.3)
    ax_t.legend(loc="upper left")

    ax_c = ax_t.twinx()
    ax_c.bar(
        reliable["Threshold"],
        reliable["Coverage"],
        width=0.015,
        alpha=0.2,
        color="tab:orange",
    )
    ax_c.set_ylabel("Coverage (share of days signalled)")
    ax_c.set_ylim(0, 1)

    st.pyplot(fig_t)

    best = reliable.loc[reliable["Precision"].idxmax()]
    st.info(
        f"Best reliable threshold: **{best['Threshold']:.3f}** — "
        f"{int(best['Signals'])} signals ({best['Coverage']:.1%} of test days), "
        f"precision {best['Precision']:.1%} "
        f"({best['Lift vs base rate']:+.1%} vs base rate). Rows with fewer than "
        f"{MIN_SIGNALS} signals are excluded: precision computed on a handful "
        "of days is sampling noise, not evidence."
    )
else:
    st.info(
        f"No threshold produced at least {MIN_SIGNALS} signals, so no precision "
        "figure here would be statistically meaningful."
    )

# --- Cross-market comparison
st.subheader("Cross-market comparison")
st.write(
    "Running the identical pipeline across US and Japanese equities and indices "
    "tests whether any apparent edge is a property of the method or of one "
    "particular series."
)

if st.button("Run comparison (takes ~30 seconds)"):
    results = {}
    progress = st.progress(0.0)
    for i, (tk, label) in enumerate(COMPARISON_TICKERS.items(), start=1):
        res = evaluate_ticker(tk)
        if res:
            results[f"{label}  [{tk}]"] = res
        progress.progress(i / len(COMPARISON_TICKERS))
    progress.empty()

    if results:
        comp = pd.DataFrame(results).T
        st.dataframe(
            comp.style.format(
                {
                    "Accuracy": "{:.1%}",
                    "Baseline": "{:.1%}",
                    "Edge": "{:+.1%}",
                    "ROC-AUC": "{:.3f}",
                    "Up rate": "{:.1%}",
                    "Test days": "{:.0f}",
                }
            ),
            use_container_width=True,
        )
            mean_edge = comp["Edge vs naive"].mean()
            n_positive = int((comp["Edge vs naive"] > 0).sum())
            st.caption(
                f"Measured against the best constant strategy in each test "
                f"window, the mean edge across {len(comp)}series is "
                f"{mean_edge:+.1%}, positive on {n_positive} of {len(comp)}. "
                "The 'Edge' column compares against the training-period "
                "majority instead, and can be inflated when the direction "
                "flips between periods."
            )   
            
        
    else:
        st.warning("No comparison data could be retrieved.")

# --- Live prediction
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