# 📈 Stock Trend Prediction

A next-day price direction classifier built on technical indicators — and, more importantly, an honest evaluation of whether such a classifier works at all.

The application retrieves market data from Yahoo Finance, engineers scale-invariant features, trains a Logistic Regression model with a chronological hold-out, and reports its performance **against a naive baseline** in an interactive Streamlit interface.

**Educational demonstration. Not investment advice.**

## 🌐 Live Demo

**English Version** https://stocktrend-dcc8wybbjikjdlgtg6j7pj.streamlit.app/

**Japanese Version / 日本語版** https://stocktrend-ibgvicgbw3qbljc623ky3e.streamlit.app/

---

## What This Project Actually Tests

Most stock-prediction portfolio projects report an accuracy figure and stop there. That figure is almost always meaningless, for two reasons: it is measured on a randomly shuffled split (which leaks future information into training), and it is never compared against the trivial strategy of predicting the majority class every day.

This project asks a narrower and more answerable question:

> **Do price-derived technical indicators carry enough information to predict next-day direction better than a naive baseline — and does any apparent edge hold up across different markets?**

The answer this implementation arrives at is largely **no**, and reporting that plainly is part of the result.

Testing across eight series also surfaced a second finding that was not the original goal: **the standard baseline comparison is itself exploitable.** When the market direction flips between the training and test windows, comparing against the training-period majority can manufacture an edge of several percentage points out of nothing. One of the eight series does exactly this. Catching it required adding a second, stricter baseline — described below.

---

## Design Decisions

The five choices that shape the implementation:

1. **Features are scale-invariant ratios, not raw price levels.**
2. **The train/test split is chronological, never random.**
3. **Model accuracy is always reported against a baseline — and against a second, stricter one that survives a change of market regime.**
4. **The output is a probability, not just a binary label.**
5. **Threshold analysis asks the real question:** is there a subset of days where the signal is strong enough to be worth acting on?

---

## Pipeline

```
Ticker → Yahoo Finance (5y) → Feature engineering → Chronological split
       → Logistic Regression → Out-of-sample metrics → Threshold analysis
       → Cross-market comparison → Next-day signal → Streamlit
```

---

## Data

Historical OHLCV data is retrieved with **yfinance** using `period="5y"` and `auto_adjust=True`, so prices are already adjusted for splits and dividends. Results are cached for one hour.

---

## Feature Engineering

Raw MA5 / MA25 / MA75 and Bollinger band levels are **price levels**: non-stationary and near-perfectly collinear with each other. Feeding them to a linear model teaches it the price level rather than the direction. Every feature here is therefore a ratio or a bounded index.

| Feature | Definition | What it captures |
| --- | --- | --- |
| `dev_ma5` | `Close / MA5 − 1` | Deviation from the short-term trend |
| `dev_ma25` | `Close / MA25 − 1` | Deviation from the medium-term trend |
| `dev_ma75` | `Close / MA75 − 1` | Deviation from the long-term trend |
| `ma5_vs_ma25` | `MA5 / MA25 − 1` | Short vs medium trend alignment (crossover, continuous) |
| `ma25_vs_ma75` | `MA25 / MA75 − 1` | Medium vs long trend alignment |
| `rsi14` | 14-day RSI ÷ 100 | Momentum, bounded 0–1 |
| `pct_b` | `(Close − Lower) / (Upper − Lower)` | Position within the Bollinger band |
| `band_width` | `(Upper − Lower) / MA25` | Volatility regime |
| `ret_1d` | 1-day return | Short-term reversal / momentum |
| `ret_5d` | 5-day return | Week-scale momentum |
| `vol_20d` | 20-day std of daily returns | Realised volatility |

RSI is computed with Wilder's exponential smoothing (`ewm(alpha=1/14)`) rather than a simple rolling mean.

Bollinger bands use `MA25 ± 2 × 25-day standard deviation`; they enter the model only through `pct_b` and `band_width`.

---

## Target

```
target = 1  if  Close[t+1] > Close[t]
target = 0  otherwise
```

The final row is dropped, since tomorrow's close is unknown for the most recent day. This is the point where next-day leakage would otherwise occur.

---

## Model

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, C=1.0)),
])
```

Logistic Regression is chosen deliberately over a stronger model: its standardised coefficients are directly interpretable and comparable, which makes it a good instrument for establishing whether the *features* carry signal before reaching for model capacity. Scaling sits inside the pipeline so the scaler is fitted on training data only.

---

## Evaluation

The last **20%** of the timeline is held out. A random split would train on days that come after the test days, which inflates accuracy substantially and invalidates the result.

Reported metrics:

- **Accuracy** — shown alongside the baseline, never alone
- **Baseline accuracy** — always predicting the majority class from the training set, labelled with which class that actually was
- **Edge** — accuracy minus baseline
- **Edge vs naive** — accuracy minus the best constant strategy in the test window; the stricter figure, and the one to read when the market regime changed (see below)
- **ROC-AUC** — threshold-independent ranking quality; 0.50 is chance
- **Precision / Recall / F1** — reported for completeness
- **Confusion matrix**

F1 is deliberately flagged as misleading in the interface: because up-days are the majority class, predicting "Up" unconditionally scores higher on F1 than the model does. When the edge is zero or negative, the application says so explicitly rather than burying it.

---

## Threshold Analysis

A classifier does not have to take a position every day. Raising the decision threshold above 0.50 trades coverage for precision: fewer signals, each carrying more conviction.

The application sweeps thresholds from 0.50 to 0.75 and reports, for each:

- **Signals** — number of days the threshold fires
- **Coverage** — share of test days signalled
- **Precision** — hit rate on those days
- **Lift vs base rate** — precision minus the unconditional up-day rate

Rows producing fewer than **20 signals** are marked unreliable and excluded from the "best threshold" selection. Precision computed on a handful of days is sampling noise, not evidence — and omitting this guard is how spurious edges get reported.

---

## Cross-Market Comparison

The identical pipeline is run across six series:

| Ticker | Market |
| --- | --- |
| `AAPL` | Apple (US) |
| `MSFT` | Microsoft (US) |
| `^GSPC` | S&P 500 (US index) |
| `7203.T` | Toyota (JP) |
| `6758.T` | Sony (JP) |
| `^N225` | Nikkei 225 (JP index) |
| `PTT.BK` | PTT (TH) |
| `^SET.BK` | SET Index (TH index) |

This tests whether an apparent edge is a property of the *method* or of one particular series. A method with genuine predictive power would show a consistent edge across markets, not a mix of signs.

---

## Two Baselines, and Why the Second One Matters

The application reports the edge against **two** reference points, because the obvious one turns out to be exploitable.

**Edge** compares the model against the majority class of the *training* period. This is the standard check, and it is the one most projects stop at.

**Edge vs naive** compares against the best constant strategy in the *test* period — always-'Up' or always-'Down', whichever scored higher on those specific days.

When the market direction is stable across both windows, the two figures are identical. When the direction flips between training and test, they diverge, and the first one becomes badly inflated: the model inherits the training-period bias, the baseline it is measured against inherits the same bias, and the model looks good for having been wrong in a slightly less costly way than the reference.

### The clearest example

Running the comparison across all eight series produced this:

| Series | Accuracy | Edge | Edge vs naive | ROC-AUC |
| --- | --- | --- | --- | --- |
| Apple (US) | 54.7% | +0.8% | +0.8% | 0.582 |
| Microsoft (US) | 48.7% | −1.3% | −1.3% | 0.513 |
| S&P 500 (US) | 52.5% | −3.0% | −3.0% | 0.451 |
| Toyota (JP) | 41.3% | −9.1% | −9.1% | 0.427 |
| Sony (JP) | 50.4% | −7.0% | −7.0% | 0.520 |
| Nikkei 225 (JP) | 50.0% | −3.0% | −3.0% | 0.522 |
| PTT (TH) | 60.7% | +1.3% | +1.3% | 0.539 |
| **SET Index (TH)** | **51.6%** | **+8.1%** | **−5.0%** | **0.487** |

Seven of the eight rows agree exactly. The SET Index does not.

Its training window was a falling market, its test window a rising one. Against the training-period majority ('Down') the model appears to gain 8.1% — by a wide margin the strongest result in the table. Against always-'Up', which would have scored 56.6% on those days, the model's 51.6% is 5 points *worse* than doing nothing. Its ROC-AUC of 0.487 — below chance — points the same way.

Measured properly, the mean edge across the eight series is **−3.3%**, positive on **2 of 8**, and both positive cases are around one percentage point on roughly 230 test days, which is inside sampling noise.

The single-ticker view carries the same check: when the model beats the training-period baseline but loses to the best constant strategy in the test window, the interface says so directly rather than displaying the green number alone.

---

## Next-Day Signal

After evaluation, the model is refitted on the full dataset and applied to the most recent row of features. The output is a direction, a confidence figure, and — critically — the raw probability:

```
P(Up) = 0.512   →   values near 0.50 carry no meaningful signal
```

Standardised model coefficients are exposed in an expander, sorted by absolute magnitude, so feature influence can be inspected directly.

---

## Price History

Five years of closing prices with MA5, MA25 and MA75 overlaid.

---

## Tech Stack

Python · pandas · NumPy · yfinance · scikit-learn · Matplotlib · Streamlit

---

## Skills Demonstrated

- Financial time-series data collection and preprocessing
- Feature engineering with attention to stationarity and collinearity
- Leakage-aware experimental design (chronological splits, target alignment)
- Baseline-relative model evaluation, including recognising when a baseline is itself misleading
- Threshold and precision–coverage trade-off analysis
- Robustness testing across independent datasets and market regimes
- Interactive application development and deployment

---

## Limitations

- Price-derived technical indicators only — no fundamentals, macro data, or sentiment
- Transaction costs, slippage and liquidity are not modelled, so reported precision does not translate into returns
- Predicted probabilities are uncalibrated; `P(Up) = 0.62` does not mean a 62% hit rate
- A single chronological hold-out, not walk-forward validation
- No backtest — direction accuracy and profitability are different questions
- Direction only; a 0.1% gain and an 8% gain are the same label

The output is **not a trading signal**.

---

## Future Work

- Walk-forward validation across rolling windows
- Probability calibration (Platt scaling / isotonic regression)
- Longer-horizon targets (5–20 day direction), where the signal-to-noise ratio is more favourable than at 1 day
- Volatility forecasting, which is substantially more predictable than direction
- Volume-based and cross-asset features
- Comparison against Random Forest and gradient-boosted trees
- A cost-aware backtest

---

# 🇯🇵 日本語概要

## 株価トレンド予測アプリ

本プロジェクトは、テクニカル指標を用いて**翌営業日の株価の方向（上昇・下降）を分類する機械学習アプリケーション**です。

ただし、目的は予測を当てることではありません。**「価格由来のテクニカル指標だけで方向を予測できるのか」を正しく検証すること**が目的です。

### 特徴量

移動平均線・RSI・ボリンジャーバンドを使用しますが、**そのままの値は使いません**。価格の水準は非定常であり、移動平均線同士は強く相関するため、線形モデルには不向きです。そのため、すべて比率または0〜1に収まる指標に変換しています（例：`終値 ÷ MA25 − 1`）。全11特徴量。

### 評価設計

- データ分割は**時系列順**（ランダム分割はデータリーケージを起こします）
- 精度は必ず**ベースライン**（毎日「上昇」と答える単純な手法）と比較して表示
- ROC-AUC、混同行列、閾値ごとの適合率も報告
- 有効な予測力がない場合、その旨を明示的に警告として表示

### 閾値分析

毎日ポジションを取る必要はありません。閾値を上げることで、シグナル数と適合率のトレードオフを検証します。シグナルが20件未満の行は、統計的に意味がないため除外しています。

### 市場間比較

同一の手法を日米タイの8銘柄・指数に適用します。本当に予測力があるなら、**一貫してベースラインを上回るはず**です。符号が混在する場合、それは偶然の範囲と判断されます。

### 二つ目のベースライン（本プロジェクトの発見）

検証の過程で、**ベースライン比較そのものが抜け道になり得る**ことが分かりました。

学習期間とテスト期間で相場の方向が反転すると、学習期間の多数派との比較では、実力がなくても数％の「優位」が生まれます。タイのSET指数がその例です。

- 学習期間の多数派（下降）との比較：**+8.1%** — 8銘柄中で最も良い数字
- テスト期間で最良の単純戦略（毎日「上昇」＝56.6%）との比較：**−5.0%**
- ROC-AUCは0.487で、ランダム以下

そこで、テスト期間における最良の単純戦略との比較も併記するようにしました。この基準で測ると、8銘柄の平均は**−3.3%**、上回ったのは**2銘柄のみ**です。

### 結論

本プロジェクトで示したのはモデルの性能ではなく、**検証の設計そのもの**です。効果がなかったことをそのまま報告できること、そして自分の評価基準自体を疑うことも、分析の重要な一部だと考えています。

---

## Disclaimer

For educational and portfolio purposes only. Nothing here constitutes financial advice or an investment recommendation.
