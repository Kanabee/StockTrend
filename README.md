

# 📈 Stock Trend Prediction

A deployed machine-learning web application that classifies next-day stock
price direction from technical indicators — and reports honestly that it does
not work.

**Live demo:** [English] https://stocktrend-dcc8wybbjikjdlgtg6j7pj.streamlit.app/
---

## Result First

Tested on AAPL over 236 unseen trading days (2025-09-26 → 2026-09-03), trained
on the preceding 944 days:

| Metric | Model | Baseline |
|---|---|---|
| Accuracy | **54.7%** | 53.8% (always predict "Up") |
| ROC-AUC | **0.582** | 0.500 |
| Precision | 56.1% | — |
| Recall | 72.4% | — |
| F1 | 0.632 | 0.700 |

**The model carries no economically meaningful edge.** It beats the naive
baseline by 0.9 percentage points — about two days out of 236, which is inside
the range of noise. Precision of 56.1% against recall of 72.4% shows a
systematic bias toward predicting "Up": the model calls up on 164 of 236 days
when only 127 actually rise. Acting on that would mean 72 losing entries, and
transaction costs alone would erase the margin.

ROC-AUC of 0.582 is the one figure above chance that means something: the model
ranks up-days above down-days slightly better than random. Whether that
ranking is exploitable at any decision threshold is examined below.

Note that **F1 is lower than the naive baseline would score.** Predicting "Up"
every single day yields F1 = 0.700 against the model's 0.632, because the
positive class is the majority. F1 is reported here only to show why it is the
wrong headline metric for this problem.

---

## Why This Is the Expected Result

If daily price direction were predictable from moving averages and RSI alone,
the edge would have been arbitraged away long ago. A model reporting 90%+
accuracy on this task is almost always leaking future information — commonly by
using a random train/test split on time-ordered data, or by including a feature
unavailable at prediction time.

This project is built to make that failure mode impossible to hide, and to
report what remains.

---

## Design Decisions

### Scale-invariant features

The original implementation fed raw MA5, MA25, MA75, and Bollinger band levels
directly into the classifier. Those are **price levels**: non-stationary (a
stock at 50 USD in 2021 and 200 USD in 2026 is the same signal at a different
scale) and near-perfectly collinear with one another. A linear model given
those inputs learns the price level, not the direction.

Every feature was rebuilt as a ratio or a bounded index:

| Feature | Definition | Captures |
|---|---|---|
| `dev_ma5/25/75` | `Close / MA − 1` | Position relative to trend, in % |
| `ma5_vs_ma25`, `ma25_vs_ma75` | `MA_short / MA_long − 1` | Crossover signal |
| `rsi14` | 14-day RSI, Wilder's smoothing | Momentum, bounded 0–1 |
| `pct_b` | `(Close − Lower) / (Upper − Lower)` | Standard Bollinger %B |
| `band_width` | `(Upper − Lower) / MA25` | Relative volatility |
| `ret_1d`, `ret_5d` | Simple returns | Short-horizon momentum |
| `vol_20d` | Rolling σ of returns | Realized volatility |

Features are standardised inside a scikit-learn `Pipeline`, so scaling is
fitted on training data only and never sees the test set.

### Chronological split

The last 20% of the timeline is held out. A random split on time-ordered data
lets the model train on days that follow the days it is tested on — the single
most common source of inflated results in this kind of project.

### Baseline comparison

Equity markets rise slightly more often than they fall, so a classifier that
always predicts "Up" scores 52–54% on most series. Every accuracy figure in
this project is reported against that baseline, because accuracy alone is
meaningless here.

### Probability, not a label

The app surfaces P(Up) rather than a binary call, so a reader can see when the
model is at 0.53 and carrying no conviction.

---

## Signal Threshold Analysis

A classifier does not have to trade every day. Raising the decision threshold
trades coverage for precision: fewer signals, each carrying more conviction.
The app sweeps thresholds from 0.50 to 0.75 and reports precision, coverage,
and lift over the base rate at each.

Thresholds producing fewer than 20 signals are excluded from the summary.
Precision computed on a handful of days is sampling noise, and picking the
threshold with the prettiest number is how backtests get overfitted.

*[Insert your threshold table here after running the app — the question is
whether precision rises meaningfully above the base rate while enough signals
remain.]*

---

## Cross-Market Comparison

The same pipeline is run across US and Japanese equities and indices — AAPL,
MSFT, S&P 500, Toyota (7203.T), Sony (6758.T), Nikkei 225 — to test whether
any apparent edge belongs to the method or to one particular series.

A method with genuine predictive power shows a consistent edge across markets.
A mix of positive and negative edges indicates noise.

*[Insert your comparison table here after running it.]*

---

## How It Works

```
Ticker → yfinance API → 5 years of daily data → scale-invariant features
    → chronological split → Logistic Regression → out-of-sample evaluation
    → threshold analysis → P(Up) → Streamlit
```

The model is retrained per session on the ticker the user enters, so the
application demonstrates the full pipeline rather than serving a pre-fitted
artefact.

---

## Limitations

- **Technical indicators only.** No fundamentals, macroeconomic variables,
  news, or sentiment.
- **No transaction costs or liquidity modelling.** Any apparent edge would need
  to survive spreads, commissions and slippage before meaning anything.
- **Single chronological split**, not walk-forward validation. The result
  reflects one test period.
- **Logistic Regression only.** Tree-based and gradient-boosted models were not
  compared.
- **Historical relationships do not guarantee future performance.**

Predictions from this application are **not investment advice or trading
signals**. The project exists to demonstrate an analytical pipeline and honest
model evaluation.

---

## Next Steps

- Walk-forward validation across multiple non-overlapping test windows.
- A backtest incorporating transaction costs, to convert accuracy into
  something with an economic interpretation.
- Comparison against Random Forest and gradient boosting.
- Additional features: volume, MACD, sector-relative returns.
- Probability calibration, so a stated 0.60 corresponds to a 60% outcome rate.

---

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · yfinance · Matplotlib · Streamlit

## Running Locally

```bash
pip install -r requirements.txt
streamlit run code.py
```

---

# 🇯🇵 日本語概要

## 株価トレンド予測 Web アプリケーション

テクニカル指標を用いて翌営業日の株価方向を分類する機械学習アプリケーションを
開発し、日英2言語版としてデプロイしました。

### 結論：本モデルに実務上有意な予測力はありません

AAPL を対象に、学習944日・検証236日（2025-09-26 〜 2026-09-03）の時系列順分割
で評価した結果は以下の通りです。

| 指標 | モデル | ベースライン |
|---|---|---|
| 正解率 | **54.7%** | 53.8%（常に「上昇」と予測） |
| ROC-AUC | **0.582** | 0.500 |
| 適合率 | 56.1% | — |
| 再現率 | 72.4% | — |

ベースラインとの差は 0.9 ポイント、236日中およそ2日分に相当し、誤差の範囲内
です。また、適合率56.1%に対し再現率72.4%という結果は、モデルが「上昇」側に
偏った予測をしていることを示します。実際には127日しか上昇していないにもかかわ
らず164日を上昇と予測しており、72回の誤ったエントリーは取引コストだけで利益を
消失させます。

**この結果は想定通りです。** 移動平均やRSIのみから日次の値動きが予測できるので
あれば、その優位性はとうに市場で解消されているはずです。正解率90%超を報告する
モデルは、多くの場合リークを含んでいます。本プロジェクトは、そのリークが起こり
得ない構造を作った上で、残った結果をそのまま報告することを目的としています。

### 手法上の工夫

**1. スケール不変な特徴量への変換**
当初の実装では移動平均やボリンジャーバンドを価格の水準のまま特徴量としていま
した。これらは非定常であり（2021年の50ドルと2026年の200ドルは同じシグナルの
異なるスケール）、相互に強く相関するため、線形モデルは「方向」ではなく「価格
水準」を学習してしまいます。そこで、移動平均乖離率、%B、バンド幅比率、実現
ボラティリティなど、すべて比率または有界指標に置き換えました。

**2. 時系列順の分割**
ランダム分割では、テスト対象日より後の日を学習に使ってしまいます。これはこの種
のプロジェクトで結果が過大評価される最大の原因であり、直近20%を時系列順に
ホールドアウトすることで回避しています。

**3. ベースラインとの比較を必須化**
株式市場は下落より上昇の日がやや多いため、「常に上昇」と予測するだけで52〜54%
の正解率が得られます。この比較なしに正解率を報告することに意味はありません。

**4. しきい値分析**
分類器は毎日取引する必要はありません。判定しきい値を上げれば、シグナル数と引き
換えに確信度の高い日だけを選別できます。ただし、シグナル数が20件未満のしきい値
は集計から除外しています。数日分で算出した適合率は標本誤差であり、最も見栄えの
良いしきい値を選ぶことがバックテストの過剰適合そのものだからです。

**5. 二値ラベルではなく確率を出力**
P(Up) を表示することで、モデルが0.53という「確信のない状態」にあることを利用者
が判断できるようにしました。

### 限界

テクニカル指標のみを使用しており、ファンダメンタルズ、マクロ経済指標、ニュース、
市場センチメントは含んでいません。取引コストおよび流動性も考慮していません。
検証は単一期間の時系列分割によるものであり、ウォークフォワード検証は未実施です。

本アプリケーションの予測は投資判断のためのシグナルではなく、分析プロセスとモデル
評価手法の実証を目的としたものです。
