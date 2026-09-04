
# 📈 Stock Trend Prediction

A deployed machine-learning web application that classifies next-day stock
price direction from technical indicators — and reports honestly that it does
not work.

**Live demo:** [English](https://stocktrend-dcc8wybbjikjdlgtg6j7pj.streamlit.app/) 
            [Japan](https://stocktrend-rwojp3jczsj3plndxggegq.streamlit.app/)
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

ROC-AUC of 0.582 is the one figure above chance that means something on this
ticker: the model ranks up-days above down-days slightly better than random.
Restricting to high-conviction days lifts precision to 62.3% at a 0.525
threshold, 8.5 points above the base rate.

**That apparent signal does not survive contact with other markets.** Run
across six US and Japanese series, mean ROC-AUC is **0.5015** — chance — and
the edge over baseline is negative on five of six. Apple's result sits 1.5
standard deviations above a distribution centred on nothing. See the
cross-market comparison below.

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

| Threshold | Signals | Coverage | Precision | Lift vs base rate | 95% CI |
|---|---|---|---|---|---|
| 0.500 | 164 | 69.5% | 56.1% | +2.3% | [48.5%, 63.7%] |
| 0.525 | 106 | 44.9% | **62.3%** | **+8.5%** | [53.1%, 71.5%] |
| 0.550 | 40 | 16.9% | **65.0%** | **+11.2%** | [50.2%, 79.8%] |
| 0.575 | 12 | 5.1% | 75.0% | — | *excluded, n < 20* |
| 0.600 | 3 | 1.3% | 66.7% | — | *excluded, n < 20* |
| 0.625 | 2 | 0.8% | 100.0% | — | *excluded, n < 20* |

Base rate: 53.8% of test days rose (127 of 236).

**Precision rises monotonically with conviction.** At the default 0.50
cut-off the model is indistinguishable from the baseline. Restricting to days
where it assigns P(Up) ≥ 0.525 leaves 106 signals at 62.3% precision — 8.5
points above base rate. At 0.550, 40 signals at 65.0%.

This is consistent with the ROC-AUC of 0.582: the model ranks days better than
it classifies them. The signal exists in the ordering and the default
threshold discards it.

### How much weight this deserves

Two things stop this from being a finding rather than a hint.

**Sample size.** At threshold 0.525, precision is 62.3% ± 9.2 points at 95%
confidence — a one-sided z of 1.80, p = 0.036. Suggestive on its own.

**Selection.** Eleven thresholds were swept and the best chosen after seeing
the test set. Adjusting for that (Bonferroni, ×11) takes p to 0.39. The result
does not survive a correction for having looked eleven times.

**The 0.625 row is the argument for the exclusion rule.** It shows 100%
precision — on two days. Without a minimum-signal filter, that row is what a
careless write-up would headline.

The honest summary: a conviction filter appears to recover a real ranking
signal, but the evidence is one ticker over one test period and does not clear
a multiple-comparison correction. Confirming it requires walk-forward
validation across several markets and periods, and a backtest with transaction
costs — 40 trades at 65% precision says nothing about profit until the size of
the winning and losing moves is known.

---

## Cross-Market Comparison

The same pipeline is run across US and Japanese equities and indices — AAPL,
MSFT, S&P 500, Toyota (7203.T), Sony (6758.T), Nikkei 225 — to test whether
any apparent edge belongs to the method or to one particular series.

A method with genuine predictive power shows a consistent edge across markets.
A mix of positive and negative edges indicates noise.

| Series | Accuracy | Baseline | Edge | ROC-AUC | Test days |
|---|---|---|---|---|---|
| Apple (US) | 54.7% | 53.8% | **+0.8%** | 0.582 | 236 |
| Microsoft (US) | 48.7% | 50.0% | −1.3% | 0.513 | 236 |
| S&P 500 (US) | 52.5% | 55.5% | −3.0% | 0.451 | 236 |
| Toyota (JP) | 41.7% | 50.0% | −8.3% | 0.429 | 230 |
| Sony (JP) | 50.4% | 57.0% | −6.5% | 0.515 | 230 |
| Nikkei 225 (JP) | 49.1% | 53.0% | −3.9% | 0.519 | 230 |

**Mean edge: −3.7%. Positive on 1 of 6.**

**Mean ROC-AUC across the six series: 0.5015** — indistinguishable from chance
(sd 0.055, t = 0.07). Individual values scatter symmetrically around 0.50:
Apple sits 0.082 above, Toyota 0.071 below.

### This settles the question

Apple's 0.582 is **1.5 standard deviations above the cross-market mean** — the
top of a distribution centred on chance, not evidence of a method that works.
The threshold result above is best read the same way: it was the best of eleven
thresholds on the best of six series.

The method does not generalise. Applied to five other liquid, well-covered
markets, it loses to a naive baseline every time, and loses most on the
Japanese equities. Whatever pattern the model found in Apple's 2025–2026 price
history is a property of that series and that window.

**This was the point of running the comparison.** A single-ticker result with a
positive-looking number is the easiest way to fool yourself in this kind of
work, and the cheapest correction is to run the identical pipeline elsewhere
before believing it.

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
分類器は毎日取引する必要はありません。判定しきい値を上げれば、シグナル数と
引き換えに確信度の高い日だけを選別できます。

| しきい値 | シグナル数 | 適合率 | ベースライン比 |
|---|---|---|---|
| 0.500 | 164 | 56.1% | +2.3% |
| 0.525 | 106 | **62.3%** | **+8.5%** |
| 0.550 | 40 | **65.0%** | **+11.2%** |
| 0.625 | 2 | 100.0% | 除外（n<20） |

確信度を上げるにつれて適合率が単調に上昇しており、これは ROC-AUC 0.582 と整合
します。つまりモデルは「分類」よりも「順位付け」に情報を持っており、既定の
しきい値0.50がその情報を捨てていたことになります。

**ただし、この結果は「発見」ではなく「示唆」に留めるべきです。** しきい値0.525
における適合率62.3%は95%信頼区間で ±9.2ポイント（片側 z = 1.80、p = 0.036）
であり、単独では有意水準を満たすものの、11通りのしきい値を検証した上で最良の
ものを事後的に選択しているため、多重比較補正（Bonferroni ×11）を行うと
p = 0.39 となり有意性は失われます。

なお、しきい値0.625の行は適合率100%を示していますが、これはわずか2日分の結果
です。シグナル数20件未満を集計から除外している理由がここにあります。この種の
数値を成果として提示することが、バックテストにおける過剰適合そのものです。

**5. 複数市場での再現性検証**
同一のパイプラインを米国・日本の6銘柄／指数に適用し、AAPLで見られた優位性が
「手法の性質」か「特定銘柄の性質」かを検証しました。

| 銘柄 | 正解率 | ベースライン | 差 | ROC-AUC |
|---|---|---|---|---|
| Apple (US) | 54.7% | 53.8% | **+0.8%** | 0.582 |
| Microsoft (US) | 48.7% | 50.0% | −1.3% | 0.513 |
| S&P 500 | 52.5% | 55.5% | −3.0% | 0.451 |
| トヨタ (JP) | 41.7% | 50.0% | −8.3% | 0.429 |
| ソニー (JP) | 50.4% | 57.0% | −6.5% | 0.515 |
| 日経225 | 49.1% | 53.0% | −3.9% | 0.519 |

**平均 ROC-AUC は 0.5015 であり、偶然と区別できません**（標準偏差0.055）。
各銘柄の値は0.50を中心にほぼ対称に分布しており（AAPLは+0.082、トヨタは
−0.071）、ベースラインを上回ったのは6銘柄中1銘柄のみ、平均では −3.7% でした。

**この検証により結論が確定します。** AAPLの ROC-AUC 0.582 は、平均から標準
偏差1.5個分上振れした値に過ぎず、手法に予測力があることを示すものではありま
せん。前項のしきい値分析の結果も同様に、「6銘柄中で最も良かった銘柄における、
11通り中で最も良かったしきい値」であったと解釈するのが妥当です。

単一銘柄の良好な結果をそのまま受け入れることが、この種の分析における最大の
落とし穴です。同一パイプラインを他市場に適用するという最も安価な検証を行った
上で結論を出す設計としました。

**6. 二値ラベルではなく確率を出力**
P(Up) を表示することで、モデルが0.53という「確信のない状態」にあることを利用者
が判断できるようにしました。

### 限界

テクニカル指標のみを使用しており、ファンダメンタルズ、マクロ経済指標、ニュース、
市場センチメントは含んでいません。取引コストおよび流動性も考慮していません。
検証は単一期間の時系列分割によるものであり、ウォークフォワード検証は未実施です。

本アプリケーションの予測は投資判断のためのシグナルではなく、分析プロセスとモデル
評価手法の実証を目的としたものです。
