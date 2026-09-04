"""
株価トレンド予測 — 翌営業日の値動き方向を分類するモデル

設計上の主要な判断（詳細はREADME参照）:
  1. 特徴量は価格水準ではなく、スケールに依存しない比率を使用する。
  2. 学習・検証の分割は必ず時系列順で行う。ランダム分割は行わない。
  3. 精度は常に多数派クラス（ベースライン）と比較して報告する。
  4. 出力は二値ラベルではなく確率とする。
  5. しきい値分析により「行動する価値のあるシグナルが存在する日はあるか」を検証する。
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

TEST_SIZE = 0.2          # 直近20%を検証用にホールドアウト
LOOKBACK_YEARS = "5y"
MIN_SIGNALS = 20         # これ未満のシグナル数では適合率を解釈できない

COMPARISON_TICKERS = {
    "AAPL": "アップル（米国）",
    "MSFT": "マイクロソフト（米国）",
    "^GSPC": "S&P500（米国指数）",
    "7203.T": "トヨタ自動車（日本）",
    "6758.T": "ソニーグループ（日本）",
    "^N225": "日経225（日本指数）",
}


# ----------------------------------------------------------------------
# 1. データ取得
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
# 2. 特徴量エンジニアリング
# ----------------------------------------------------------------------
def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """RSI（Wilderの平滑化による標準的な算出方法）"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    すべての特徴量を比率または有界指標として設計する。

    移動平均（MA5/MA25/MA75）やボリンジャーバンドの上下限をそのまま用いると、
    これらは「価格の水準」であるため非定常であり（2021年の50ドルと2026年の
    200ドルは同じシグナルの異なるスケールに過ぎない）、かつ相互に極めて強く
    相関する。この状態で線形モデルに投入すると、モデルは値動きの「方向」では
    なく「価格水準」を学習してしまう。比率化により両方の問題を回避する。
    """
    out = pd.DataFrame(index=df.index)
    close = df["Close"]

    ma5 = close.rolling(5).mean()
    ma25 = close.rolling(25).mean()
    ma75 = close.rolling(75).mean()
    std25 = close.rolling(25).std()

    upper = ma25 + 2 * std25
    lower = ma25 - 2 * std25

    # トレンド：終値が各移動平均からどれだけ乖離しているか（％）
    out["dev_ma5"] = close / ma5 - 1
    out["dev_ma25"] = close / ma25 - 1
    out["dev_ma75"] = close / ma75 - 1

    # クロスオーバー：短期トレンドと中長期トレンドの関係
    out["ma5_vs_ma25"] = ma5 / ma25 - 1
    out["ma25_vs_ma75"] = ma25 / ma75 - 1

    # モメンタム：RSIは元々0〜100の有界指標であるため0〜1に変換のみ
    out["rsi14"] = rsi(close, 14) / 100

    # ボラティリティ内での位置：%Bはボリンジャーバンドの標準的な正規化手法
    band_width = (upper - lower).replace(0, np.nan)
    out["pct_b"] = (close - lower) / band_width
    out["band_width"] = band_width / ma25          # 相対的なボラティリティ

    # 短期リターンと実現ボラティリティ
    out["ret_1d"] = close.pct_change(1)
    out["ret_5d"] = close.pct_change(5)
    out["vol_20d"] = close.pct_change().rolling(20).std()

    return out


def build_target(df: pd.DataFrame) -> pd.Series:
    """翌営業日の終値が当日終値を上回れば1、それ以外は0"""
    return (df["Close"].shift(-1) > df["Close"]).astype(int)


def assemble(df: pd.DataFrame):
    X = build_features(df)
    y = build_target(df)

    data = X.copy()
    data["target"] = y
    # 最終行は「翌日の終値」が存在しないため目的変数が定義できない
    data = data.iloc[:-1].replace([np.inf, -np.inf], np.nan).dropna()

    return data.drop(columns="target"), data["target"], X


# ----------------------------------------------------------------------
# 3. モデル構築と評価
# ----------------------------------------------------------------------
def make_model() -> Pipeline:
    # スケーリングは必須。特徴量ごとに数値の範囲が異なると、正則化の効き方が
    # 特徴量によって不均一になってしまう。
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ]
    )


def evaluate(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    時系列順のホールドアウト検証。

    ランダム分割を行うと、検証対象日より後の日付を学習に使用してしまう。
    これはこの種の分析で結果が過大評価される最大の原因である。
    """
    split = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = make_model().fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    # ベースライン：学習データで多数派だったクラスを常に予測する単純な戦略。
    # 株式市場は下落より上昇の日がやや多いため、通常52〜54%程度になる。
    # モデルはこれを上回らなければ意味がない。
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
        "up_rate_test": float(y_test.mean()),
        "y_test": y_test,
        "proba": proba,
    }


# ----------------------------------------------------------------------
# 4. しきい値分析
# ----------------------------------------------------------------------
def threshold_table(
    y_test: pd.Series, proba: np.ndarray, base_rate: float
) -> pd.DataFrame:
    """
    分類器は毎日取引する必要はない。判定しきい値を上げれば、シグナル数と
    引き換えに確信度の高い日だけを選別できる。そのトレードオフが実際に
    成立するかを検証する。価格情報のみのモデルでは成立しないことも多い。
    """
    rows = []
    y = y_test.to_numpy()

    for t in np.arange(0.50, 0.76, 0.025):
        mask = proba >= t
        n = int(mask.sum())
        precision = float(y[mask].mean()) if n else np.nan
        rows.append(
            {
                "しきい値": round(float(t), 3),
                "シグナル数": n,
                "対象日割合": n / len(y),
                "適合率": precision,
                "ベースライン比": (
                    precision - base_rate if n >= MIN_SIGNALS else np.nan
                ),
                "有効": n >= MIN_SIGNALS,
            }
        )

    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def evaluate_ticker(ticker: str):
    """複数市場比較で使用する簡易評価"""
    prices = load_prices(ticker)
    if prices.empty:
        return None
    X, y, _ = assemble(prices)
    if len(X) < 250:
        return None
    m = evaluate(X, y)
    return {
        "正解率": m["accuracy"],
        "ベースライン": m["baseline_accuracy"],
        "差分": m["accuracy"] - m["baseline_accuracy"],
        "ROC-AUC": m["roc_auc"],
        "上昇日割合": m["up_rate_test"],
        "検証日数": m["n_test"],
    }


# ----------------------------------------------------------------------
# 5. Streamlit ユーザーインターフェース
# ----------------------------------------------------------------------
st.set_page_config(page_title="株価トレンド予測", page_icon="📈")
st.title("📈 株価トレンド予測")
st.caption(
    "テクニカル指標を用いた翌営業日の値動き方向の分類モデルです。"
    "分析手法の実証を目的としたものであり、投資助言ではありません。"
)

ticker = st.text_input("銘柄コード", value="AAPL").strip().upper()
st.caption("例：AAPL（アップル）、7203.T（トヨタ自動車）、^N225（日経225）")

if not ticker:
    st.stop()

prices = load_prices(ticker)
if prices.empty:
    st.error(f"「{ticker}」のデータを取得できませんでした。銘柄コードをご確認ください。")
    st.stop()

X, y, X_full = assemble(prices)
if len(X) < 250:
    st.error("学習と検証を行うには履歴データが不足しています。")
    st.stop()

metrics = evaluate(X, y)
edge = metrics["accuracy"] - metrics["baseline_accuracy"]

# --- 評価を先に表示する。予測結果より前に、正直な数値を置く。
st.subheader("検証データにおける性能")
st.write(
    f"学習 {metrics['n_train']} 日、検証 {metrics['n_test']} 日"
    f"（{metrics['test_start']} 〜 {metrics['test_end']}）。"
    "分割は時系列順で行っています。"
)

c1, c2, c3 = st.columns(3)
c1.metric("正解率", f"{metrics['accuracy']:.1%}", f"{edge:+.1%}（ベースライン比）")
c2.metric("ベースライン（常に多数派）", f"{metrics['baseline_accuracy']:.1%}")
c3.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

c4, c5, c6 = st.columns(3)
c4.metric("適合率", f"{metrics['precision']:.1%}")
c5.metric("再現率", f"{metrics['recall']:.1%}")
c6.metric("F1スコア", f"{metrics['f1']:.3f}")

st.caption(
    "F1スコアは参考として表示していますが、この問題設定では誤解を招く指標です。"
    "上昇日が多数派であるため、「毎日上昇と予測する」だけでモデルより高いF1が"
    "得られてしまいます。情報を持つのは ROC-AUC とベースラインとの差分です。"
)

if edge <= 0:
    st.warning(
        "本モデルは検証期間においてベースラインを上回っていません。"
        "この結果をそのまま報告することも分析の一部です。価格ベースの"
        "テクニカル指標のみから方向性の優位性が得られることは、本来期待"
        "すべきものではありません。"
    )

with st.expander("混同行列を表示"):
    st.dataframe(
        pd.DataFrame(
            metrics["confusion"],
            index=["実際：下降", "実際：上昇"],
            columns=["予測：下降", "予測：上昇"],
        )
    )

# --- しきい値分析
st.subheader("シグナルしきい値分析")
base_rate = metrics["up_rate_test"]
tbl = threshold_table(metrics["y_test"], metrics["proba"], base_rate)

st.write(
    "既定の0.50というしきい値は、すべての取引日について判断を下すことを"
    "強制します。しきい値を上げれば、確信度の高い日だけに限定して行動できます。"
    f"検証すべきは、上昇日の基準割合である {base_rate:.1%} を有意に上回る"
    "適合率が得られ、かつ十分なシグナル数が残るかどうかです。"
)

st.dataframe(
    tbl.style.format(
        {
            "対象日割合": "{:.1%}",
            "適合率": "{:.1%}",
            "ベースライン比": "{:+.1%}",
        },
        na_rep="—",
    ),
    hide_index=True,
    use_container_width=True,
)

reliable = tbl[tbl["有効"]].dropna(subset=["適合率"])
if not reliable.empty:
    fig_t, ax_t = plt.subplots(figsize=(9, 4))
    ax_t.plot(
        reliable["しきい値"],
        reliable["適合率"],
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
    ax_t.set_xlabel("Decision threshold - P(Up)")
    ax_t.set_ylabel("Precision")
    ax_t.grid(alpha=0.3)
    ax_t.legend(loc="upper left")

    ax_c = ax_t.twinx()
    ax_c.bar(
        reliable["しきい値"],
        reliable["対象日割合"],
        width=0.015,
        alpha=0.2,
        color="tab:orange",
    )
    ax_c.set_ylabel("Coverage")
    ax_c.set_ylim(0, 1)

    st.pyplot(fig_t)

    best = reliable.loc[reliable["適合率"].idxmax()]
    st.info(
        f"最も適合率が高い有効なしきい値：**{best['しきい値']:.3f}** — "
        f"シグナル数 {int(best['シグナル数'])} 件"
        f"（検証日数の {best['対象日割合']:.1%}）、"
        f"適合率 {best['適合率']:.1%}"
        f"（ベースライン比 {best['ベースライン比']:+.1%}）。\n\n"
        f"シグナル数が {MIN_SIGNALS} 件未満の行は集計から除外しています。"
        "数日分で算出した適合率は標本誤差であり、根拠にはなりません。"
        "最も見栄えの良いしきい値を選ぶことが、バックテストにおける"
        "過剰適合そのものです。"
    )
else:
    st.info(
        f"シグナル数が {MIN_SIGNALS} 件以上となるしきい値が存在しないため、"
        "統計的に意味のある適合率は算出できません。"
    )

# --- 複数市場での比較
st.subheader("複数市場での比較")
st.write(
    "同一のパイプラインを米国・日本の個別銘柄および指数に適用し、"
    "観測された優位性が「手法の性質」によるものか、"
    "「特定銘柄の性質」によるものかを検証します。"
)

if st.button("比較を実行（約30秒）"):
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
                    "正解率": "{:.1%}",
                    "ベースライン": "{:.1%}",
                    "差分": "{:+.1%}",
                    "ROC-AUC": "{:.3f}",
                    "上昇日割合": "{:.1%}",
                    "検証日数": "{:.0f}",
                }
            ),
            use_container_width=True,
        )
        mean_edge = comp["差分"].mean()
        mean_auc = comp["ROC-AUC"].mean()
        n_positive = int((comp["差分"] > 0).sum())
        st.caption(
            f"{len(comp)} 銘柄におけるベースラインとの差分の平均："
            f"{mean_edge:+.1%}。上回ったのは {len(comp)} 銘柄中 {n_positive} 銘柄。"
            f"平均 ROC-AUC は {mean_auc:.4f} です。\n\n"
            "真に予測力を持つ手法であれば、市場をまたいで一貫した優位性を"
            "示すはずです。符号が銘柄によって入れ替わる場合、それは偶然の"
            "変動と解釈するのが妥当です。"
        )
    else:
        st.warning("比較用のデータを取得できませんでした。")

# --- 当日の予測
st.subheader("翌営業日のシグナル")
final_model = make_model().fit(X, y)

latest = X_full.replace([np.inf, -np.inf], np.nan).dropna().iloc[[-1]]
prob_up = float(final_model.predict_proba(latest[X.columns])[0, 1])

direction = "上昇 📈" if prob_up >= 0.5 else "下降 📉"
st.metric(
    f"{ticker} — 翌営業日",
    direction,
    f"確信度 {max(prob_up, 1 - prob_up):.1%}",
)
st.progress(prob_up)
st.caption(
    f"上昇確率 P(Up) = {prob_up:.3f}。"
    "0.50付近の値は実質的な情報を持ちません。"
)

with st.expander("モデルの係数を表示"):
    st.caption(
        "標準化後の係数です。絶対値が大きいほど、そのテクニカル指標を"
        "モデルが強く参照していることを示します。"
    )
    coefs = pd.Series(
        final_model.named_steps["clf"].coef_[0], index=X.columns
    ).sort_values(key=abs, ascending=False)
    st.dataframe(coefs.rename("係数（標準化後）").to_frame())

# --- 株価チャート
st.subheader("株価の推移")
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
    "【限界】テクニカル指標のみを使用しており、ファンダメンタルズ、"
    "マクロ経済指標、ニュース、市場センチメントは含んでいません。"
    "取引コストおよび流動性も考慮していません。"
    "本アプリケーションの出力は投資判断のためのシグナルではありません。"
)
