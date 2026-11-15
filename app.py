"""英超胜平负概率预测 —— Streamlit 页面。运行：streamlit run app.py"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.metrics import calibration_table

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"

st.set_page_config(page_title="英超胜平负概率预测", page_icon="⚽", layout="wide")

MODELS = {"bookmaker": "博彩赔率（Bet365 去抽水）", "elo": "Elo 评分", "poisson": "泊松进球", "base_rate": "基线：历史频率"}
COLORS = {"博彩赔率（Bet365 去抽水）": "#2a9d8f", "Elo 评分": "#e76f51", "泊松进球": "#264653", "基线：历史频率": "#adb5bd"}


@st.cache_data
def load():
    pred = pd.read_csv(RES / "predictions.csv", parse_dates=["Date"])
    allm = pd.read_csv(ROOT / "data" / "epl_matches.csv", parse_dates=["Date"])
    return pred, allm


def form(allm, team, date, n=5):
    """球队在 date 之前最近 n 场的结果，如 'W W D L W' 和积分。"""
    g = allm[(allm["Date"] < date) & ((allm["HomeTeam"] == team) | (allm["AwayTeam"] == team))].tail(n)
    res = []
    for r in g.itertuples():
        home = r.HomeTeam == team
        if r.FTR == "D":
            res.append("D")
        elif (r.FTR == "H") == home:
            res.append("W")
        else:
            res.append("L")
    pts = sum({"W": 3, "D": 1, "L": 0}[x] for x in res)
    return " ".join(res) if res else "-", pts


pred, allm = load()

st.title("⚽ 英超赛事胜平负概率预测")
st.warning("**仅供学习研究，不构成投注建议。** 本项目是统计建模练习，模型在回测中没有跑赢博彩赔率。")

tab_up, tab1, tab_form, tab2, tab_log, tab3 = st.tabs(
    ["🔮 下一轮预测", "📅 历史逐场", "📈 球队状态曲线", "📊 回测表现", "📝 赛后对答案", "ℹ️ 关于"])

# ------------------------------------------------------------ 逐场预测 ----
with tab1:
    seasons = sorted(pred["Season"].unique(), reverse=True)
    c1, c2 = st.columns(2)
    season = c1.selectbox("赛季", seasons)
    s = pred[pred["Season"] == season]
    rounds = sorted(s["Round"].unique())
    rnd = c2.selectbox("轮次（按赛程顺序每 10 场一轮）", rounds, index=len(rounds) - 1)
    g = s[s["Round"] == rnd].sort_values("Date")
    st.caption("每场预测都只用该比赛日之前的数据（滚动回测产生），并非事后诸葛亮。")

    rows = []
    for r in g.itertuples():
        hf, hp = form(allm, r.HomeTeam, r.Date)
        af, ap = form(allm, r.AwayTeam, r.Date)
        rows.append({
            "日期": r.Date.strftime("%Y-%m-%d"),
            "对阵": f"{r.HomeTeam} vs {r.AwayTeam}",
            "实际比分": f"{r.FTHG}-{r.FTAG}",
            "主队近5场": f"{hf}（{hp}分）", "客队近5场": f"{af}（{ap}分）",
            "Elo 主/平/客": f"{r.elo_pH:.0%} / {r.elo_pD:.0%} / {r.elo_pA:.0%}",
            "泊松 主/平/客": f"{r.poisson_pH:.0%} / {r.poisson_pD:.0%} / {r.poisson_pA:.0%}",
            "赔率 主/平/客": f"{r.bookmaker_pH:.0%} / {r.bookmaker_pD:.0%} / {r.bookmaker_pA:.0%}",
        })
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    st.subheader("单场详情")
    labels = [f"{r.HomeTeam} vs {r.AwayTeam}" for r in g.itertuples()]
    pick = st.selectbox("选一场比赛", labels)
    r = g.iloc[labels.index(pick)]
    long = []
    for k, nm in MODELS.items():
        for o, lab in zip("HDA", ["主胜", "平", "客胜"]):
            long.append({"模型": nm, "结果": lab, "概率": r[f"{k}_p{o}"]})
    fig = px.bar(pd.DataFrame(long), x="结果", y="概率", color="模型", barmode="group",
                 color_discrete_map=COLORS, text_auto=".0%")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    actual = {"H": "主胜", "D": "平", "A": "客胜"}[r["FTR"]]
    fig.update_layout(title=f"{pick}　实际结果：{actual}（{int(r['FTHG'])}-{int(r['FTAG'])}）", height=420)
    st.plotly_chart(fig, width='stretch')

    st.write(f"本轮 {len(g)} 场中，最高概率选项命中数：" + "，".join(
        f"{MODELS[k]} {int(sum(((g[[f'{k}_pH', f'{k}_pD', f'{k}_pA']].values.argmax(axis=1)) == g['FTR'].map({'H': 0, 'D': 1, 'A': 2}).values)))} 场"
        for k in ["bookmaker", "elo", "poisson"]))

# ------------------------------------------------------------ 回测表现 ----
with tab2:
    ov = pd.read_csv(RES / "metrics_overall.csv")
    st.subheader("整体表现（2018/19 起，共 %d 场）" % ov["场次"].iloc[0])
    st.dataframe(ov.style.format({"LogLoss": "{:.4f}", "Brier": "{:.4f}", "准确率": "{:.1%}"}, na_rep="—"),
                 width='stretch', hide_index=True)
    st.caption("LogLoss、Brier 越小越好（三分类乱猜分别是 1.099、0.667）。结论：模型明显好于基线，但没有跑赢博彩赔率。")

    bs = pd.read_csv(RES / "metrics_by_season.csv")
    metric = st.radio("逐赛季指标", ["LogLoss", "Brier", "Accuracy"], horizontal=True)
    fig = px.line(bs, x="Season", y=metric, color="Model", markers=True)
    st.plotly_chart(fig, width='stretch')
    st.caption("2026/27 赛季仍在进行，场次少，波动大。")

    st.subheader("校准曲线")
    st.caption("横轴是模型给的概率，纵轴是真实发生的频率。越贴近对角线越「说到做到」。")
    mdl = st.multiselect("模型", ["bookmaker", "elo", "poisson"], default=["bookmaker", "elo", "poisson"],
                         format_func=lambda k: MODELS[k])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="black"), name="完美校准"))
    for k in mdl:
        p = pred[[f"{k}_pH", f"{k}_pD", f"{k}_pA"]].values
        t = calibration_table(p, pred["FTR"].values, bins=10)
        fig.add_trace(go.Scatter(x=t["pred"], y=t["actual"], mode="lines+markers", name=MODELS[k],
                                 line=dict(color=COLORS[MODELS[k]]), customdata=t["n"],
                                 hovertemplate="预测 %{x:.1%}<br>实际 %{y:.1%}<br>样本 %{customdata}<extra></extra>"))
    fig.update_layout(xaxis_title="预测概率", yaxis_title="实际发生频率", height=480)
    st.plotly_chart(fig, width='stretch')

    bt = pd.read_csv(RES / "bootstrap_logloss_diff.csv")
    st.subheader("差距是不是运气？（bootstrap 95% 区间）")
    st.dataframe(bt.rename(columns={"model": "模型", "vs": "对比对象", "mean_logloss_diff": "LogLoss 平均差（正=更差）",
                                    "ci_lo": "区间下限", "ci_hi": "区间上限"}).round(4),
                 width='stretch', hide_index=True)

# ------------------------------------------------------------ 下一轮预测 ----
with tab_up:
    up = pd.read_csv(RES / "upcoming.csv", parse_dates=["Date"])
    st.caption("用截至最近一场已赛比赛的全部数据实时拟合，预测官方赛程表里的未来比赛。数据来自 football-data.co.uk，更新频率约每周一次。")
    if up.empty:
        st.info("当前赛程表里没有英超比赛（可能正值国际比赛日，或赛程尚未公布）。比赛日临近、数据更新后这里会自动出现预测。")
    else:
        st.dataframe(pd.DataFrame({
            "日期": up["Date"].dt.strftime("%Y-%m-%d"), "对阵": up["HomeTeam"] + " vs " + up["AwayTeam"],
            "Elo 主/平/客": [f"{a:.0%} / {b:.0%} / {c:.0%}" for a, b, c in zip(up.elo_pH, up.elo_pD, up.elo_pA)],
            "泊松 主/平/客": [f"{a:.0%} / {b:.0%} / {c:.0%}" for a, b, c in zip(up.poisson_pH, up.poisson_pD, up.poisson_pA)],
            "赔率 主/平/客": [("—" if pd.isna(a) else f"{a:.0%} / {b:.0%} / {c:.0%}") for a, b, c in zip(up.bookmaker_pH, up.bookmaker_pD, up.bookmaker_pA)],
        }), width="stretch", hide_index=True)
        st.caption("这些是实时预测，尚无评估成绩；比赛踢完后会进入「赛后对答案」。")

# ---------------------------------------------------------- 球队状态曲线 ----
with tab_form:
    eh = pd.read_csv(RES / "elo_history.csv", parse_dates=["Date"])
    st.caption("Elo 分数就是球队的「战力分」：赢球上升、输球下降。曲线的起伏就是球队状态的起伏。")
    latest = eh.sort_values("Date").groupby("Team").tail(1)
    default = latest.sort_values("Elo", ascending=False)["Team"].head(4).tolist()
    teams = st.multiselect("选择球队（可多选对比）", sorted(eh["Team"].unique()), default=default)
    start = st.selectbox("起始赛季", sorted(eh["Season"].unique()), index=len(eh["Season"].unique()) - 3)
    sub = eh[(eh["Team"].isin(teams)) & (eh["Season"] >= start)]
    fig = px.line(sub, x="Date", y="Elo", color="Team")
    fig.update_layout(height=480, yaxis_title="Elo 分数")
    st.plotly_chart(fig, width="stretch")
    st.subheader("当前 Elo 排名（最新一场之后）")
    rank = latest.sort_values("Elo", ascending=False)
    rank = rank[rank["Season"] == eh["Season"].max()][["Team", "Elo"]].round(0).astype({"Elo": int})
    st.dataframe(rank.rename(columns={"Team": "球队"}), hide_index=True, width="stretch")

# ---------------------------------------------------------- 赛后对答案 ----
with tab_log:
    st.caption("每条预测都在比赛开始前写入日志，之后永不修改，赛后只补充实际结果——所以这是真正的事前预测记录。")
    lg = pd.read_csv(RES / "live_log.csv", parse_dates=["Date"])
    if lg.empty:
        st.info("日志还是空的：从下一次有英超赛程的更新开始记录，比赛踢完后会在这里「对答案」。")
    else:
        done = lg.dropna(subset=["FTR"])
        st.write(f"已记录 {len(lg)} 场，其中已踢完 {len(done)} 场。")
        if len(done):
            from src.metrics import log_loss, accuracy
            rows = []
            for k in ["elo", "poisson", "bookmaker"]:
                d = done.dropna(subset=[f"{k}_pH"])
                if len(d):
                    p = d[[f"{k}_pH", f"{k}_pD", f"{k}_pA"]].values
                    rows.append({"模型": MODELS[k], "场次": len(d), "LogLoss": log_loss(p, d["FTR"].values),
                                 "准确率": accuracy(p, d["FTR"].values)})
            st.dataframe(pd.DataFrame(rows).style.format({"LogLoss": "{:.3f}", "准确率": "{:.0%}"}), hide_index=True, width="stretch")
            st.caption("样本很少时指标波动极大，不要过度解读。")
        st.dataframe(lg.sort_values("Date", ascending=False), hide_index=True, width="stretch")
# ------------------------------------------------------------------ 关于 ----
with tab3:
    st.markdown((ROOT / "docs" / "方法说明.md").read_text(encoding="utf-8"))



