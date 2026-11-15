"""按时间滚动的回测。

铁律：预测某一天的比赛时，模型只能看到「严格早于这一天」的比赛结果。
做法：把全部比赛按日期分组，逐天前进；每一天先预测、再把当天结果「揭晓」给历史。

超参数（Elo 的 K 值、泊松的时间半衰期等）只在「调参期」(2017/18 赛季，仅用 2016/17 及 2017/18
赛季内已发生的比赛) 上选定，选定之后冻结，再去跑「测试期」(2018/19 起)。
测试期的结果不会被用来回头改任何设置。
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from .metrics import log_loss, brier, accuracy
from .models import AlwaysHomeModel, BaseRateModel, EloModel, PoissonModel, odds_to_prob, OUTCOMES

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TUNE_SEASON = "2017/18"
TEST_START_SEASON = "2018/19"


def rolling_predict(model, df: pd.DataFrame, target_mask: pd.Series, every: int = 1) -> pd.DataFrame:
    """对 target_mask 为真的比赛做滚动预测。every>1 时只在每 every 个比赛日预测（仅调参用，加速）。"""
    out = []
    dates = sorted(df.loc[target_mask, "Date"].unique())
    for i, d in enumerate(dates):
        if i % every:
            continue
        today = df[(df["Date"] == d)]
        hist = df[df["Date"] < d]          # 严格早于当天——没有未来数据
        p = model.predict(today, hist)
        res = today[["Season", "Date", "Round", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].copy()
        res[["pH", "pD", "pA"]] = p
        out.append(res)
    return pd.concat(out, ignore_index=True)


def score(pred: pd.DataFrame) -> float:
    return log_loss(pred[["pH", "pD", "pA"]].values, pred["FTR"].values)


def tune(df: pd.DataFrame) -> dict:
    """在调参期选超参数。"""
    mask = df["Season"] == TUNE_SEASON
    best = {}
    elo_grid = [dict(k=k, hfa=h, theta=t) for k, h, t in itertools.product([4, 7, 10, 15, 20], [50, 70, 90], [0.5, 0.6, 0.7, 0.8])]
    scores = []
    for g in elo_grid:
        scores.append((score(rolling_predict(EloModel(**g), df, mask)), g))
    scores.sort(key=lambda x: x[0])
    best["elo"] = scores[0][1]
    best["elo_grid"] = [(float(s), g) for s, g in scores]

    pgrid = [dict(half_life=hl, ridge=r) for hl, r in itertools.product([90, 180, 365, 730], [0.5, 1.0, 2.0, 5.0])]
    scores = []
    for g in pgrid:
        scores.append((score(rolling_predict(PoissonModel(**g), df, mask, every=3)), g))
    scores.sort(key=lambda x: x[0])
    best["poisson"] = scores[0][1]
    best["poisson_grid"] = [(float(s), g) for s, g in scores]
    return best


def run(retune: bool = False) -> pd.DataFrame:
    RESULTS.mkdir(exist_ok=True)
    df = data.load() if not data.CLEAN.exists() else pd.read_csv(data.CLEAN, parse_dates=["Date"])
    cfg_path = RESULTS / "tuned_params.json"
    if retune or not cfg_path.exists():
        cfg = tune(df)
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    print("冻结的超参数：", cfg["elo"], cfg["poisson"])

    seasons = sorted(df["Season"].unique())
    test_mask = df["Season"].isin([s for s in seasons if s >= TEST_START_SEASON])
    test = df[test_mask]
    base = test[["Season", "Date", "Round", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].reset_index(drop=True)

    preds = {}
    for key, model in [("base_rate", BaseRateModel()), ("always_home", AlwaysHomeModel()),
                       ("elo", EloModel(**cfg["elo"])), ("poisson", PoissonModel(**cfg["poisson"]))]:
        print("回测", key, "...")
        p = rolling_predict(model, df, test_mask)
        p = p.set_index(["Date", "HomeTeam", "AwayTeam"])
        preds[key] = p.loc[list(zip(base["Date"], base["HomeTeam"], base["AwayTeam"])), ["pH", "pD", "pA"]].values

    # 博彩赔率（Bet365，去抽水）。只在有赔率的比赛上比较。
    preds["bookmaker"] = odds_to_prob(test["B365H"], test["B365D"], test["B365A"])

    wide = base.copy()
    for key, p in preds.items():
        wide[[f"{key}_pH", f"{key}_pD", f"{key}_pA"]] = p
    wide["Notes"] = ""
    wide.to_csv(RESULTS / "predictions.csv", index=False)
    return wide


def summarize(wide: pd.DataFrame) -> pd.DataFrame:
    keys = ["always_home", "base_rate", "bookmaker", "elo", "poisson"]
    names = {"always_home": "基线：永远押主胜", "base_rate": "基线：历史频率", "bookmaker": "博彩赔率（Bet365 去抽水）",
             "elo": "Elo 评分", "poisson": "泊松进球"}
    y = wide["FTR"].values
    rows = []
    for k in keys:
        p = wide[[f"{k}_pH", f"{k}_pD", f"{k}_pA"]].values
        rows.append(dict(模型=names[k], 场次=len(wide),
                         LogLoss=np.nan if k == "always_home" else log_loss(p, y),
                         Brier=np.nan if k == "always_home" else brier(p, y),
                         准确率=accuracy(p, y)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    w = run()
    s = summarize(w)
    s.to_csv(RESULTS / "metrics_overall.csv", index=False)
    print(s.round(4).to_string(index=False))

