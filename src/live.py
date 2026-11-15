"""实时部分：预测未来赛程、记录 Elo 状态曲线、维护「赛前预测 → 赛后对答案」日志。

规则：日志里的预测只在比赛开始前写入一次，之后永不改动，赛后只补充实际结果。
这样日志才是真正的「事前预测」，不能事后修改。
"""
from __future__ import annotations

import io
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from .models import EloModel, PoissonModel, odds_to_prob

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
LOG = RES / "live_log.csv"
UPCOMING = RES / "upcoming.csv"
ELO_HIST = RES / "elo_history.csv"

LOG_COLS = ["made_at", "Date", "HomeTeam", "AwayTeam", "elo_pH", "elo_pD", "elo_pA",
            "poisson_pH", "poisson_pD", "poisson_pA", "bookmaker_pH", "bookmaker_pD", "bookmaker_pA",
            "FTHG", "FTAG", "FTR"]


def fetch_fixtures() -> pd.DataFrame:
    """football-data.co.uk 提供未来一周左右的赛程（含赔率）。国际比赛日等时段可能没有英超场次，返回空表。"""
    req = urllib.request.Request(FIXTURES_URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        raw = urllib.request.urlopen(req, timeout=60).read()
    except Exception as e:
        print("赛程下载失败：", e)
        return pd.DataFrame()
    df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    df = df[df["Div"] == "E0"].copy()
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    return df.reset_index(drop=True)


def _params():
    return json.loads((RES / "tuned_params.json").read_text(encoding="utf-8"))


def predict_fixtures(played: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    """用截至目前所有已赛数据，预测赛程表里的比赛。"""
    cfg = _params()
    out = fx[["Date", "HomeTeam", "AwayTeam"]].copy()
    out[["elo_pH", "elo_pD", "elo_pA"]] = EloModel(**cfg["elo"]).predict(fx, played)
    out[["poisson_pH", "poisson_pD", "poisson_pA"]] = PoissonModel(**cfg["poisson"]).predict(fx, played)
    if {"B365H", "B365D", "B365A"} <= set(fx.columns) and fx[["B365H", "B365D", "B365A"]].notna().all().all():
        out[["bookmaker_pH", "bookmaker_pD", "bookmaker_pA"]] = odds_to_prob(fx["B365H"], fx["B365D"], fx["B365A"])
    else:
        out[["bookmaker_pH", "bookmaker_pD", "bookmaker_pA"]] = float("nan")
    return out


def elo_history(played: pd.DataFrame) -> pd.DataFrame:
    rec: list = []
    EloModel(**_params()["elo"]).ratings_before(played, record=rec)
    return pd.DataFrame(rec, columns=["Date", "Season", "Team", "Elo"])


def update_log(preds: pd.DataFrame, played: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    log = pd.read_csv(LOG, parse_dates=["Date"]) if LOG.exists() else pd.DataFrame(columns=LOG_COLS)
    # 1) 补全已踢完比赛的实际结果（不改动当初的预测）
    key = played.set_index(["Date", "HomeTeam", "AwayTeam"])[["FTHG", "FTAG", "FTR"]]
    log = log.astype({"FTHG": "float64", "FTAG": "float64", "FTR": "object"})
    for i, r in log[log["FTR"].isna()].iterrows():
        k = (r["Date"], r["HomeTeam"], r["AwayTeam"])
        if k in key.index:
            res = key.loc[k]
            log.loc[i, "FTHG"], log.loc[i, "FTAG"], log.loc[i, "FTR"] = float(res["FTHG"]), float(res["FTAG"]), res["FTR"]
    # 2) 只记录「严格晚于今天」且尚未记录的比赛，确保一定是赛前预测
    have = set(zip(log["Date"], log["HomeTeam"], log["AwayTeam"]))
    new = preds[(preds["Date"] > today) & np.array([(d, h, a) not in have for d, h, a in zip(preds["Date"], preds["HomeTeam"], preds["AwayTeam"])], dtype=bool)].copy()
    if len(new):
        new["made_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        log = pd.concat([log, new.reindex(columns=LOG_COLS)], ignore_index=True)
    log = log.reindex(columns=LOG_COLS)
    log.to_csv(LOG, index=False)
    return log


def run(today: pd.Timestamp | None = None, fx: pd.DataFrame | None = None) -> None:
    RES.mkdir(exist_ok=True)
    played = pd.read_csv(data.CLEAN, parse_dates=["Date"])
    today = today if today is not None else pd.Timestamp(datetime.now(timezone.utc).date())
    elo_history(played).to_csv(ELO_HIST, index=False)
    fx = fetch_fixtures() if fx is None else fx
    fx = fx[fx["Date"] >= today] if len(fx) else fx
    if len(fx):
        preds = predict_fixtures(played, fx)
        preds.to_csv(UPCOMING, index=False)
    else:
        preds = pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam"])
        pd.DataFrame(columns=LOG_COLS[1:13]).to_csv(UPCOMING, index=False)
        print("当前赛程表里没有英超比赛（可能是国际比赛日）。")
    update_log(preds, played, today)
    print("upcoming:", len(preds))


if __name__ == "__main__":
    run()

