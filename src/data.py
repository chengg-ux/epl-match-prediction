"""下载并整理 football-data.co.uk 的英超历史数据。

数据来源：https://www.football-data.co.uk/englandm.php （免费公开，仅供学习研究）
"""
from pathlib import Path
import io
import urllib.request

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "epl_matches.csv"

# 2016/17 ~ 2025/26 共 10 个完整赛季 + 进行中的 2026/27；代码 1617 表示 2016/17
SEASONS = [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(2016, 2027)]
URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"


def season_label(code: str) -> str:
    return f"20{code[:2]}/{code[2:]}"


def download(refresh: bool = False) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for code in SEASONS:
        path = RAW / f"E0_{code}.csv"
        if path.exists() and not refresh and code != SEASONS[-1]:
            continue
        req = urllib.request.Request(URL.format(code=code), headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                path.write_bytes(r.read())
            print("downloaded", code)
        except Exception as e:  # 网络失败时，若本地已有缓存就继续用
            print("download failed", code, e)


def load() -> pd.DataFrame:
    frames = []
    for code in SEASONS:
        path = RAW / f"E0_{code}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, encoding="latin-1").dropna(subset=["HomeTeam", "FTR"])
        frames.append(df.assign(Season=season_label(code)))
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    # 赔率：优先用 Bet365（所有赛季都有），再用多家均值作对照
    keep = ["Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
            "B365H", "B365D", "B365A", "AvgH", "AvgD", "AvgA", "PSH", "PSD", "PSA"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].sort_values(["Date", "HomeTeam"]).reset_index(drop=True)
    # 轮次：每个赛季内按日期顺序，每 10 场为一轮（英超 20 队，每轮 10 场）
    df["Round"] = df.groupby("Season").cumcount() // 10 + 1
    return df


def build() -> pd.DataFrame:
    download()
    df = load()
    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN, index=False)
    return df


if __name__ == "__main__":
    d = build()
    print(d.shape)
    print(d.groupby("Season").size())
    print(d.isna().sum()[lambda s: s > 0])
