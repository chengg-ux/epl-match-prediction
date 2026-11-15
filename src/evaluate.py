"""读取 results/predictions.csv，生成分赛季指标、置信区间和校准曲线。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import log_loss_each, brier_each, accuracy, calibration_table

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
NAMES = {"base_rate": "Base rate", "bookmaker": "Bookmaker (Bet365)", "elo": "Elo", "poisson": "Poisson"}


def probs(w, k):
    return w[[f"{k}_pH", f"{k}_pD", f"{k}_pA"]].values


def by_season(w):
    rows = []
    for s, g in w.groupby("Season"):
        for k in NAMES:
            p, y = probs(g, k), g["FTR"].values
            rows.append(dict(Season=s, Model=NAMES[k], N=len(g), LogLoss=log_loss_each(p, y).mean(),
                             Brier=brier_each(p, y).mean(), Accuracy=accuracy(p, y)))
    return pd.DataFrame(rows)


def bootstrap_diff(w, a, b="bookmaker", n_boot=2000, seed=0):
    """模型 a 相对 b 的 log-loss 平均差（正数 = a 更差），按比赛有放回重抽样得 95% 区间。"""
    y = w["FTR"].values
    d = log_loss_each(probs(w, a), y) - log_loss_each(probs(w, b), y)
    rng = np.random.default_rng(seed)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def calibration_plot(w, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (title, ks) in zip(axes, [("All outcomes pooled", None), ("Home win", 0), ("Away win", 2)]):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
        for k, c in zip(["bookmaker", "elo", "poisson"], ["#2a9d8f", "#e76f51", "#264653"]):
            p = probs(w, k)
            y = w["FTR"].values
            if ks is None:
                t = calibration_table(p, y, bins=10)
            else:
                oh = (y == "HDA"[ks]).astype(float)
                t = calibration_table(np.column_stack([p[:, ks]] * 1 + [np.zeros(len(p))] * 2), y, bins=10)
                # 只看单一结果：重新分箱
                edges = np.linspace(0, 1, 9)
                b = np.clip(np.digitize(p[:, ks], edges) - 1, 0, 7)
                d = pd.DataFrame({"b": b, "p": p[:, ks], "o": oh}).groupby("b").agg(pred=("p", "mean"), actual=("o", "mean"), n=("o", "size"))
                t = d[d["n"] >= 20].reset_index()
            ax.plot(t["pred"], t["actual"], "o-", color=c, label=NAMES[k], ms=4)
        ax.set_title(title); ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
        ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    w = pd.read_csv(RESULTS / "predictions.csv", parse_dates=["Date"])
    bs = by_season(w)
    bs.to_csv(RESULTS / "metrics_by_season.csv", index=False)
    print(bs.pivot(index="Season", columns="Model", values="LogLoss").round(4))
    rows = []
    for a in ["elo", "poisson", "base_rate"]:
        m, lo, hi = bootstrap_diff(w, a)
        rows.append(dict(model=NAMES[a], vs="Bookmaker", mean_logloss_diff=m, ci_lo=lo, ci_hi=hi))
    m, lo, hi = bootstrap_diff(w, "elo", "base_rate")
    rows.append(dict(model="Elo", vs="Base rate", mean_logloss_diff=m, ci_lo=lo, ci_hi=hi))
    m, lo, hi = bootstrap_diff(w, "poisson", "elo")
    rows.append(dict(model="Poisson", vs="Elo", mean_logloss_diff=m, ci_lo=lo, ci_hi=hi))
    bt = pd.DataFrame(rows)
    bt.to_csv(RESULTS / "bootstrap_logloss_diff.csv", index=False)
    print(bt.round(4).to_string(index=False))
    calibration_plot(w, RESULTS / "calibration.png")


if __name__ == "__main__":
    main()

