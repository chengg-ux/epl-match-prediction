"""评估指标。概率数组列顺序 [主胜, 平, 客胜]，结果用 'H'/'D'/'A' 表示。"""
import numpy as np

_IDX = {"H": 0, "D": 1, "A": 2}


def onehot(y) -> np.ndarray:
    idx = np.array([_IDX[v] for v in y])
    out = np.zeros((len(idx), 3))
    out[np.arange(len(idx)), idx] = 1
    return out


def log_loss_each(p, y, eps=1e-12) -> np.ndarray:
    """每场的 log-loss = -ln(给真实结果的概率)。越小越好；乱猜(1/3)=1.0986。"""
    p = np.clip(np.asarray(p, float), eps, 1)
    return -np.log((p * onehot(y)).sum(axis=1))


def brier_each(p, y) -> np.ndarray:
    """每场的 Brier = 三个结果的 (预测概率-实际0/1)^2 之和。越小越好；乱猜=0.667。"""
    return ((np.asarray(p, float) - onehot(y)) ** 2).sum(axis=1)


def log_loss(p, y) -> float:
    return float(log_loss_each(p, y).mean())


def brier(p, y) -> float:
    return float(brier_each(p, y).mean())


def accuracy(p, y) -> float:
    pred = np.asarray(p).argmax(axis=1)
    return float((pred == onehot(y).argmax(axis=1)).mean())


def calibration_table(p, y, bins=10) -> "pd.DataFrame":
    """把所有 (比赛, 结果) 的预测概率拍平，分箱后看「平均预测概率 vs 实际发生频率」。"""
    import pandas as pd
    p = np.asarray(p, float).ravel()
    o = onehot(y).ravel()
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    df = pd.DataFrame({"bin": b, "p": p, "o": o})
    g = df.groupby("bin").agg(pred=("p", "mean"), actual=("o", "mean"), n=("o", "size")).reset_index()
    return g
