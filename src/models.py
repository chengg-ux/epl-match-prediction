"""三个由简到繁的模型。所有模型都遵循同一个接口：

    model.predict(date_df, history_df) -> (n, 3) 概率数组，列顺序 [主胜, 平, 客胜]

其中 history_df 只包含「当天之前」的比赛，调用方（backtest.py）负责保证这一点。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import poisson

OUTCOMES = ["H", "D", "A"]


# ---------------------------------------------------------------- 基线 ----
def odds_to_prob(h, d, a) -> np.ndarray:
    """博彩赔率 -> 隐含概率。赔率倒数之和 >1（庄家抽水），按比例归一化去掉抽水。"""
    raw = np.column_stack([1 / np.asarray(h, float), 1 / np.asarray(d, float), 1 / np.asarray(a, float)])
    return raw / raw.sum(axis=1, keepdims=True)


class BaseRateModel:
    """基线 1：不看队伍，只用历史上主胜/平/客胜的占比作为所有比赛的概率。
    「永远押主胜」的准确率就等于这一模型里最大那一项的命中率。"""

    name = "基线：历史频率"

    def predict(self, today: pd.DataFrame, hist: pd.DataFrame) -> np.ndarray:
        freq = hist["FTR"].value_counts(normalize=True).reindex(OUTCOMES).fillna(1 / 3).values
        return np.tile(freq, (len(today), 1))


class AlwaysHomeModel:
    """基线 1b：永远预测主胜（给 0.98 的概率，仅用于算准确率；log-loss 没有意义不报告）。"""

    name = "基线：永远主胜"

    def predict(self, today, hist):
        return np.tile([1.0, 0.0, 0.0], (len(today), 1))


# ------------------------------------------------------------------ Elo ----
class EloModel:
    """Elo 评分：每队有一个「战力分」，赢了涨、输了跌，爆冷涨跌更多。

    - k: 每场最多调整多少分（越大对近期越敏感）
    - hfa: 主场优势折算成多少 Elo 分
    - theta: 控制平局概率（越大平局越多）；用「有序 logit」把分差转成胜/平/负
    - 进球差越大，调整幅度越大（乘 ln(|净胜球|+1)+1，借鉴 FiveThirtyEight/ClubElo）
    - 新球队（升班马）起始分 init_new，其余球队起始 1500
    """

    name = "Elo 评分"

    def __init__(self, k=20.0, hfa=60.0, theta=0.55, init_new=1400.0, season_regress=0.2):
        self.k, self.hfa, self.theta = k, hfa, theta
        self.init_new, self.season_regress = init_new, season_regress

    def _probs(self, diff):
        x = diff / 400.0 * np.log(10)
        p_home = expit(x - self.theta)
        p_away = expit(-x - self.theta)
        return np.column_stack([p_home, 1 - p_home - p_away, p_away])

    def ratings_before(self, hist: pd.DataFrame) -> dict:
        r: dict[str, float] = {}
        last_season = None
        for row in hist.itertuples(index=False):
            if last_season is not None and row.Season != last_season:
                # 新赛季：所有队向 1500 回归一点（人员变动、不确定性变大）
                for t in r:
                    r[t] = 1500 + (r[t] - 1500) * (1 - self.season_regress)
            last_season = row.Season
            rh = r.setdefault(row.HomeTeam, self.init_new if r else 1500.0)
            ra = r.setdefault(row.AwayTeam, self.init_new if r else 1500.0)
            exp_home = 1 / (1 + 10 ** (-(rh + self.hfa - ra) / 400))
            score = {"H": 1.0, "D": 0.5, "A": 0.0}[row.FTR]
            mult = np.log(abs(row.FTHG - row.FTAG) + 1) + 1
            delta = self.k * mult * (score - exp_home)
            r[row.HomeTeam] = rh + delta
            r[row.AwayTeam] = ra - delta
        return r

    def predict(self, today, hist):
        r = self.ratings_before(hist)
        diff = np.array([
            r.get(h, self.init_new) + self.hfa - r.get(a, self.init_new)
            for h, a in zip(today["HomeTeam"], today["AwayTeam"])
        ])
        return self._probs(diff)


# -------------------------------------------------------------- 泊松 ----
class PoissonModel:
    """泊松进球模型：假设主、客队进球数各自服从泊松分布。

    主队期望进球 = exp(主场优势 + 主队进攻力 - 客队防守力)
    客队期望进球 = exp(客队进攻力 - 主队防守力)
    进攻力/防守力从历史比分里「最大似然」估计；越近的比赛权重越大（半衰期 half_life 天），
    再加一点收缩（ridge）让样本少的球队不被极端值带偏。最后把 0~10 球的比分概率表
    求和，得到胜/平/负概率。
    """

    name = "泊松进球"

    def __init__(self, half_life=365.0, ridge=1.0, max_days=1500, max_goals=10):
        self.half_life, self.ridge = half_life, ridge
        self.max_days, self.max_goals = max_days, max_goals

    def fit(self, hist: pd.DataFrame, now: pd.Timestamp):
        h = hist[hist["Date"] >= now - pd.Timedelta(days=self.max_days)]
        teams = sorted(set(h["HomeTeam"]) | set(h["AwayTeam"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        hi = h["HomeTeam"].map(idx).values
        ai = h["AwayTeam"].map(idx).values
        hg = h["FTHG"].values.astype(float)
        ag = h["FTAG"].values.astype(float)
        age = (now - h["Date"]).dt.days.values.astype(float)
        w = 0.5 ** (age / self.half_life)
        ridge = self.ridge

        def unpack(p):
            return p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1]

        def nll(p):
            att, dfn, ha, mu = unpack(p)
            lh = mu + ha + att[hi] - dfn[ai]
            la = mu + att[ai] - dfn[hi]
            ll = w * (hg * lh - np.exp(lh) + ag * la - np.exp(la))
            pen = ridge * (np.sum(att ** 2) + np.sum(dfn ** 2))
            val = -ll.sum() + pen
            eh = w * (np.exp(lh) - hg)   # d(-ll)/d(lh)
            ea = w * (np.exp(la) - ag)
            g_att = np.bincount(hi, eh, n) + np.bincount(ai, ea, n) + 2 * ridge * att
            g_def = -np.bincount(ai, eh, n) - np.bincount(hi, ea, n) + 2 * ridge * dfn
            return val, np.concatenate([g_att, g_def, [eh.sum()], [eh.sum() + ea.sum()]])

        p0 = np.zeros(2 * n + 2)
        p0[2 * n] = 0.25
        p0[2 * n + 1] = 0.2
        res = minimize(nll, p0, jac=True, method="L-BFGS-B")
        att, dfn, ha, mu = unpack(res.x)
        self.att = {t: att[i] for t, i in idx.items()}
        self.dfn = {t: dfn[i] for t, i in idx.items()}
        self.ha, self.mu = ha, mu
        return self

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        lh = np.exp(self.mu + self.ha + self.att.get(home, 0.0) - self.dfn.get(away, 0.0))
        la = np.exp(self.mu + self.att.get(away, 0.0) - self.dfn.get(home, 0.0))
        g = np.arange(self.max_goals + 1)
        return np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))

    def predict(self, today, hist):
        self.fit(hist, today["Date"].min())
        out = []
        for h, a in zip(today["HomeTeam"], today["AwayTeam"]):
            m = self.score_matrix(h, a)
            p = np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])
            out.append(p / p.sum())
        return np.array(out)

