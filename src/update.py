"""一键更新：下载最新数据 -> 重跑滚动回测 -> 更新指标和图 -> 预测未来赛程并维护预测日志。
运行：python -m src.update
（超参数沿用 results/tuned_params.json，不会因为新数据被重新调整。）
"""
from . import backtest, data, evaluate, live


def main():
    data.build()
    w = backtest.run()
    backtest.summarize(w).to_csv(backtest.RESULTS / "metrics_overall.csv", index=False)
    evaluate.main()
    live.run()


if __name__ == "__main__":
    main()
