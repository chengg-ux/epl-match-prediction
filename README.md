# 英超赛事胜平负概率预测

用公开历史数据，从简单到复杂做了三个模型（基线、Elo、泊松进球），用**按时间滚动**的回测与博彩赔率隐含概率对比，并用 Streamlit 做成可交互网页。

> **仅供学习研究，不构成投注建议。** 结果显示模型没有跑赢博彩赔率，详见 [docs/方法说明.md](docs/方法说明.md)。

## 安装与运行（Windows，Python 3.12）

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

浏览器会自动打开 http://localhost:8501 。仓库已包含数据和回测结果，无需联网即可运行页面。

## 重新生成数据与结果（可选）

```bash
python -m src.data          # 下载并整理数据到 data/
python -m src.backtest      # 滚动回测，输出 results/predictions.csv、metrics_overall.csv（约 1–3 分钟）
python -m src.evaluate      # 分赛季指标、bootstrap 区间、校准曲线图
```

加 `retune=True`（见 `src/backtest.py` 的 `run`）可重新在 2017/18 赛季上调参。

## 一键更新（含未来赛程预测）

```bash
python -m src.update        # 下载最新数据 -> 重跑回测 -> 更新指标 -> 预测未来赛程 -> 更新「赛后对答案」日志（约 2–3 分钟）
```

- 超参数沿用 `results/tuned_params.json`，不会因新数据重新调整。
- `.github/workflows/update.yml` 让 GitHub 每周一、周四自动运行上面的命令并提交结果，Streamlit 网页随之刷新。
- 未来赛程来自 football-data.co.uk 的 `fixtures.csv`，国际比赛日等时段可能没有英超场次，页面会显示提示。
- `results/live_log.csv` 只在比赛开始前写入预测、赛后补结果，用来事后检验模型。

## 免费部署

把整个文件夹推到 GitHub，在 [Streamlit Community Cloud](https://streamlit.io/cloud) 选择该仓库、入口文件填 `app.py` 即可。

## 目录

```
app.py              Streamlit 页面（6 个标签页）
src/live.py         未来赛程预测、Elo 状态曲线、赛前预测日志
src/update.py       一键更新
src/data.py         数据下载与整理
src/models.py       基线 / Elo / 泊松
src/backtest.py     按时间滚动的回测与调参
src/metrics.py      log-loss、Brier、准确率、校准
src/evaluate.py     分赛季指标、置信区间、校准图
data/               原始与整理后的数据
results/            回测结果表与图
docs/方法说明.md    一页方法说明
docs/简历描述.md    简历描述草稿
```

## 数据来源与声明

数据来自 [football-data.co.uk](https://www.football-data.co.uk/)，仅用于学习研究；使用请遵守该站条款并注明来源。

