---
layout: default
title: 产物地图
nav_order: 8
---

# 产物地图（训练 / 验证 / 回测 / 生产）

本页是各环节运行产物的位置与字段说明。所有目录都在 `.gitignore` 中（运行资产，
可重建），代码与文档才入库。

## 训练：MLflow 实验（`mlruns/`）

| 实验 ID | 名称 | 内容 |
| --- | --- | --- |
| `156869948604814731` | `minute_enhanced_tk10_nd8` | Champion 及历次全量训练（11 recorder） |
| `731773051679704979` | `minute_enhanced_rolling` | 滚动重训主模型 |
| `950921557012461449` | `minute_enhanced_rolling_xgb` | 门控 XGBoost 候选 |

每个 recorder 的产物：

```
mlruns/<experiment>/<recorder>/
├── artifacts/params.pkl          # 模型权重（LightGBM/XGBoost booster）
├── artifacts/pred.pkl            # 预测分数（train/valid/test 段逐日）
├── artifacts/label.pkl           # 对应 label 真实值
├── artifacts/sig_analysis/ic.pkl # 逐日 IC / RankIC（ric.pkl）序列
├── metrics/                      # IC、ICIR、Rank IC、l2.train/valid 曲线
├── params/                       # 完整训练配置（handler/model/数据段），可复现
└── artifacts/code_*.txt          # 训练时代码快照与 git diff
```

## 回测：62 日影子回放（`data/historical_shadow_replay/<日期>/`）

66 个交易日、每天 25 个文件，构成完整的"打分 → 下单 → 成交 → 对账"审计链：

| 文件 | 内容 |
| --- | --- |
| `run_manifest.json` | 模型 recorder ID、模型/特征 schema SHA256、锁状态 |
| `preflight.json` / `bar_quality.json` / `universe_quality.json` / `universe_raw.csv` / `universe_eligible.csv` | 盘前检查：股池规模、分钟 bar 覆盖率、剔除名单 |
| `factor_bars.parquet` / `features.parquet` | 当日分钟 bar 与 18 因子值 |
| `scores.csv` | 全池打分（含 `change_941`、涨跌停阈值） |
| `rebalance_decision.json` | keep / sell / buy_ranked 决策及理由 |
| `buy_orders.csv` / `sell_orders.csv` / `buy_fills.csv` / `sell_fills.csv` | 订单与模拟成交（失败有 fill_status：NO_BAR / 涨停拦截） |
| `cash_after_sell.json` / `cash_after_buy.json` | 两阶段现金状态 |
| `positions_before/after_sell/after_close.csv` | 三时点持仓快照（含 hold_days） |
| `daily_reconciliation.json` | 逐日对账（仓位 mismatch、现金差） |
| `signal_parity.json` | 与生产链路的一致性比对（score max diff、Top10 overlap） |
| `shadow_day_result.json` | 当日总结（NAV、PASS/FAIL） |
| `execution_bar_0941.parquet` | 9:41 执行 bar |

老版回放：`data/historical_signals_20260601_20260720/`（61 日，已被新链路取代）。

## 纸面跟踪（真实交易日）

| 文件 | 内容 |
| --- | --- |
| `data/live_signals.csv` | 每日 Top10 信号（665 行，7 个信号日） |
| `data/live_settle.csv` | 逐日结算记录 |
| `data/live_nav.csv` | 组合净值（结算日口径） |

最新状态：信号至 2026-07-14，结算至 2026-07-13，此后无新记录（恢复待定）。

## 生产信号（`data/production_signals/<日期>/`）

- `candidate_pool.csv`：当日候选池；
- `orders.csv`：调仓清单（人工提交券商）；
- `metadata.json`：模型与数据质量元信息。

## 验证产物

| 位置 | 状态 |
| --- | --- |
| 各 recorder `sig_analysis/ic.pkl` | ✅ valid 段逐日 IC |
| `data/model_governance/`（CANDIDATE manifest / validation 报告） | 治理流程已就绪，尚无实际产出 |
| `data/index_stage_joint_predictions/`（joint TVT 逐 fold pkl） | 缓存未保留，重跑 `scripts/validate_index_stage_joint_models.py` 再生 |
| `logs/full_run_*/`（00_baseline → 05_summary.md） | 一次性全链路运行日志 |

## 查看入口

```bash
# 某天回放的完整决策链
ls data/historical_shadow_replay/2026-06-29/

# 某次训练的 IC 曲线
cat mlruns/156869948604814731/<recorder_id>/metrics/IC

# 62 日回放总结
cat logs/full_run_20260711_231633/05_summary.md
```
