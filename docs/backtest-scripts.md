---
layout: default
title: 回测与脚本解析
nav_order: 7.5
---

# 回测与脚本解析

scripts/ 与 qrun/ 下每个入口的用途、命令行参数与内部执行链。运维视角的速查见
[运维手册](operations.md)，配置口径见[配置说明](configs.md)；本文回答的是
"**这个脚本内部做了什么、参数怎么给、产物落在哪**"。

## 脚本地图（按用途分组）

| 组 | 脚本 | 一句话定位 | 状态 |
|---|---|---|---|
| 数据准备 | `python -m scripts.build_overlay` | 端到端重建 overlay（symlink + 股池 + 涨跌停 + 18 因子物化） | 现役，幂等 |
| 数据准备 | `scripts/materialize_minute.py` | 只重物化分钟因子（改公式后免重拉股池） | 现役 |
| 训练回测 | `qrun/run.py <yml>` | 训练 + IC + TD0 策略回测，落 MLflow recorder | 现役 ★ |
| 生产/回放 | `scripts/intraday_production.py` | 盘中生产 6 步 + 治理 2 步 | 现役 ★ |
| 生产/回放 | `scripts/replay_intraday_shadow.py` | 生产链路的历史适配回放（方案 B 全模拟） | 现役 ★ |
| 重训治理 | `scripts/retrain.py` | 盘后滚动候选训练 + 门控 + 人工晋升 | 现役 |
| 验证研究 | `scripts/rolling_validate.py` | 19 任务滚动重训验证（vs Champion 单次训练） | 研究 |
| 验证研究 | `scripts/validate_factor_challengers.py` | 因子变体 × 4 窗口 A/B | 研究 |
| 验证研究 | `scripts/validate_xgb_purged_rolling_gate.py` | 19 步 embargo walk-forward 验证 XGB 门控 | 研究（未过准入） |
| 验证研究 | `scripts/validate_risk_overlay_purged.py` / `_quarters.py` | 风险叠加层验证 | 研究（未过准入） |
| 验证研究 | `scripts/validate_index_stage_joint_models.py` | 指数阶段 joint 模型验证 | 研究（未过准入） |
| 验证研究 | `scripts/validate_prediction_blend.py` | 预测混合验证 | 研究（未过准入） |
| 对照诊断 | `compare_models.py` / `compare_position_sizing.py` / `compare_drawdown_solutions.py` / `compare_anticrowd.py` | 模型/仓位/回撤/反拥挤对照 | 研究工具 |
| 对照诊断 | `stability_check.py` / `diag_drawdown_*.py` / `diagnose_*.py` / `cheap_falsify_reversal_tail.py` | 稳定性与归因诊断 | 研究工具 |
| 报告 | `scripts/make_report.py` | recorder → 13 个 plotly HTML 报告 | 现役 |
| legacy | `live_forward.py` / `live_catchup.py` / `realtime_signal.py` | 旧 P1 纸面跟踪/实时信号（已被 6 步链路取代） | 历史 |

> ⚠️ **无参数脚本会立即执行**：`rolling_validate`、`compare_models`、
> `compare_position_sizing`、`stability_check` 等没有 argparse，`--help` 无效，
> 调用即开始训练/分析（会写 mlruns 与 data/）。用前先看清源码窗口。

## 主干链 A：qrun 训练回测（13 秒全流程）

```bash
conda run -n qlib_ifind_beta python qrun/run.py qrun/workflow_minute_enhanced_tk10_nd8.yaml
```

内部执行链（`run.py` 加载 yml → 修正 limit_threshold list→tuple → qlib.init →
`task_train`）：

```text
DatasetH.prepare(DK_L/DK_I)
  └ MinuteEnhancedHandler：读 overlay 的 18 因子 day.bin + label 表达式
      → DropnaProcessor 整行剔除（缺任一因子即弃）
HFLGBModel.fit（train 42,122 行，binary，早停看 valid）
SignalRecord   → pred.pkl / label.pkl（test 5,776 行）
SigAnaRecord   → sig_analysis/{ic,ric}.pkl（逐日 IC / RankIC）
PortAnaRecord  → portfolio_analysis/*.pkl
  └ TopkDropoutStrategyTD0：shift=0，T 日信号 T 日成交
      topk=10 / n_drop=8 / hold_thresh=1 / 涨跌停禁交易
  └ Exchange 撮合：买 $price_941（09:41）/ 卖 $close（T+1）
      成本 0.05%/0.15%/min 5 元
  └ risk_analysis：年化 / IR / 最大回撤（基准 SH000300）
recorder.save_objects(config) → mlruns/<实验>/<recorder>/
```

### DK_L / DK_I 是什么

Qlib `DataHandlerLP` 的三个数据视图（`handler.py:53-55`）：

| Key | 全称 | 内容 | 谁在用 |
|---|---|---|---|
| `DK_R` | raw | 表达式引擎直出（含 NaN、label 未处理） | SigAnaRecord 对齐 label 时 |
| `DK_I` | infer | raw + **shared** 处理器 | 模型预测 / 盘中推理 |
| `DK_L` | learn | raw + shared + **learn** 处理器 | 训练 fit |

本项目 shared 只有一个 `DropnaProcessor(feature)`（缺任一因子的行整行剔除），
learn 处理器为空——因此 **DK_I 与 DK_L 输出完全相同**；label 永不做标准化，
IC 直接对原始收益计算。

### 数据样式与产物示例（recorder `f364dcb3` 实取）

**① 特征矩阵（DK_R 视角，SZ300164，18 列节选 4 + label）：**

| 日期 | startup_mom_1m | close_pos_5m | vol_vs_yest | overnight_gap | LABEL |
|---|---|---|---|---|---|
| 04-01 | 0.0038 | 0.8178 | 55.28 | -0.0125 | +0.0221 |
| 04-02 | -0.0044 | 0.0479 | 92.67 | +0.0307 | -0.0181 |
| 04-03 | -0.0032 | 0.8003 | 44.42 | -0.0139 | +0.0358 |

DropnaProcessor 的效果（test 段）：**6,200 原始行 → 5,776 有效行**（缺任一因子
的 424 行整行剔除，前视/数据事故护栏）。

**② pred.pkl / label.pkl（test 段 head5）：**

| datetime | instrument | score（pred） | LABEL（label） |
|---|---|---|---|
| 2026-04-01 | SH600066 | 0.4089 | +0.0092 |
| 2026-04-01 | SH600158 | 0.4570 | -0.0139 |
| 2026-04-01 | SH600184 | 0.4210 | -0.0776 |
| 2026-04-01 | SH600250 | 0.4353 | +0.0157 |
| 2026-04-01 | SH600268 | 0.4538 | -0.0055 |

注意行数差：label.pkl 是 DK_R 口径（6,200 行），pred 是 Dropna 后（5,776 行），
IC 计算按交集对齐——这就是上表"SigAnaRecord 用 DK_R"的体现。score 是 binary
模型的横截面分数，只看排序不看绝对值。

**③ sig_analysis/ic.pkl（62 个交易日逐日 IC，节选）：**

```text
2026-04-01  +0.1676    2026-04-07  +0.0937
2026-04-02  -0.0595    2026-04-08  -0.0737
2026-04-03  +0.0336    …
mean=0.0548  std=0.1083  →  ICIR=0.51（ic.pkl 的 mean/std 即 ICIR）
```

**④ portfolio_analysis/report_normal_1day.pkl（逐日组合报告，节选）：**

| 日期 | return | bench | cost | turnover |
|---|---|---|---|---|
| 04-01 | +0.0196 | +0.0171 | 0.0005 | 0.95 |
| 04-02 | -0.0149 | -0.0104 | 0.0015 | 1.52 |
| 04-03 | -0.0205 | -0.0085 | 0.0013 | 1.32 |

`return`=策略日收益（含成本）、`bench`=SH000300、`cost`=当日交易成本拖累、
`turnover`≈1.4 档对应 n_drop=8 的 80% 换手帽。risk_analysis 的年化/IR/回撤就是
对这些日序列做的汇总。

**⑤ recorder 产物树：**

```text
mlruns/156869948604814731/f364dcb3…/
├── artifacts/   params.pkl(模型) pred.pkl label.pkl
│                sig_analysis/{ic,ric}.pkl  portfolio_analysis/*.pkl
│                dataset/ task/ config/ code_*.txt（代码快照与 diff）
├── metrics/     IC  ICIR  Rank IC  Rank ICIR（四行数字）
├── params/      完整训练配置（从 yml 展开，可复现）
└── tags/
```

实测（2026-09-06 重跑）：IC 0.0548 / RankIC 0.0612；超额含成本年化 +148.2%、
IR 3.98、回撤 -12.93%；与冻结 Champion pred **逐位相同**。字段口径逐项见
[配置说明](configs.md)。

## 主干链 B：历史影子回放 replay_intraday_shadow.py

```bash
conda run -n qlib_ifind_beta python scripts/replay_intraday_shadow.py \
    --start 2026-06-29 --end 2026-07-02 \
    --output data/historical_shadow_replay_walkthrough_YYYYMMDD \
    [--initial-cash 1000000] [--allow-unreferenced-scores]
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--start/--end` | 2026-04-01 / 2026-07-02 | 回放区间（须有 universe 快照缓存） |
| `--output` | `data/historical_shadow_replay` | 输出根目录（**勿指向 62 日验收目录**） |
| `--initial-cash` | 1,000,000 | 初始现金 |
| `--allow-unreferenced-scores` | 关 | 允许超出 Champion pred.pkl 日期做前向影子评分；不开则无参照日直接报错（防"自说自话"） |

每个交易日的循环（同一套生产函数 `qlib_ifind_beta/live/intraday.py`，行情来自
1min bins 逐分钟重放）：

```text
① 冻结审计根   run/active_model_manifest.json + preflight.json
② 股池        universe_raw(100) → eligible(剔 ST) + positions_before（昨日持仓 hold_days+1）
③ K线重放     09:31–09:40 逐根幂等 upsert（校验 idempotent）→ factor_bars.parquet
④ 质量门禁    validate_factor_bars：恰好十根/覆盖率 → bar_quality.json（不足即 FAIL）
⑤ 特征组装    18 因子（分钟量口径分母）→ features.parquet（<80 只即 FAIL）
⑥ 成交辅助    09:41 bar → price_941/change_941
⑦ 打分+零漂移  冻结模型矩阵推理 → scores.csv；与 Champion pred.pkl 比对
              signal_parity.json（max_abs_diff>1e-12 即 FAIL）
⑧ 调仓决策    plan_topk_dropout：keep/sell/buy_ranked（n_drop≤8）
⑨ 方案B执行   sell_orders → sell_fills（跌停拦截）→ cash_after_sell
              → buy_orders（涨停拦截+顺延+等分现金取整）→ buy_fills → cash_after_buy
⑩ 对账收盘    positions_after_close + daily_reconciliation + NAV（收盘价估值）
```

指标口径（`_metrics`）：几何年化 `(1+total)^(252/n)-1`；成交假设 09:41 close 全量
成交、成本 0.05%/0.15%、无滑点——**方案 B（次日 09:41 卖），不等价于 Champion
回测（T+1 收盘卖）**，对照结果见生产设计 §6（年化差 -51.8pct）。62 日验收结论见
[回放日志](backtest-log/2026-07-21-intraday-shadow-replay.md)。

## 主干链 C：盘中生产 intraday_production.py（8 个子命令）

| 子命令 | 关键参数 | 时刻 | 产物 |
|---|---|---|---|
| `preflight` | `--date` | 08:50 | `run_manifest.json`、`preflight.json`（日历/模型/昨日对账门禁） |
| `universe` | `--date` | 09:00 | `universe_raw.csv`(100)、`universe_eligible.csv`、instruments 更新 |
| `collect-factor-bars` | `--date --minute HH:MM [--max-workers]` | 09:31–09:40 逐分钟 | 幂等 upsert `factor_bars.parquet`（严禁用"最近十根"覆盖窗口） |
| `score-and-plan-sells` | `--date --positions <csv> [--factor-confirmed]` | 09:40 后 | `features.parquet`、`scores.csv`、`rebalance_decision.json`、`sell_orders.csv` |
| `build-buy-orders` | `--date --sell-fills --cash-before [--factor-confirmed]` | 09:41 后 | `buy_orders.csv`（涨停拦截/顺延/真实现金驱动） |
| `reconcile` | `--date --expected-positions --broker-positions --expected-cash --broker-cash` | 15:05 | `daily_reconciliation.json`（不过则次日 preflight 拒绝启动） |
| `retrain` | `--asof` | 盘后 | Candidate recorder（不发布） |
| `promote-model` | `--manifest --approved-by` | 人工 | 晋升 + online tag（重跑全部防篡改校验） |

`--factor-confirmed`：公司行动/复权因子人工确认开关（默认 fail-closed，除权日
不静默沿用昨日 factor）。逐环节合同（S0–S10）见
[盘中生产设计](superpowers/specs/2026-07-20-intraday-production-signal-design.md)，
纸面跟踪全流程见[模拟盘流程](paper-trading.md)。

## 重训治理 retrain.py

```bash
conda run -n qlib_ifind_beta python scripts/retrain.py --test-start 2026-07-02
conda run -n qlib_ifind_beta python scripts/retrain.py \
    --promote-manifest data/model_governance/xxx.json --approved-by <姓名>
```

| 参数 | 说明 |
|---|---|
| `--test-start` | test 段起始日，默认上海时区今天；按日历精确反推窗口 |
| `--promote-manifest` | 只做晋升（配合 `--approved-by`），不训练 |

窗口算法：`train 90 / valid 20 / embargo 1 / test ≤20` 个交易日——embargo 一日
是因为 **label = T+1 收盘价**，T 日模型的验证标签最多只能用到 T-2；XGB 任务是
HFLGB 任务的深拷贝仅换 model 段。产物：两个 recorder + `ensemble_gate.pkl` +
`candidate_model_manifest.json`（状态 CANDIDATE，详见[模型说明](models.md)）。

## 验证研究脚本群（Codex 线，均未通过生产准入）

| 脚本 | 做什么 | 依赖 |
|---|---|---|
| `rolling_validate.py` | RollingGen(step=20) 生成 19 个滚动任务逐个训练，拼接全部 test 段 pred 统一算 IC/回测，对照 Champion 单次训练 | 无（即跑 19 次训练） |
| `validate_factor_challengers.py` | `--window {W1'26Q2,W2'25Q2,W3'25Q4,W4'24Q4}` × `--variant {champion18,pruned15,turnover19,…,xgb18,path21}` 的因子/模型变体 A/B，`--skip-backtest` 只算 IC | 无 |
| `validate_xgb_purged_rolling_gate.py` | 19 步 walk-forward 重训双模型，验证 XGB 门控在防泄漏口径下是否仍成立 | `data/xgb_rolling_gate_ab.json`（需先跑上游） |
| `validate_risk_overlay_purged.py` / `_quarters.py` | 风险叠加层（回撤控制）的分季与 purged 验证 | 前者需 `xgb_purged_rolling_gate_ab.json`；后者需 `rolling_90d_purged_*` 实验 |
| `validate_index_stage_joint_models.py` | 指数阶段 + 个股 joint 模型（joint TVT） | `xgb_purged_rolling_gate_ab.json` |
| `validate_prediction_blend.py` | 预测层混合权重验证 | `factor_challenger_ab` 实验 |

结论与门控细节统一见[验证与研究](validation.md)；已证伪方向不要重复投入。

## 对照 / 诊断 / 报告

- `compare_models.py`：LGBM / XGBoost / CatBoost / Linear 滚动对照（无参数，即跑即训）；
- `compare_position_sizing.py`：仓位层方案对比（§55：A 波动率目标 / B 滚动 IC
  择时 / C 基准趋势，输出 Calmar 对照表）；`stability_check.py`：三子期 Calmar
  稳定性 + 仓位分布（两者均为无参数分析脚本）；
- `diag_drawdown_*` / `diagnose_*` / `compare_drawdown_solutions.py` /
  `cheap_falsify_reversal_tail.py`：回撤归因、共振诊断等一次性研究工具，用前读
  源码确认窗口；
- `make_report.py`：`--recorder-id`（缺省取最新）→ `reports/` 下 13 个 plotly
  HTML（离线可看）。

## legacy 入口（勿用于生产）

| 脚本 | 参数 | 说明 |
|---|---|---|
| `live_forward.py` | `--date [--skip-universe] [--skip-materialize]` | 旧 P1 纸面跟踪日入口，已被 6 步链路取代 |
| `live_catchup.py` | `--start --end` | 旧模拟盘批量补跑 |
| `realtime_signal.py` | `--date [--topk] [--max-workers] [--dry-run]` | 旧单脚本实时信号；`--dry-run` 用历史 1min 数据，已被 intraday_production 取代 |
