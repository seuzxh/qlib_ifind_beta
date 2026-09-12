---
layout: default
title: 生产代码与脚本
parent: "代码导读"
nav_order: 2
---

# 生产代码与脚本（live/ · realtime/ · scripts/ · qrun/）

## live/intraday.py —— 盘中六步的全部纯函数（412 行）★

全仓最核心的生产模块。CLI 的 8 个子命令只是它的薄封装。按数据流顺序：

| 层 | 函数 | 干什么 |
|---|---|---|
| 基础设施 | `DayPaths`（单日目录）/ `write_json/csv/parquet`（原子写 + SHA256）/ `sha256_file` / `schema_sha256` / `calendar_gate`（交易日门禁） | 审计链的物理层 |
| S1-S3 | `upsert_bars`（(date,code,bar_time) 幂等）/ `validate_factor_bars`（恰好十根门禁） | K 线采集与质量 |
| S4 | `assemble_features`（调 minute_factors 共享公式 + 分钟量分母） | 18 因子矩阵 |
| S5 | `plan_topk_dropout`（keep/sell/buy_ranked，Qlib 语义）/ `build_sell_orders` | 打分后调仓决策 |
| S6-S7 | `apply_sell_fills`（真实现金）/ `build_execution_snapshot`（09:41 bar）/ `build_buy_orders`（涨停拦截+顺延+等分取整）/ `apply_buy_fills` | 方案 B 两阶段 |
| S8 | `reconcile_positions`（订单/成交/持仓/现金四对账） | 收盘闭环 |
| 治理 | `make_active_manifest`（recorder 引用 + 校验和锁定） | 模型不可变引用 |

不变量：失败 fail-closed（不降级、不猜）；一切产物带上游 SHA256。

## live/historical_replay.py —— 回放适配器（163 行）

`HistoricalReplaySource`：`bars(date, codes)`（11 根）/ `previous_volumes`
（T-k 全天分钟量）/ `daily_info`（T-1 close/factor，日频 bin 有洞时分钟降级）/
`valuation_close`。用 binio 直读绕过表达式引擎（性能），被
`replay_intraday_shadow.py` 消费——**生产函数 + 历史数据源 = 全链路回放**。

## realtime/ —— 盘中在线路径

- `data_fetch.py`（465 行）：kline-fetcher 服务适配（`KLINE_API_BASE_URL`）。
  `fetch_realtime_bars(code, count=11)` 取最近 11 根——调用方必须校验
  `bar_time`（09:41 后"最近十根"会漂移成 09:32-09:41，生产严禁直接覆盖窗口）；
- `signal.py`（277 行）：`_bars_to_arrays → _compute_all_factors`（调
  minute_factors）→ `_predict_in_memory`（load_model_bundle + 矩阵推理，
  `use_online` 开关走冻结 Champion 或滚动候选）。**零漂移合同的实现在这**：
  同一 18 因子输入，与 Champion 历史 pred 逐位一致。

## scripts/ —— 12 个入口（按用途）

| 组 | 脚本 | 一句话 |
|---|---|---|
| 数据 | `build_overlay.py`（+ `-m`） | 端到端重建 overlay（股池/涨跌停/因子/label 腿） |
| 数据 | `materialize_minute.py` / `update_universe.py`（08:30 cron，配 `cron_update_universe.sh`） | 只重物化 / 只更名单 |
| 训练 | `qrun/run.py` | yml → task_train（list→tuple 修复 + MLFLOW env） |
| 训练 | `retrain.py` | 滚动候选（90/20/embargo1/test20）+ 门控 + `--promote-manifest` 人工晋升 |
| 训练 | `rolling_validate.py` / `compare_models.py` / `validate_factor_challengers.py` / `validate_gap_adjusted.py` / `validate_xgb_purged_rolling_gate.py` | 研究验证（结论见 validation.md，多数为已证伪存档） |
| 生产 | `intraday_production.py` | 六步 + 治理 CLI（子命令参数表见运维手册） |
| 生产 | `replay_intraday_shadow.py` | 历史影子回放 |
| 报告 | `make_report.py` | recorder → 13 个 plotly HTML |

## qrun/ —— 唯一现役配置

`workflow_minute_enhanced_tk10_nd8.yaml`（Champion 合同）+ `run.py`
（两个本机坑的修复器）。字段逐项解释见[配置说明](../configs.md)。

## tests/ —— 10 文件 76 用例

离线可跑、网络全 mock。改动的最小验证矩阵：

```text
改 minute_factors.py   → test_minute_factors + test_materialize_minute + test_realtime_signal
改 materialize*.py     → test_materialize_minute（22 bin 完整性）
改 handler/特征清单    → test_highbeta_handler + 全量（影响面大）
改 intraday.py         → test_intraday_production
改门控/混合            → test_model_ensemble
改 retrain 窗口        → test_retrain
```

## 新人第一周建议

1. 跑通 [全流程导读](../pipeline-walkthrough.md) 的四条命令（overlay → qrun →
   replay → pytest），对照每步产物；
2. 精读 `minute_factors.py`（131 行）+ `live/intraday.py`（412 行）——
   一个定义"算什么"，一个定义"怎么落地"；
3. 改一个小东西（比如给 factors.md 修个错字）走完 PR 流程；
4. 红线（AGENTS.md）：无未来数据、不自动下单、不自动晋升、token 不入库、
   运行资产不入 Git。
