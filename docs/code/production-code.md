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

### 六步输入输出示例（T = 2026-07-21）

**S1–S3 采集与质量门禁** —— `upsert_bars` 的 `incoming`（每股 10 行，列缺一不可）：

```text
         date      code bar_time   open   high    low  close  volume
0 2026-07-21  SZ300001    09:31  12.30  12.36  12.28  12.35  452100
...
9 2026-07-21  SZ300001    09:40  12.41  12.44  12.38  12.42  388000
```

`validate_factor_bars` 返回 `(complete, quality)`：

```python
complete = ["SZ300001", "SH600520", ...]         # 恰好集齐十根的股票
quality = {
    "universe_count": 104, "complete_bar_count": 102, "coverage": 0.9808,
    "missing": {"SZ300750": ["09:36", "09:37"]},  # 缺哪些 bar_time
    "bar_window": ["09:31", "09:40"],
    "status": "PASS",                             # 完整数 >= MIN_CANDIDATES(80)
}
```

**S4 特征矩阵** —— `assemble_features(bars, prev_volumes, daily_info, codes)`：

```python
prev_volumes = {"SZ300001": [3.1e8, 2.8e8, 3.5e8, 2.9e8]}  # T-1/T-2/T-3/T-5 全天分钟量
daily_info   = {"SZ300001": {"prev_close": 12.30, "prev_factor": 1.0,
                             "factor": 1.0, "factor_confirmed": True}}
# 输出：code + 18 因子列，列序 = MinuteEnhancedHandler.ENHANCED_FIELDS（模型合同）
#         code  startup_mom_1m  ...  vol_vs_yest  vol_vs_yest_t2  ...  overnight_gap
# 0  SZ300001          0.0041  ...         2.31            2.05  ...         0.0081
```

**S5 调仓决策** —— `plan_topk_dropout(scores, positions)`：

```python
# scores:    [code, score, limit_up]             # S4 矩阵过模型后的打分
# positions: [code, quantity, sellable_quantity, hold_days]
{"keep": ["SH600520", ...],        # 留仓
 "sell": ["SZ300750", ...],        # 调出（先卖）
 "blocked_sell": [],               # 想卖但 T+1/可用不足被拦
 "buy_ranked": ["SZ300001", ...],  # 调入候选，按分数排序
 "topk": 10, "n_drop": 8, "hold_thresh": 1}
```

`build_sell_orders` → 卖单：

```text
  client_order_id      code action  quantity    execution_policy           reason
0    20260721-S-001  SZ300750   SELL      2000  SCHEME_B_UNVALIDATED  dropout_bottom
```

**S6–S7 两阶段执行** —— `apply_sell_fills` 消费券商成交回报（先卖）：

```python
# fills: [client_order_id, code, filled_quantity, average_price, fee]
(positions_after_sell, {"cash_after_sell": 1081234.56, "status": "PASS"})
```

`build_execution_snapshot`（09:41 bar 快照）→ `build_buy_orders`（涨停拦截 → 顺延 → 等分 → 100 股取整，后买）：

```text
# snapshot:       code  price_941  change_941
#             0  SZ300001      12.43      0.0106

# buy orders:
  client_order_id  rank      code action  quantity  reference_price  estimated_amount
0    20260721-B-001     1  SZ300001    BUY      8000            12.43           99440.0
```

`apply_buy_fills` → `(positions, {"cash_after_buy": ..., "status": "PASS"})`；
新买仓位 `hold_days=0`、`sellable_quantity` 不变（T+1 锁定）。

**S8 收盘对账** —— `reconcile_positions`：

```python
{"status": "PASS",                  # 持仓零错位且 |cash_difference| <= 1 元
 "position_mismatches": [],         # 否则 [{"code", "expected", "actual"}]
 "cash_difference": 0.42,           # 券商现金 - 本地推算
 "actual_position_count": 10}
```

**治理** —— `make_active_manifest`：

```python
{"trade_date": "2026-07-21", "experiment": "minute_enhanced_tk10_nd8",
 "recorder_id": "93d435e0ef20464784553949eb3859a5", "artifact": "params.pkl",
 "model_sha256": "9f2c…", "feature_schema_sha256": "a1b0…",
 "qlib_version": "0.9.7", "status": "LOCKED",
 "generated_at": "2026-07-21T08:55:00+08:00"}
```

CLI 子命令 → `data/production_signals/<date>/` 落盘文件的对应关系：

```text
preflight            → active_model_manifest.json / preflight.json / run_manifest.json
universe             → universe_raw.csv / universe_eligible.csv / universe_quality.json
collect-factor-bars  → factor_bars.parquet / bar_collection_status.csv
score-and-plan-sells → features.parquet / scores.csv / rebalance_decision.json
                       / sell_orders.csv（S4+S5）
build-buy-orders     → sell_fills.csv / positions_after_sell.csv / cash_after_sell.json
                       / execution_bar_0941.parquet / buy_orders.csv（S6-S7）
reconcile            → positions_after_close.csv / daily_reconciliation.json（S8）
```

## live/historical_replay.py —— 回放适配器（163 行）

`HistoricalReplaySource`：`bars(date, codes)`（11 根）/ `previous_volumes`
（T-k 全天分钟量）/ `daily_info`（T-1 close/factor，日频 bin 有洞时分钟降级）/
`valuation_close`。用 binio 直读绕过表达式引擎（性能），被
`replay_intraday_shadow.py` 消费——**生产函数 + 历史数据源 = 全链路回放**。

四个访问器的输出形状与上面六步的输入一一对应：

```python
src = HistoricalReplaySource()

src.bars("2026-07-21", codes)
# → [date, code, bar_time, open, high, low, close, volume, amount,
#    fetch_time, source="historical_1min"]，每股 11 行（10 因子 bar + 09:41 执行 bar）

src.previous_volumes("2026-07-21", codes)
# → {"SZ300001": [T-1, T-2, T-3, T-5 全天分钟量]}

src.daily_info("2026-07-21", codes)
# → {"SZ300001": {"prev_close": 12.30, "prev_factor": 1.0, "open": 12.31,
#                 "factor": 1.0, "factor_confirmed": True}}
#    日频 bin 有洞时走分钟降级，多一个 "historical_source": "minute_fallback"

src.valuation_close("2026-07-21", codes)
# → {"SZ300001": 12.55}   # 后复权收盘，供收盘估值
```

## realtime/ —— 盘中在线路径

- `data_fetch.py`（465 行）：kline-fetcher 服务适配（`KLINE_API_BASE_URL`）。
  `fetch_realtime_bars(code, count=11)` 取最近 11 根——调用方必须校验
  `bar_time`（09:41 后"最近十根"会漂移成 09:32-09:41，生产严禁直接覆盖窗口）；
- `signal.py`（277 行）：`_bars_to_arrays → _compute_all_factors`（调
  minute_factors）→ `_predict_in_memory`（load_model_bundle + 矩阵推理，
  `use_online` 开关走冻结 Champion 或滚动候选）。**零漂移合同的实现在这**：
  同一 18 因子输入，与 Champion 历史 pred 逐位一致。

两端的输入输出：

```python
fetch_realtime_bars("SZ300001", count=11)
# → [{"date": "2026-07-21", "time": "09:31", "open": 12.30, "high": 12.36,
#     "low": 12.28, "close": 12.35, "volume": 452100, "amount": 5580435.0},
#    ... 共 11 根，截至最新分钟]

generate_realtime_signal("2026-07-21")
# → {"date": "2026-07-21", "n_candidates": 102,
#    "candidates": [{"code": "SZ300001", "score": 0.83,
#                    "limit_up": 0.10, "limit_down": -0.10,
#                    "change_941": 0.0106, "price_941": 12.43}, ...],
#    "topk": [... 10 只，已剔除 change_941 >= limit_up 的涨停股 ...],
#    "model_source": "rolling_hflgb",   # 或 frozen / rolling_ensemble
#    "ensemble_weight": 0.0}
```

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
