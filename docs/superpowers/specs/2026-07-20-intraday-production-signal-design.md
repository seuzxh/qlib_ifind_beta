---
layout: default
title: "盘中生产设计合同"
parent: "生产基线"
nav_order: 1
---

# 883926 盘中生产信号、人工下单与滚动训练方案（Qlib 对齐版）

> 日期：2026-07-20  
> 状态：第一阶段已实施；2026-07-21 已完成 62 日历史影子回放，真实盘中行情、人工券商 CSV 和至少 5 个交易日连续影子运行仍待验证  
> Qlib 运行基线：本项目已安装 `qlib 0.9.7`，生产必须锁定该版本；`latest` Guide 仅用于前向兼容审查  
> 生产 Champion：`minute_enhanced_tk10_nd8` / recorder `93d435e0ef20464784553949eb3859a5`

## 1. Review 结论与设计边界

本方案采用“**Qlib 研究与决策核心 + 外部盘中数据和人工执行适配层**”，不另建一套与 Qlib 平行的模型、调仓或模型注册体系。

Qlib 核心负责：

- `DataHandlerLP / DatasetH`：训练、验证、历史推理的数据处理和时间切分。
- `HFLGBModel / XGBModel`：训练与预测。
- `Recorder / task_train / OnlineToolR`：模型、任务、指标和在线标签管理。
- `TopkDropoutStrategyTD0 / Order / Position / Exchange`：选股、换仓和交易约束语义。
- `SignalRecord / SigAnaRecord / PortAnaRecord`：研究期预测与回测记录。

外部生产适配层负责：

- 883926 T 日盘前成分快照。
- 09:31–09:40 十根闭合分钟 K 线的逐分钟采集。
- 09:41 成交参考行情的独立采集。
- 最新交易日的内存特征适配、CSV 导出、人工成交回填和对账。
- Champion/Candidate 人工审批 manifest；manifest 引用 Qlib Recorder，不复制或替代模型。

官方依据：

- [Qlib Workflow Guide](https://qlib.readthedocs.io/en/latest/component/workflow.html)：标准工作流包含数据加载/处理/切分、模型训练/推理/保存、信号分析和回测。
- [DataHandlerLP Guide](https://qlib.readthedocs.io/en/latest/component/data.html)：区分原始数据 `DK_R`、推理数据 `DK_I` 和学习数据 `DK_L`。
- [Recorder Guide](https://qlib.readthedocs.io/en/latest/component/recorder.html)：模型、预测、指标和配置应保存在 Recorder artifact 中。
- [TopkDropoutStrategy Guide](https://qlib.readthedocs.io/en/latest/component/strategy.html)：Topk/Drop 决定持仓数量和每日卖出/买入数量。
- [Online Serving Guide](https://qlib.readthedocs.io/en/latest/component/online.html)：通过 Trainer、OnlineStrategy、OnlineTool 管理滚动任务和在线模型；行情更新与真实订单执行仍由使用方负责。
- [Microsoft/Qlib GitHub](https://github.com/microsoft/qlib)：生产升级 Qlib 前，以固定版本源码和本项目回归测试为准，不直接跟随 `main`。
- GitHub 核心源码：[DataHandlerLP](https://github.com/microsoft/qlib/blob/main/qlib/data/dataset/handler.py)、[DatasetH](https://github.com/microsoft/qlib/blob/main/qlib/data/dataset/__init__.py)、[TopkDropoutStrategy](https://github.com/microsoft/qlib/blob/main/qlib/contrib/strategy/signal_strategy.py)、[Exchange](https://github.com/microsoft/qlib/blob/main/qlib/backtest/exchange.py)、[RollingGen](https://github.com/microsoft/qlib/blob/main/qlib/workflow/task/gen.py)、[Trainer](https://github.com/microsoft/qlib/blob/main/qlib/model/trainer.py)、[OnlineManager](https://github.com/microsoft/qlib/blob/main/qlib/workflow/online/manager.py)。

### 1.1 两个必须区分的兼容性

1. **模型信号兼容**：相同 18 维特征输入时，盘中适配层输出的 score、排序和 Top10 必须与 Champion 历史推理一致。
2. **交易收益兼容**：真实买卖时间、价格、涨跌停和成本必须与回测 Exchange 一致。

目前模型信号路径已有历史零漂移证据；交易收益路径尚不等价，因为用户选择“上午先卖旧仓，再买新仓”，而 Champion 回测卖出价为 `$close`。该差异必须单独回测，不能只凭相同 Top10 宣称复现 Champion 收益。

## 2. 冻结 Champion 契约

### 2.1 模型、数据和策略口径

| 项目 | 冻结值 | 权威来源 |
|---|---|---|
| Qlib provider | `data/qlib_root` | `qrun/workflow_minute_enhanced_tk10_nd8.yaml` |
| 实验名 | `minute_enhanced_tk10_nd8` | 同上、`qlib_ifind_beta/config.py` |
| Recorder ID | `93d435e0ef20464784553949eb3859a5` | `qlib_ifind_beta/config.py` |
| 模型本体 | `params.pkl`，`HFLGBModel(binary)` 内含 LightGBM Booster | `mlruns/156869948604814731/<recorder>/artifacts/params.pkl` |
| 股池 | `highbeta883926`，T 日盘前成分 | `data/qlib_root/instruments/highbeta883926.txt` |
| 模型特征 | 固定 18 列、固定顺序 | `MinuteEnhancedHandler.ENHANCED_FIELDS` |
| Label | `Ref($close, -1) / $price_941 - 1` | Champion YAML / `CHAMPION_LABEL_EXPR` |
| 策略 | `TopkDropoutStrategyTD0` | Champion YAML / `qlib_ifind_beta/td0_strategy.py` |
| 参数 | `topk=10, n_drop=8, hold_thresh=1` | Champion YAML |
| 买入价 | `$price_941` | Champion Exchange |
| 卖出价 | `$close` | Champion Exchange |
| 成本 | open 0.05%、close 0.15%、min 5 元 | Champion Exchange |
| 涨跌停 | 买：`$change_941 >= $limit_up`；卖：`$change <= $limit_down` | Champion Exchange |

### 2.2 18 维模型特征

固定列顺序：

```text
startup_mom_1m, startup_mom_3m, startup_mom_5m, startup_total,
accel_1m, accel_3m, accel_5m,
close_pos_1m, close_pos_3m, close_pos_5m,
vol_ratio_1m, vol_ratio_3m, vol_ratio_5m, vol_vs_yest,
vol_vs_yest_t2, vol_vs_yest_t3, vol_vs_yest_t5, overnight_gap
```

辅助字段不进入模型：

- `price_941`：09:41 闭合 K 线 close，Champion 买入成交参考价。
- `change_941`：09:41 价格相对 T-1 不复权收盘的涨跌幅，用于涨停拦截。
- `limit_up / limit_down`：按证券板块计算的交易限制。

### 2.3 十根因子 K 线与第十一根辅助 K 线

严格定义：

```text
因子窗口：09:31, 09:32, ..., 09:40，共 10 根闭合 K 线
成交辅助：09:41 闭合 K 线，共 1 根；不参与 18 维特征
```

因此流程不是一次性把十一根都称为“因子数据”：

1. 09:31–09:40 每分钟分别采集并幂等落盘。
2. 09:40 闭合后即可计算 18 维特征和模型 score。
3. 09:41 闭合后再取得 `price_941/change_941`，完成涨停过滤和数量计算。

当前 `realtime/signal.py` 和 `live/production.py` 一次读取 11 根后统一计算，是已有实现；目标生产编排要拆成上述两个数据契约，但必须复用同一套因子公式。

## 3. 已有数据与文件来源目录

| 数据/对象 | 已有来源 | 使用方式 | 所有权 |
|---|---|---|---|
| 日交易日历 | `/home/zxh/.qlib/qlib_data/cn_data/calendars/day.txt` | 判断 T、T-1、滚动窗口 | 外部只读 |
| 分钟交易日历 | `/home/zxh/.qlib/qlib_data/cn_data_1min/calendars/1min.txt` | 映射日期和 240 个分钟 slot | 外部只读 |
| 历史日频 OHLCVF | `/home/zxh/.qlib/qlib_data/cn_data/features/<code>/` | T-1 close/factor、回测 | 外部只读 |
| 历史 1 分钟 OHLCVVWAP | `/home/zxh/.qlib/qlib_data/cn_data_1min/features/<code>/` | T-k 全天分钟量、历史重放 | 外部只读 |
| 成分快照缓存 | `data/universe_snapshots.csv` | 保存 p03473 每日原始快照 | 项目数据 |
| Qlib 时变 instruments | `data/qlib_root/instruments/highbeta883926.txt` | Dataset/Strategy 的 T 日股池 | 项目数据 |
| 日频物化特征 | `data/qlib_root/features/<code>/*.day.bin` | 历史 DatasetH、训练与回测 | 项目数据 |
| Champion 任务 | `mlruns/156869948604814731/93d.../artifacts/task` | 复核模型、Handler、segments | Qlib Recorder |
| Champion 模型 | 同 recorder 下 `artifacts/params.pkl` | 生产加载模型本体 | Qlib Recorder |
| Champion 历史预测/标签 | 同 recorder 下 `pred.pkl`、`label.pkl` | 零漂移和审计 | Qlib Recorder |
| 滚动 HFLGB | `minute_enhanced_rolling` Recorder 的 `params.pkl` | Candidate/online 模型 | Qlib Recorder |
| 滚动 XGB | `minute_enhanced_rolling_xgb` Recorder 的 `params.pkl` | 25% 门控候选 | Qlib Recorder |
| 滚动门控 | HFLGB Recorder 的 `ensemble_gate.pkl` | 验证指标、权重、test 段 | Qlib Recorder |
| 当日生产目录 | `data/production_signals/YYYY-MM-DD/` | 单日不可变输入/输出 | 目标生产数据 |

凭证只通过现有配置或环境变量读取，不写入任何 CSV/JSON：

- iFinD token：`QLIB_IFIND_TOKEN_FILE`，默认 `/home/zxh/qlib_data/.ifind_token`。
- 分钟服务：现有 `kline-fetcher` 适配器及 `KLINE_API_BASE_URL`。

## 4. 总体数据流和时间表

```text
Qlib日历 + T-1历史数据 + Champion Recorder + 昨日券商持仓
                         ↓
                   盘前运行快照
                         ↓
iFinD p03473(T) → 原始100只成分 → eligibility → 当日可评分股池
                         ↓
09:31…09:40 逐分钟闭合采集 → 十根因子K线 → 18维 feature matrix
                         ↓
Champion params.pkl → score Series → TopkDropoutTD0 股票选择/卖出集合
                         ↓
人工上午卖出 → sell fills → 可用现金和实际剩余持仓
                         ↓
09:41辅助行情 → 涨停拦截/顺延 → 买入数量 → buy CSV
                         ↓
人工买入 → 成交回报 → 收盘持仓/现金对账
                         ↓
盘后更新Qlib数据 → T-1 label完整 → rolling task_train → Candidate Recorder
                         ↓
验证报告 + candidate manifest → 人工批准 → T+1 active manifest
```

| 时间 | 环节 | 核心输出 |
|---|---|---|
| 08:50 | S0 盘前检查和运行冻结 | `run_manifest.json`、`preflight.json` |
| 09:00 | S1 T 日成分快照 | `universe_raw.csv`、`universe_eligible.csv` |
| 09:20 | S2 持仓/模型锁定 | `positions_before.csv`、`active_model_manifest.json` |
| 09:29 | 启动采集 | `collector.lock` |
| 09:31–09:40 | S3 十根因子 K 线逐分钟采集 | `factor_bars.parquet`、`bar_collection_status.csv` |
| 09:40:02 起 | S4 校验与特征组装 | `bar_quality.json`、`features.parquet` |
| 09:40 后 | S5 模型评分与调仓选择 | `scores.csv`、`rebalance_decision.json`、`sell_orders.csv` |
| 上午卖出后 | S6 卖出回填 | `sell_fills.csv`、`cash_after_sell.json` |
| 09:41 闭合后 | S7 辅助行情、买单计算 | `execution_bar_0941.parquet`、`buy_orders.csv` |
| 15:05 | S8 收盘对账 | `positions_after_close.csv`、`daily_reconciliation.json` |
| 15:40 后 | S9 数据更新和训练 | Qlib Recorder artifacts、`candidate_model_manifest.json` |
| 当晚 | S10 验证和人工晋级 | `validation_report.json`、下一交易日 active manifest |

## 5. 逐环节输入、组装、来源与输出

### S0. 盘前检查和单日运行冻结

**Qlib 对应层**：Provider/calendar、Recorder；外部运行治理。

| 类别 | 内容 |
|---|---|
| 输入信息 | 目标日期 T；Qlib 日历；昨日对账状态；Champion/active recorder；代码版本；行情/iFinD 健康状态 |
| 已有来源 | `DAY_CAL`；`daily_reconciliation.json`；MLflow/Recorder；`qlib_ifind_beta/config.py`；行情服务健康接口 |
| 数据组装 | 计算 T-1/T-2/T-3/T-5；加载 `params.pkl` 并核对对象类型；计算 artifact SHA256、18 列 schema hash、代码 commit；检查昨日持仓已对账；建立单日互斥锁 |
| 输出信息 | `run_manifest.json`、`preflight.json`、`collector.lock` |

`run_manifest.json` 是当日审计根，不是模型本体：

```json
{
  "trade_date": "2026-07-21",
  "qlib_version": "0.9.7",
  "experiment": "minute_enhanced_tk10_nd8",
  "recorder_id": "93d435e0ef20464784553949eb3859a5",
  "model_artifact": "params.pkl",
  "model_sha256": "...",
  "feature_schema_sha256": "...",
  "strategy": {"topk": 10, "n_drop": 8, "hold_thresh": 1},
  "status": "LOCKED"
}
```

失败条件：非交易日、模型不可加载、schema/checksum 不一致、昨日对账未通过、存在另一份不同 run manifest。失败后只允许人工处置旧仓，不生成新买单。

### S1. 获取并冻结 T 日 883926 成分快照

**Qlib 对应层**：动态 instruments；外部 DataLoader 上游。

| 类别 | 内容 |
|---|---|
| 输入信息 | `883926.TI`、目标日期 T、专用 iFinD 成分快照账号 |
| 已有来源 | `universe.fetch_constituents()`；iFinD `data_pool/p03473`；历史缓存 `data/universe_snapshots.csv` |
| 数据组装 | 保存接口原始 100 行；`000536.SZ → SZ000536`；验证日期、100 行、唯一代码；单独计算 `eligible`，不得覆盖原始集合；将 T 日成员更新为 Qlib `highbeta883926` instruments |
| 输出信息 | `universe_raw.csv`、`universe_eligible.csv`、`universe_quality.json`；持久缓存和 instruments 更新 |

最小 schema：

```csv
date,code_ifind,code_qlib,name,eligible,exclude_reason,source
2026-07-21,300164.SZ,SZ300164,通源石油,true,,ifind_p03473
```

过滤只影响“可买入”，不能删除原始成分记录。旧持仓即使已不在 T 日股池，也必须进入 S5 的持仓处理。

失败条件：快照日期不是 T、原始数量不等于 100、代码重复、使用 T-1 快照冒充 T 日快照。

### S2. 冻结真实持仓、现金和 active model

**Qlib 对应层**：`Position`、Recorder/OnlineTool；外部券商状态适配。

| 类别 | 内容 |
|---|---|
| 输入信息 | 券商盘前持仓、可用/冻结现金、未完成订单；active model manifest |
| 已有来源 | 第一阶段人工导出的券商 CSV；上一日 `positions_after_close.csv`；Qlib Recorder |
| 数据组装 | 统一 Qlib code；核对数量、可卖数量、成本价、冻结量；将真实持仓转换成 Qlib-compatible Position snapshot；核对 manifest 指向的 Recorder 和 `params.pkl` |
| 输出信息 | `positions_before.csv`、`cash_before.json`、`open_orders_before.csv`、冻结的 `active_model_manifest.json` |

`positions_before.csv` 至少包含：

```csv
code,quantity,sellable_quantity,cost_price,market_value,source_snapshot_time
SH600032,2000,2000,7.10,14600,2026-07-21T09:18:00+08:00
```

第一阶段 active model 固定为 frozen Champion。滚动模型只有经过 S10 人工批准，才能成为下一交易日的 active；盘中不得更换。

### S3. 09:31–09:40 十根因子 K 线逐分钟采集

**Qlib 对应层**：外部实时 DataLoader；历史落盘后映射到 Qlib 1min provider。

| 类别 | 内容 |
|---|---|
| 输入信息 | `universe_eligible.csv` 中全部待评分代码；目标分钟 `09:31…09:40`；实时分钟行情 |
| 已有来源 | `qlib_ifind_beta/realtime/data_fetch.py`；kline-fetcher；历史对照 `/home/zxh/.qlib/qlib_data/cn_data_1min` |
| 数据组装 | 每分钟闭合后并发采集当分钟；只接受 T 日和指定 `bar_time`；校验 OHLC、volume/amount；按 `(date, code, bar_time)` 幂等 upsert；保存 fetch_time、请求次数和来源 |
| 输出信息 | 增量 `factor_bars.parquet`、`bar_collection_status.csv`、原始响应审计日志 |

每一分钟的操作是独立的：

```text
09:31:02 → 只整理 09:31
09:32:02 → 只整理 09:32
...
09:40:02 → 只整理 09:40
```

`factor_bars.parquet` 逻辑 schema：

```csv
date,code,bar_time,open,high,low,close,volume,amount,vwap,fetch_time,attempt,source
2026-07-21,SZ300164,09:31,68.10,68.50,67.90,68.42,135000,9216450,68.27,09:31:03.120,1,kline
```

禁止使用“请求时最近十根”直接覆盖目标窗口，因为 09:41 后最近十根会变成 09:32–09:41。

### S4. 十根 K 线质量检查、历史上下文和 18 维特征组装

**Qlib 对应层**：`DataHandlerLP` 的 raw→infer 处理语义；最新日采用受控内存适配器。

| 类别 | 内容 |
|---|---|
| 输入信息 | `factor_bars.parquet`；T-1/T-2/T-3/T-5 全天分钟量；T-1 close/factor；T 日首根 open；T 日复权因子或除权除息状态 |
| 已有来源 | `minute_factors.compute_day_factors()`；`realtime.signal._compute_all_factors()`；1min bins；日频 close/factor bins |
| 数据组装 | 检查每股时间集合严格为 09:31–09:40；缺哪根只补哪根；计算 14 个基础分钟因子、3 个历史量比和 `overnight_gap`；inf→NaN；严格按 `ENHANCED_FIELDS` 排列；执行与 `DropnaProcessor(feature)` 等价的整行剔除 |
| 输出信息 | `bar_quality.json`、`features.parquet`、`factor_quality.csv`、`feature_schema.json` |

历史上下文必须同为分钟量口径：

```text
vol_vs_yest_tk = T日09:31–09:40成交量总和 / (T-k日全天分钟成交量 / 240)
k ∈ {1, 2, 3, 5}
```

不能用日频 `volume.day.bin` 代替分钟成交量总和；项目已有审计表明两者单位/复权口径不一致。

`overnight_gap/change_941` 必须使用不复权价格。当前实时代码在 T 日 factor 尚未进入日频 bin 时默认 `factor[T]=factor[T-1]`；普通交易日成立，但除权除息日可能错误。生产版必须接入 T 日公司行动/复权因子检查：能取得 T 日 factor 时使用真实值；无法确认且证券存在公司行动时，将该股票标记为 `feature_invalid=corporate_action_factor_unknown`，不能静默沿用昨日 factor。

特征输出建议使用 MultiIndex 语义：

```text
index   = (datetime=T, instrument=code)
columns = 18 个无 `$` 的固定字段，另保存带 `$` 的 Qlib 映射
```

质量门禁：原始 100 只中有效特征少于 80 只、字段顺序/hash 不一致、任一完整股票不是恰好十根，都不生成新买单。

实现前必须先消除公式重复：14 因子已共享 `compute_day_factors`，4 个 extra 仍在实时和物化路径分别实现，应提取到同一纯函数。

### S5. 加载 Champion、评分并生成 Qlib-compatible 调仓选择

**Qlib 对应层**：Model/Recorder、Signal、`TopkDropoutStrategyTD0`、Position/Order。

| 类别 | 内容 |
|---|---|
| 输入信息 | 18 维 `features.parquet`；active manifest；`positions_before.csv`；T 日可交易状态 |
| 已有来源 | Champion `params.pkl`；`model_ensemble.predict_feature_matrix()`；`td0_strategy.py`；Champion YAML |
| 数据组装 | 从 Recorder 加载而不是复制模型；矩阵预测并生成按 score 降序的 Series；把真实持仓映射为 Position；调用与 Qlib TopkDropout 相同的选股/Drop 语义；保留完整候选和旧仓 score；最多卖 8 只、目标最多 10 只、最低持有 1 日 |
| 输出信息 | `scores.csv`、`rebalance_decision.json`、`sell_orders.csv`、`target_members_preliminary.csv` |

`scores.csv` 保存全部有效股票：

```csv
rank,code,score,is_current_holding,in_universe,feature_valid
1,SZ300164,0.49497,false,true,true
```

`rebalance_decision.json` 必须保存策略输入和结果：

```json
{
  "strategy_class": "TopkDropoutStrategyTD0",
  "topk": 10,
  "n_drop": 8,
  "hold_thresh": 1,
  "keep": ["SH600032"],
  "planned_sell": ["SH600111", "SZ002222"],
  "planned_buy_ranked": ["SZ300164", "SZ300040"]
}
```

不得另写一个只按“今日 Top10 减昨日 Top10”的简化算法。应从 Qlib `TradeDecisionWO/Order` 或与其共享的纯调仓内核导出 CSV；否则停牌、无法卖出、持有期和现金语义会漂移。

盘中矩阵推理绕过了标准 `DatasetH.model.predict(dataset)`，因此它只能作为明确的兼容边界。每次模型、特征或 Qlib 版本变化都必须用历史日期验证：特征 max diff、score max diff `<1e-6`。Top10 必须在相同股池和确定性二级排序下完全相同；2026-07-21 回放发现同分候选会受输入顺序影响，二级排序修复前需单独披露 Top10 集合差异。

### S6. 人工卖出和真实成交回填

**Qlib 对应层**：Order/Position/Exchange 的真实执行适配。

| 类别 | 内容 |
|---|---|
| 输入信息 | `sell_orders.csv`；券商可卖量；人工成交回报；盘前现金 |
| 已有来源 | S5 Qlib-compatible sell Order；人工导出的券商 fill CSV |
| 数据组装 | 核对 code/direction/order quantity；按真实 filled quantity、average price、fee 更新现金；未成交/部分成交剩余持仓继续占槽位；保存人工修改原因 |
| 输出信息 | `sell_fills.csv`、`positions_after_sell.csv`、`cash_after_sell.json` |

```csv
client_order_id,code,requested_quantity,filled_quantity,average_price,fee,status,fill_time
20260721-S-001,SH600111,1200,1200,12.50,22.50,FILLED,09:40:35
```

可用现金：

```text
cash_after_sell = cash_before + Σ(filled_quantity × average_price - fee)
```

不得用计划卖出金额生成买单。

### S7. 09:41 辅助行情、涨停拦截和买单组装

**Qlib 对应层**：Exchange buy price、limit threshold、trade unit；外部 CSV exporter。

| 类别 | 内容 |
|---|---|
| 输入信息 | S5 候选顺序；S6 实际剩余持仓和现金；09:41 闭合辅助 K 线；T-1 raw close/factor；交易单位和成本 |
| 已有来源 | kline-fetcher；`materialize_minute.py` 的 `price_941/change_941` 公式；`materialize.board_limit()`；Champion Exchange 成本 |
| 数据组装 | 独立采集 09:41；计算 `price_941/change_941`；涨停候选不买并按 score 顺延；先扣未卖旧仓占用槽位；新增槽位间等分真实可用资金；考虑 open_cost/min_cost/safety buffer；按 100 股向下取整 |
| 输出信息 | `execution_bar_0941.parquet`、`execution_quality.json`、`reserve_candidates.csv`、`buy_orders.csv` |

```csv
rank,code,action,quantity,reference_price,max_price,estimated_amount,reason
1,SZ300164,BUY,200,68.97,69.31,13794,topk_dropout_fill
```

限制：模型 score 可在 09:40 后产生，但与 Champion 完全一致的 `$price_941` 必须等待 09:41 K 线闭合。因此文档不再承诺 09:40:10 已生成最终数量买单。

### S8. 收盘成交、持仓与现金对账

**Qlib 对应层**：Position 状态持久化、Record/审计；外部券商 reconciliation。

| 类别 | 内容 |
|---|---|
| 输入信息 | 全部 sell/buy fills；券商收盘持仓和现金；S5/S7 目标；09:41 参考价 |
| 已有来源 | 人工券商导出；单日生产目录 |
| 数据组装 | 逐笔核对方向、数量、均价、费用；重建实际 Position；计算买入相对 `price_941` 滑点、上午卖出滑点、未成交原因；与券商现金核对；生成次日权威持仓快照 |
| 输出信息 | `fills.csv`、`positions_after_close.csv`、`cash_after_close.json`、`daily_reconciliation.json` |

对账不通过时，次日 S0 必须失败，直至人工修复并留下审计说明。

### S9. 盘后 Qlib 数据更新、Label 完整性和滚动训练

**Qlib 对应层**：Provider → DataHandlerLP → DatasetH → task_train → Recorder。

| 类别 | 内容 |
|---|---|
| 输入信息 | T 日收盘日频数据；T 日完整 240 根分钟数据；历史 instruments；滚动 task template；Qlib 日历 |
| 已有来源 | 日/分钟 Qlib 数据源；`materialize_minute.py`；`scripts/retrain.py`；Champion handler/label 配置 |
| 数据组装 | 更新 provider；物化 T 日 18 因子、`price_941/change_941`；确认 T-1 label 已完整而 T label 尚不可用；按交易日精确生成 train90/valid20/embargo1/test≤20；DatasetH 使用 DK_L 训练、DK_I 验证；分别 task_train HFLGB 和 XGB |
| 输出信息 | 两个新的 Qlib Recorder；各自 `params.pkl/task/dataset`；HFLGB recorder 的 `ensemble_gate.pkl`；外部 `candidate_model_manifest.json` |

截至规则：

```text
盘后日期 = T
最新完整 label 样本 = T-1
候选最早可用日期 = T+1
```

模型物理产物仍是 Qlib `.pkl`：

```text
minute_enhanced_rolling/<hflgb_recorder>/artifacts/params.pkl
minute_enhanced_rolling_xgb/<xgb_recorder>/artifacts/params.pkl
minute_enhanced_rolling/<hflgb_recorder>/artifacts/ensemble_gate.pkl
```

`candidate_model_manifest.json` 只是治理索引：

```json
{
  "status": "CANDIDATE",
  "trained_asof": "2026-07-21",
  "usable_from": "2026-07-22",
  "latest_complete_label_date": "2026-07-20",
  "hflgb_experiment": "minute_enhanced_rolling",
  "hflgb_recorder_id": "...",
  "hflgb_artifact": "params.pkl",
  "xgb_experiment": "minute_enhanced_rolling_xgb",
  "xgb_recorder_id": "...",
  "xgb_artifact": "params.pkl",
  "gate_artifact": "ensemble_gate.pkl",
  "feature_schema_sha256": "...",
  "data_snapshot_sha256": "..."
}
```

当前 `scripts/retrain.py` 手工生成精确交易日窗口并调用 `task_train + OnlineToolR`，这符合 Qlib 组件化用法，但并未实际调用 `RollingGen`。文档和代码不得再声称已由 RollingGen 生成任务，除非后续确实改造。

当前 `step=20` 表示每日检查、每个模型冻结最多 20 个交易日，不是每日训练一个新模型。若未来改成真正日更 `step=1`，必须重新完成独立前向验证。

### S10. 候选验证、人工批准和 T+1 发布

**Qlib 对应层**：Recorder metrics、OnlineToolR；外部审批治理。

| 类别 | 内容 |
|---|---|
| 输入信息 | Candidate 两个 Recorder；`ensemble_gate.pkl`；active Champion；同窗 validation 数据；模拟最新日特征 |
| 已有来源 | `model_ensemble.py`；Recorder `task/params.pkl`；历史 Champion `pred.pkl/label.pkl` |
| 数据组装 | 校验 recorder provenance、train/valid/test 和 embargo；预测非恒定、NaN/coverage；计算 IC、RankIC、TD0 策略收益、成本、换手、最大回撤；与 active 同窗对比；执行最新日结构模拟和历史零漂移测试；人工审核 |
| 输出信息 | `validation_report.json`、审批记录；批准后生成下一交易日 `active_model_manifest.json`，并更新 OnlineToolR tag |

推荐发布顺序：

```text
训练完成（Recorder FINISHED）
→ 写完 gate 和 candidate manifest
→ 验证报告 PASS/REVIEW
→ 人工 APPROVED
→ 原子写 T+1 active manifest
→ reset_online_tag
→ T+1 盘前再次加载与 checksum 检查
```

当前实现是在 gate 写完后直接 `reset_online_tag`，缺少 Candidate→人工批准步骤。实施本方案时必须把“训练完成”和“成为 online/active”拆开。

## 6. 方案 B 与 Champion 回测口径

用户确认采用：**上午先卖旧仓，再买新仓**。它解决人工账户资金时序，但改变了 Champion 的卖出价格。

Champion 训练/回测持有期：

```text
T日 09:41 买入 → T+1 日收盘卖出
```

方案 B 实际持有期：

```text
T日 09:41 买入 → T+1 日上午卖出
```

所以第一阶段应同时保存：

- `champion_reference_return`：09:41→次日 close，用于判断模型信号是否仍有效。
- `realized_scheme_b_return`：真实买入 fill→次日上午卖出 fill，用于评估实际策略。
- `execution_gap`：二者差异及成本。

投入真实资金前必须用 Qlib Exchange 增加方案 B 的方向性 sell price 或等价的上午卖出字段，保持以下条件不变做独立回测：

- 相同 18 因子、Label 对照和模型预测。
- 相同 T 日时变股池。
- 相同 Topk10/n_drop8/hold_thresh1。
- 相同涨跌停、成本和交易单位。
- 唯一变量为卖出时间/价格。

未通过前，方案 B 输出应标记 `execution_policy=SCHEME_B_UNVALIDATED`，不得把历史 Champion 的 +收益指标直接映射到实盘。

## 7. 单日目录及产物依赖

```text
data/production_signals/YYYY-MM-DD/
├── run_manifest.json
├── preflight.json
├── active_model_manifest.json
├── universe_raw.csv
├── universe_eligible.csv
├── universe_quality.json
├── positions_before.csv
├── cash_before.json
├── factor_bars.parquet
├── bar_collection_status.csv
├── bar_quality.json
├── features.parquet
├── feature_schema.json
├── factor_quality.csv
├── scores.csv
├── rebalance_decision.json
├── sell_orders.csv
├── sell_fills.csv
├── positions_after_sell.csv
├── cash_after_sell.json
├── execution_bar_0941.parquet
├── execution_quality.json
├── reserve_candidates.csv
├── buy_orders.csv
├── buy_fills.csv
├── fills.csv
├── positions_after_close.csv
├── cash_after_close.json
├── daily_reconciliation.json
├── candidate_model_manifest.json
└── validation_report.json
```

依赖规则：下游文件必须记录上游文件 SHA256。例如 `buy_orders.csv` 的元数据必须引用 `scores.csv`、`sell_fills.csv`、`execution_bar_0941.parquet` 和 active manifest 的 hash，防止人工修改后静默重算。

## 8. 当前实现状态

| 项目 | 当前已有 | 目标修订 |
|---|---|---|
| 冻结 Champion 评分 | 已实现 | `scripts/intraday_production.py score` |
| 十根分钟采集与门禁 | 已实现 | 09:31–09:40 逐根幂等落盘，缺口 fail closed |
| 09:41 辅助行情 | 已实现 | 与十根因子 K 线分离 |
| 因子组装与推理 | 已实现 | 18 维特征，历史公共样本分数零漂移 |
| TopkDropout 调仓 | 已实现 | Top10/n_drop=8，输出 sell/keep/buy |
| 成交回填与对账 | 已实现 | sell fills→现金→buy quantity→收盘对账 |
| 滚动候选训练 | 已实现 | Candidate Recorder→验证材料→人工批准 |
| Candidate manifest | 已实现 | JSON 仅引用 Recorder 中的 `.pkl` |
| 方案 B 历史回放 | 已实现 | 62 个交易日逐日对账通过 |

## 9. 失败门禁

任一条件成立时状态设为 `HOLD_ONLY`，禁止生成新增买单：

- T 日原始快照不是 100 只、日期错误、代码重复或被旧快照替代。
- 因子窗口不是严格 09:31–09:40 十根，或有效特征股票少于 80。
- 09:41 辅助行情缺失，导致价格、涨跌停或买入数量不可确定。
- feature schema/order/hash 与模型不一致。
- active Recorder、`params.pkl` checksum 或 Qlib 版本不一致。
- 盘前持仓、可卖数量、现金与券商不一致。
- 同日已存在内容不同的正式订单文件。
- 卖出回报未导入，却尝试按预计资金生成买单。
- 昨日对账失败或存在未处理订单。

失败时禁止自动退化为旧股池、研究模型、未批准 rolling 模型，或用收盘后数据补造“实时信号”。

## 10. 实施后的人工操作流程

以下是目标 CLI 契约，当前仓库尚未全部实现：

```bash
# 08:50：锁定日期、模型、版本和上日状态
python scripts/intraday_production.py preflight --date YYYY-MM-DD

# 09:00：获取并冻结 T 日原始/eligible 成分
python scripts/intraday_production.py universe --date YYYY-MM-DD

# 09:29：常驻，每分钟分别采集 09:31…09:40
python scripts/intraday_production.py collect-factor-bars --date YYYY-MM-DD

# 09:40 后：组装特征、评分、生成卖单
python scripts/intraday_production.py score-and-plan-sells --date YYYY-MM-DD

# 人工卖出并填写 sell_fills.csv 后
python scripts/intraday_production.py import-sell-fills \
  --date YYYY-MM-DD --file data/production_signals/YYYY-MM-DD/sell_fills.csv

# 09:41 闭合后：采集辅助行情并生成最终买单
python scripts/intraday_production.py build-buy-orders --date YYYY-MM-DD

# 收盘后导入成交/持仓并对账
python scripts/intraday_production.py reconcile --date YYYY-MM-DD

# 可选：只在需要新冻结段时训练 Candidate
python scripts/intraday_production.py retrain --asof YYYY-MM-DD
python scripts/intraday_production.py validate-model --asof YYYY-MM-DD

# 人工 review validation_report 后执行
python scripts/intraday_production.py promote-model --asof YYYY-MM-DD
```

## 11. 验收测试

### 11.1 数据一致性

- 历史任意日期，逐分钟落盘重放得到的十根 K 线与 1min bin 完全一致。
- 实时纯函数特征与 `materialize_minute.py` 的 18 个 day.bin：`max_abs_diff < 1e-6`。
- 明确验证 09:41 不进入 18 维 feature matrix。
- universe_raw 恰好 100 行，Qlib instruments 查询 T 日返回同一原始 code 集。

### 11.2 Qlib 模型一致性

- 同一历史日期，内存特征与 `DatasetH.prepare(..., DK_I)` 一致。
- `predict_feature_matrix(params.pkl, X)` 与标准历史 `model.predict(dataset)`：score `max_abs_diff < 1e-6`。
- 排序和 Top10 完全相同。

### 11.3 Qlib 策略一致性

- 给定同一 score、Position、Exchange，CSV 选择的 sell/keep/buy code 集与 `TopkDropoutStrategyTD0.generate_trade_decision()` 完全一致。
- 无法卖出旧仓继续占槽位；部分成交后不会多买。
- `n_drop≤8`、最终持仓数≤10、hold_thresh=1。

### 11.4 模型治理一致性

- Candidate 训练完成不会自动改变当日 active model。
- manifest 中 recorder ID、artifact SHA256、task segments 和 gate 一致。
- 未批准 Candidate 无法被生产加载。
- active 加载失败只能回退到上一版已批准 manifest，并显式记录；不能自行选最新 Recorder。

### 11.5 收益口径一致性

- 保留原 Champion Exchange 回测作为基准。
- 方案 B 使用上午 sell price 单独回测，输出收益、回撤、换手、滑点敏感性和相对原 Champion 的 execution gap。
- 只有方案 B 前向和回测门禁通过，才删除 `SCHEME_B_UNVALIDATED` 标记。

## 12. 实施顺序

1. 先提取实时/物化共享的 18 因子纯函数，并建立零漂移测试。
2. 实现单日目录、run/active manifest、SHA256 和运行锁。
3. 实现严格逐分钟的十根因子 K 线采集及 09:40 缺口补采。
4. 将 09:41 辅助行情从因子采集拆出。
5. 将现有 frozen Champion 评分接入新的数据契约。
6. 用 Qlib-compatible Position/TopkDropout 决策替换无状态 BUY Top10 CSV。
7. 实现 sell fills→现金→buy orders 和收盘对账。
8. 将滚动训练改为 Candidate→验证→人工批准→OnlineToolR。
9. 补做方案 B 等价回测和至少 5–10 个交易日影子运行。
10. 全部验收通过后，再决定是否用于真实资金。

## 13. 2026-07-20 实施状态

| 能力 | 状态 | 实现位置/说明 |
|---|---|---|
| 十根因子 K 线与 09:41 辅助行情分离 | 已实现 | `minute_factors.compute_champion_factors`、`live/intraday.py` |
| 18 因子共享实时计算 | 已实现 | 实时路径已改用共享纯函数；历史向量物化保留独立高性能实现并由实数交叉测试约束 |
| 62 日历史影子回放 | 已实现 | 2026-04-01 至 2026-07-02；公共样本 score 零漂移、每日现金持仓对账通过 |
| 单日目录、原子写、SHA256、schema hash | 已实现 | `qlib_ifind_beta/live/intraday.py` |
| T 日快照和原始/eligible 双文件 | 已实现 | `scripts/intraday_production.py universe` |
| 按指定分钟幂等采集 | 已实现 | `collect-factor-bars --minute HH:MM`；需在真实交易日验证行情时间戳 |
| 十根完整性与候选覆盖门禁 | 已实现 | `validate_factor_bars`、`assemble_features` |
| Frozen Champion 内存评分 | 已实现 | 复用现有 `_predict_in_memory(..., use_online=False)` |
| 有状态 Topk10/n_drop8 规划 | 已实现 | `plan_topk_dropout`；保留 Qlib 语义回归门禁 |
| 卖单、部分成交、真实现金驱动买单 | 已实现 | `apply_sell_fills`、`build_buy_orders` |
| 09:41 涨停过滤与顺延 | 已实现 | `build_execution_snapshot`、`build_buy_orders` |
| 收盘持仓/现金对账 | 已实现 | `reconcile_positions` |
| Candidate/validation JSON | 已实现 | `scripts/retrain.py`；模型本体仍为 Recorder `params.pkl` |
| 人工批准后 OnlineToolR 发布 | 已实现 | `promote_candidate`；`run()` 默认不再自动 online |
| 公司行动/复权因子自动源 | 部分实现 | 当前 fail-closed；仅在人工确认无 factor 变化时使用 `--factor-confirmed` |
| 自动常驻 09:31–09:40 调度与缺口重试 | 部分实现 | 已有单分钟幂等命令，尚未包装系统调度器 |
| 券商 API | 不实施 | 第一阶段按决策仅生成 CSV、人工操作 |
| 5–10 日影子运行 | 待交易日 | 需要真实盘中数据和人工成交回报 |

方案 B 已完成同一 Champion W1 窗口的 Qlib Exchange 对照，输出：

```text
data/scheme_b_execution_validation.json
```

结果（2026-04-01～2026-07-02）：

| 指标 | Champion：09:41买/收盘卖 | 方案B：09:41买/次日09:41卖 | 差值 |
|---|---:|---:|---:|
| 绝对年化收益 | 192.32% | 140.49% | -51.82pct |
| 绝对最大回撤 | -15.30% | -11.35% | 改善3.95pct |
| 超额年化收益 | 148.21% | 96.38% | -51.82pct |
| 超额信息比率 | 3.98 | 3.28 | -0.70 |
| 超额最大回撤 | -12.93% | -10.16% | 改善2.77pct |

因此方案 B 当前保持 `SCHEME_B_UNVALIDATED/REVIEW`：回撤有所下降，但收益和信息比率明显弱于 Champion，不能直接继承原生产收益结论。是否接受该收益—回撤交换，需要结合 5–10 个交易日真实成交影子结果再决定。
