---
layout: default
title: 核心库精读
parent: "代码导读"
nav_order: 1
---

# 核心库精读（qlib_ifind_beta/）

按依赖自底向上讲解。每节给出：职责、关键符号、需要记住的不变量。

## config.py —— 一切的起点（159 行）

纯常量模块，零依赖。三组内容：

- **路径**：`PROJECT_ROOT / OVERLAY_ROOT / QLIB_DATA / CN_DATA_1MIN / DAY_CAL / MIN_CAL`；
- **冻结合同**：`CHAMPION_RECORDER_ID`（93d435e0…）、`CHAMPION_EXPERIMENT`、
  `CHAMPION_LABEL_EXPR`（**label v2**：`Ref($close1500, -1) / $close0941 - 1`，
  2026-09-12 起）、`CHAMPION_TOPK=10`；
- **字段清单**：`MINUTE_FACTOR_FIELDS`（14）+ `MINUTE_FACTOR_EXTRA_FIELDS`（4）
  → Handler 的 18 特征；`MINUTE_DEAL_PRICE_FIELD="price_941"`、
  `MINUTE_CLOSE_0941_FIELD/1500`（label v2 两腿）。

改任何常量前先读 [配置说明](../configs.md) 的修改守则——多数是 FROZEN。

## binio.py —— qlib bin 物理层（53 行）

`read_bin(path) -> (start_index, values)` / `write_bin`。格式：little-endian
float32；首 4 字节 = 该股数据在日历中的起始行号；之后每值对齐一个日历日，
缺失 = NaN（占位而非省略）。所有绕过 qlib 表达式引擎的快路径
（物化、回放）都经它。细节见[数据源与读取 §2](../data-access.md)。

## overlay.py —— 叠加层（103 行）

`link_calendars / link_instruments / link_stock / link_benchmark / write_market_file`。
把只读源 symlink 进 `data/qlib_root/`，自有 bin 写真实文件。幂等。
`write_market_file` 是 instruments 股池文件的唯一写入者。

## universe.py —— 时变股池（226 行）

- `fetch_constituents(iv_date)`：单日 p03473 快照；
- `fetch_history_snapshots(start,end)`：增量断点续拉 → `universe_snapshots.csv`
  （每 50 天 flush；08:30 cron 的 `update_universe.py` 调它）；
- `snapshots_to_segments`：long 快照 → `{code: [(d_in,d_out),…]}`
  在册区间（T 日盘前更新口径，无 shift）；
- `dump_universe`：end-to-end（拉取 + 写 instruments）。

不变量：**任意交易日 T 的在册集 = 当日真实 100 只**（无前视第一道闸）。

## materialize.py / materialize_minute.py —— 物化层（89 + 399 行）

`materialize.py`：`board_limit(code)`（按板块返回 ±0.095/0.195/0.295 涨跌停线，
涨停拦截的表达式用）+ `compute_change`（不复权涨跌幅）→ `change/limit_up/limit_down`
三个日频衍生 bin。

`materialize_minute.py`：最厚的物化模块，`materialize_minute_instrument(code)` 一股
写出 **22 个 day.bin**：14 基础因子 + 4 extra + `price_941`（V4 校正后）+
`change_941` + label v2 两腿（`close0941` 原值 / `close1500`）。向量实现（性能），
以 `minute_factors.compute_day_factors` 为 oracle（`tests/test_materialize_minute.py`
逐位对齐断言）。V4 校正（~3% 票 1min bin 误存 raw 价 → ×factor 还原）只作用于
`price_941`，**不作用于 close0941**（label v2 有意保留原值口径）。

## minute_factors.py —— 18 因子纯函数（131 行）★

`compute_day_factors(c,o,h,l,vol, prev_day_minute_vol)`：11 根 bar 数组进、
14 因子 + price_941 出；`compute_champion_factors(...)` 再加 4 extra
（vol_vs_yest_t2/t3/t5 + overnight_gap 名义口径）。**实时与物化共用这一个实现**
（零漂移的根基）；因子语义表见[因子说明](../factors.md)。

## handler 家族 —— qlib 数据集接入（73 + 23 行）

`HighBetaAlpha158(Alpha158)`：挂 shared `DropnaProcessor(feature)` 前视护栏
（缺任一因子整行剔除，防 9:41 NaN 回退路径），learn 处理器为空。
`MinuteEnhancedHandler(HighBetaAlpha158)`：**只覆盖 `get_feature_config()`**
返回固定顺序的 18 特征——Champion 合同的代码化。DK_R/DK_I/DK_L 三视图解释见
[回测与脚本解析](../backtest-scripts.md)。

## model_ensemble.py —— 门控与混合（237 行）

`correlation_metrics`（逐日 IC/RankIC）、`gate_passes`（三门槛全过才 25% 权重）、
`make_gate_metadata / validate_gate_metadata`（防篡改：gate_passed↔weight 一致、
embargo 证明）、`blend_scores`（截面 z 后加权）、`predict_feature_matrix`
（内存矩阵推理，HFLGB/XGB 通吃）、`validate_recorder_provenance`。
设计哲学：**任何异常权重归 0 回退 HFLGB，绝不猜**。详见[模型说明](../models.md)。

## evaluation.py —— TD0 手动回测（125 行）

`backtest_td0(pred, ...)`：不走 SimulatorExecutor 的轻量回测（top10 等权、
09:41 买/T+1 收盘卖、含成本），供 retrain 门控第三门槛
（年化超额）快速计算。注意与 PortAna 口径的差异（无涨跌停拦截/无 n_drop）。

## td0_strategy.py —— 交易语义（181 行）

`TopkDropoutStrategyTD0(TopkDropoutStrategy)`：把 qlib 原生的 shift=1 改为 0，
实现"T 日 09:41 信号当日成交"，与 label horizon 对齐。topk=10/n_drop=8/
hold_thresh=1 的语义不变量由 `tests/test_td0_strategy.py` 守护。

## ifind.py —— 外部客户端（225 行）

token 文件缓存 + refresh 自动续期（`QLIB_IFIND_TOKEN_FILE`）、`_post` 重试、
`fetch_data_pool`/`fetch_history_data` 两个 API 封装。**唯一允许联网的模块**，
测试全部 mock（红线：token 不入库、iFinD 只取成分名单）。
