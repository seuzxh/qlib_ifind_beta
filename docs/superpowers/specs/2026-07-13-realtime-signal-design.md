# 实时模拟盘信号生成系统设计（§56）

> 2026-07-13，`feat/realtime-signal` 分支

## 目标

每个交易日 9:41，用实时 9:31-9:40 分钟K线数据生成 top10 标的池信号。

## 执行命令

```bash
cd /home/zxh/projects/3.qlib_ifind_beta
conda run -n qlib_ifind_beta --no-capture-output python -W ignore \
    scripts/realtime_signal.py --date 2026-07-15
```

## 全流程

```
scripts/realtime_signal.py --date T
  │
  ▼
generate_realtime_signal(T)                               [signal.py]
  │
  ├─① load_universe(T, filter_st=True)                   [data_fetch.py]
  │     读 data/qlib_root/instruments/highbeta883926.txt
  │     过滤 start≤T≤end 的成分股
  │     kline-fetcher 查名称排除 ST/*ST
  │
  ├─② fetch_bars_parallel(codes, count=11)               [data_fetch.py]
  │     kline-fetcher 并行拉每只票最新 11 根分钟K（09:31-09:41）
  │     KLINE_API_BASE_URL=http://183.242.5.14:7778
  │
  ├─③ get_prev_day_volumes_multi(codes, T)               [data_fetch.py]
  │     从 cn_data_1min 读 T-1/T-2/T-3/T-5 全天分钟成交量
  │     用于 vol_vs_yest 族因子（分母）
  │
  ├─④ get_daily_close_factor(codes, T)                   [data_fetch.py]
  │     从 qlib_data 日频 bin 读 T-1 close/factor + T open/factor
  │     用于 overnight_gap + change_941
  │
  ├─⑤ _compute_all_factors()                             [signal.py]
  │     14 baseline 因子 (startup_mom/accel/close_pos/vol_ratio/vol_vs_yest)
  │     3 extra (vol_vs_yest_t2/t3/t5)
  │     overnight_gap / change_941 / amt_ratio_5m / amt_vs_yest
  │     共 18 因子（champion 消费的字段集）
  │
  ├─⑥ _predict_in_memory(T, factor_rows)                 [signal.py]
  │     构建 handler 读历史 bin (2024-2025 train 段) fit ZScoreNorm
  │     手动 ProcessInf → ZScoreNorm → Fillna 处理 T 日因子
  │     加载 FROZEN champion 模型 (93d435e0, HFLGBModel)
  │     model.model.predict() → scores → 排序
  │
  ├─⑦ 封涨停拦截
  │     change_941 >= limit_up → 剔出 topk
  │
  └─⑧ 输出
        控制台 top10 表格 + data/realtime_signals/T.csv
```

## 架构设计

### 纯内存路径（不修改 qlib 日历/bin）

实时行情从 kline-fetcher 来，自带时间戳。因子在内存中计算后直接构建
DataFrame，用 handler 历史 fit 的 ZScoreNorm 参数做标准化，然后
model.predict()。不扩展 day.txt、不写 daily bins、不写 factor bins。

### 与 P1（盘后）的区别

| | P1 盘后 (live_forward.py) | 实时盘前 (realtime_signal.py) |
|---|---|---|
| 数据源 | cn_data_1min 已同步的 bin | kline-fetcher 实时分钟K |
| 因子计算 | materialize_minute → bin | 内存 _compute_all_factors |
| 预测 | predict_day (读 bin → DatasetH) | _predict_in_memory (内存 DataFrame) |
| label | Ref($close,-1) 需要 T+1 | 用 $close dummy 绕过 DropnaLabel |

## 验证结果

- **因子精度**：realtime vs materialized 18 因子全匹配（max diff 2.81e-06）
- **ST 过滤**：SZ000838(*ST发展) 成功过滤
- **7/13 实测**：候选 99 / 入选 10，信号生成成功
- **测试**：11/11 新测试通过，73/73 全量通过

## 已知问题（5 个，待修复）

### 问题 1：Universe 缺 T 日成分股（致命）

**现象**：`data/qlib_root/instruments/highbeta883926.txt` 只缓存到
qlib_data 同步的最后日期（当前 7/10）。T 日（7/13）的 883926 成分股不存在。

**影响**：`load_universe(T)` 返回 0 只 → 无法选股。

**当前 workaround**：手动用 iFinD p03473 拉当天成分股注入 instruments 文件。

**修复方向**：在 `generate_realtime_signal` 开头自动拉 iFinD p03473 注入。

### 问题 2：daily close/factor 缺 T-1 数据

**现象**：`get_daily_close_factor` 从 qlib_data 日频 bin 读取，bin 只到 7/10。
T-1（=7/10）的 close/factor 存在，但 T 日 open/factor 不存在。

**影响**：
- `prev_close`/`prev_factor` 可读到（T-1 = 7/10 在 bin 内）
- `today_open`/`today_factor` 读不到 → `overnight_gap = NaN`
- `change_941 = NaN`（依赖 today_factor）

**实际影响范围**：仅影响 overnight_gap 和 change_941 两个因子。
14 个分钟因子 + 3 个 vol_vs_yest_t{k} 不受影响。

### 问题 3：overnight_gap=NaN → Fillna 填 0

**现象**：ZScoreNorm 后的 Fillna 把 overnight_gap 的 NaN 填为 0。

**影响**：18 因子中 1 个被错误填充，可能导致预测分布偏移。
修复问题 2 后自动解决（overnight_gap 有值后不会被 Fillna 覆盖）。

**临时缓解**：从 kline bars 的 T 日 open vs T-1 close 手算 overnight_gap，
不依赖 qlib_data bin。

### 问题 4：涨停拦截失效

**现象**：`limit_up`/`limit_down` 从 overlay bin 读取，T 日不存在。

**影响**：`change_941 >= limit_up` 判断中 limit_up=NaN → 不拦截 → 可能选入
已封涨停无法买入的票。

**修复方向**：用 `board_limit(code)` 函数（materialize.py）直接计算涨跌停比例，
不依赖 bin。或者用 T-1 close × 板块限制（主板 10%/创科 20%/BJ 30%）算阈值。

### 问题 5：ZScoreNorm 每次重新 fit（性能）

**现象**：`_predict_in_memory` 每次构建 handler + 读历史 bin + fit ZScoreNorm，
耗时约 8 秒。

**影响**：每日运行多花 8 秒（非致命）。

**修复方向**：缓存 fit 参数（mean_train / std_train / cols）到 pickle，
首次 fit 后后续直接加载。

## 文件清单

| 文件 | 职责 |
|---|---|
| `qlib_ifind_beta/realtime/__init__.py` | 包初始化 |
| `qlib_ifind_beta/realtime/data_fetch.py` | kline-fetcher 封装 + universe + prev-day 数据 |
| `qlib_ifind_beta/realtime/signal.py` | 核心信号生成（fetch → factors → predict） |
| `scripts/realtime_signal.py` | CLI 入口（--date / --dry-run / --topk） |
| `tests/test_realtime_signal.py` | 11 测试（因子计算 / 边界 / 数据读取） |

## 依赖

| 依赖 | 状态 | 备注 |
|---|---|---|
| kline-fetcher | ✅ | `KLINE_API_BASE_URL=http://183.242.5.14:7778` |
| FROZEN 模型 | ✅ | `93d435e0` HFLGBModel champion |
| cn_data_1min | ✅ | T-1 分钟量（需 cron 同步） |
| qlib_data | ✅ | 日频 close/factor（需 cron 同步） |
| iFinD token | ✅ 至 7/16 | universe 刷新需要 |
