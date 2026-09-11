---
layout: default
title: 数据源与读取
nav_order: 3.5
---

# 数据源与读取

本项目消费的全部数据：**哪些已经躺在本地、哪些需要外部获取、代码用哪条路径读**。
数据的使用方式（因子/训练/生产）见后续各页；本文只讲数据层。

## 1. 数据资产总览（已下载 / 本地）

| 数据 | 路径 | 覆盖（2026-09-11 实测） | 权限 |
|---|---|---|---|
| 日频行情 7 字段 | `~/.qlib/qlib_data/cn_data` | 2000-01-04 → **2026-09-11**（6470 天），全 A + 指数（sh000300 基准） | 只读 |
| 1 分钟 K 线 | `~/.qlib/qlib_data/cn_data_1min` | 同区间，**每天恰好 240 根**（09:31–15:00），含 factor 字段 | 只读 |
| 883926 成分快照缓存 | `data/universe_snapshots.csv` | 2024-01-02 → **2026-07-20**（615 天 × 100 只/天） | 项目数据 |
| overlay 叠加层 | `data/qlib_root` | symlink 只读源 + 自有 bin（股池/涨跌停线/18 因子/price_941） | 可写（构建产物） |
| 模型与预测 | `mlruns/` | Champion/滚动/研究 recorder | 运行资产 |
| 883926 指数本体 | overlay `features/sh883926/` | 由 `dump_index.py` 从 iFinD dump | ⚠️ close 字段有源级漂移，消费前读该模块注释 |

三层数据的**截止日期天然不同步**（外部行情管道几乎每日更新，其余靠手动）：

```text
cn_data / cn_data_1min   ────────►  2026-09-11   （外部管道自动更新）
物化因子（price_941 等）  ─────►  2026-09-04   （上次 build_overlay/materialize 时点）
universe 成分快照        ────►  2026-07-20   （上次 iFinD 拉取）← 实际瓶颈
```

因此：**07-21 之后即使行情已就位，也无法回放/生产**（缺当日成分名单）；
09-05 之后的因子 bin 也需要重物化才会存在。

## 2. bin 文件解剖（一切读取的物理层）

qlib 的 `.day.bin` / `.1min.bin` 格式（`qlib_ifind_beta/binio.py` 按官方
`FileFeatureStorage` 源码确认）：

```text
little-endian float32 数组：
  [0]      = start_index：该股票数据在日历中的起始行号
  [1:]     = 每个日历日（或每根分钟 bar）一个值，从 start_index 起对齐
             缺失日 = NaN（不是省略！）
```

实测解剖（SZ300164，`read_bin` 直读）：

| 文件 | start_index | 长度 | 说明 |
|---|---|---|---|
| `cn_data/.../close.day.bin` | 4846 | 1624 | 4846+1624=6470 = 日历总行数 ✓（原生数据随外部管道更新到当日） |
| overlay `price_941.day.bin` | 4846 | 1619 | 4846+1619=6465 → **物化截止 09-04**；前 3 个值是 NaN（该窗口无分钟数据，NaN 占位保持日历对齐） |
| `cn_data_1min/.../close.1min.bin` | 1,395,840 | 156,720 | = 653 天 × 240；新股/覆盖期外的日在 1min 日历上同样 NaN 对齐 |

注意目录名是**小写**股票代码（`features/sz300164/`）——qlib 存储层把 instrument
统一小写，与代码中的 `SZ300164` 大小写不敏感等价。

## 3. 四条读取路径

### 路径 A：qlib 表达式引擎（训练/回测/日常取数）

```python
qlib.init(provider_uri="data/qlib_root", region="cn")
D.features(instruments, ["$close", "Ref($close,-1)/$price_941-1"],
           start_time=..., end_time=...)
```

内部链路：`表达式解析（$ 字段/Ref 等算子）→ instruments 过滤 → 日历切片 →
FileFeatureStorage 按 start_index 对齐读 bin → DataFrame`。
**时变股池过滤**是关键一步：对 T 日查询，只有 instruments 文件中
`start ≤ T ≤ end` 的票会被返回（实测任意 T 日恰好 ~100 只），未来入池的票自动
不可见——这是无前视的第一道闸。

两个实测过的坑：
- **1min 查询的 `end_time` 按 00:00 截断**：`end_time="2026-07-01"` 会丢掉 7-01
  全天分钟线，须写次日 `2026-07-02`；
- **最后一天的 label 永远是 NaN**（`Ref($close,-1)` 需要 T+1 收盘），所以回测
  `end_time=07-01` 而数据 `end_time=07-02`。

1 分钟 slot 映射（每天 240 根，索引 0–239）：

```text
index 0..9   = 09:31..09:40  ← 18 个因子的唯一窗口
index 10     = 09:41         ← price_941/change_941 成交辅助
index 11..239 = 09:42..15:00 ← 本策略不使用
```

### 路径 B：binio 直读（影子回放的高性能路径）

`qlib_ifind_beta/live/historical_replay.py` 的 `HistoricalReplaySource` 绕过
表达式引擎，按日历行号直接切 bin：`bars(date, codes)` 取 11 根分钟 bar、
`previous_volumes` 取 T-k 全天分钟量、`daily_info` 取 T-1 close/factor。
已知的坑：**部分历史日频 bin 有日历空洞**（某天整行缺失），`daily_info` 对这些
票自动降级用 1 分钟 close×factor 补（`historical_source=minute_fallback`，
回放日志会计数）。

### 路径 C：实时采集（真实交易日盘中，唯一在线路径）

`qlib_ifind_beta/realtime/data_fetch.py` 调用外部 kline-fetcher 服务
（`KLINE_API_BASE_URL`，默认 `http://183.242.5.14:7778`）：
`fetch_realtime_bars(code, count=11)` 取最近 11 根，生产链路逐分钟校验
`bar_time` 后幂等 upsert 到 `factor_bars.parquet`——**严禁**用"最近十根"直接
覆盖目标窗口（09:41 后最近十根会变成 09:32–09:41）。非交易日无此数据，
测试用历史 1min dry-run 代替。

### 路径 D：人工导入（第一阶段合同）

券商持仓/成交回报由人工导出 CSV 后进入 `positions_before.csv` /
`sell_fills.csv` 等文件；对账环节逐笔核验。项目**不接券商 API**。

## 4. 未下载 / 需外部获取的数据

| 数据 | 获取方式 | 状态 |
|---|---|---|
| universe 新日期快照（>07-20） | `universe.fetch_history_snapshots(start,end)` 调 iFinD `p03473`，断点续拉入 `universe_snapshots.csv`；token 走 `QLIB_IFIND_TOKEN_FILE`（默认 `~/qlib_data/.ifind_token`，refresh token 自动续期） | **当前瓶颈**，需人工触发 |
| 日频/1min 新日期行情 | **仓库内无更新脚本**——只读源由仓库外的数据管道维护（实测它几乎每日在更新），项目红线"不生产行情数据" | 外部自动 |
| 盘中实时分钟 bar | kline-fetcher 服务（路径 C） | 仅真实交易日 09:31–09:41 |
| T 日 label | 不是下载问题而是时间边界：**T+1 收盘后才存在** | 永远晚一天 |
| 已有但不用 | 当日 09:42 后的 229 根分钟线；883926 指数本体 bin（close 字段异常） | — |

## 5. 想把流程跑到新日期（操作顺序）

```bash
# ① 确认外部管道已更新行情（日历最后一行 ≥ 目标日）
tail -1 ~/.qlib/qlib_data/cn_data/calendars/day.txt
# ② 增量拉 universe 快照（需 iFinD token；build_overlay 内置这步）
conda run -n qlib_ifind_beta python -m scripts.build_overlay
#    ——幂等：symlink 重建 + 股池刷新 + 涨跌停/18 因子/price_941 重物化到最新
# ③ 盘后滚动候选训练（可选；默认 CANDIDATE 不发布）
conda run -n qlib_ifind_beta python scripts/retrain.py --test-start <日期>
# ④ 前向影子回放（超出 Champion pred 覆盖的日期需显式解锁）
conda run -n qlib_ifind_beta python scripts/replay_intraday_shadow.py \
    --start <日期> --end <日期> --output data/<新目录> --allow-unreferenced-scores
```

## 6. 红线（数据层）

1. 只读源绝不写入；一切自有数据进 `data/qlib_root` 叠加层；
2. 不生产行情数据；iFinD 只用来取成分名单（及 883926 指数 dump）；
3. token 只经 `QLIB_IFIND_TOKEN_FILE` 环境变量/文件读取，不入库不入 CSV；
4. 运行资产（data/、mlruns/、reports/、logs/）不进 Git。

## 延伸阅读

[项目总览](project-overview.md) · [因子说明](factors.md)（18 因子怎么用这些数据） ·
[配置说明](configs.md)（provider_uri 与窗口） · [回测与脚本解析](backtest-scripts.md)
（读取之后的完整执行链） · [产物地图](artifacts.md)
