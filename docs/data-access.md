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
| 883926 成分快照缓存 | `data/universe_snapshots.csv` | 2024-01-02 → **2026-09-11**（654 天 × 100 只/天，08:30 cron 自动追加） | 项目数据 |
| overlay 叠加层 | `data/qlib_root` | symlink 只读源 + 自有 bin（股池/涨跌停线/18 因子/price_941） | 可写（构建产物） |
| 模型与预测 | `mlruns/` | Champion/滚动/研究 recorder | 运行资产 |
| 883926 指数本体 | overlay `features/sh883926/` | 由 `dump_index.py` 从 iFinD dump | ⚠️ close 字段有源级漂移，消费前读该模块注释 |

### 三类关键数据的获取路径与更新时机

| | 行情源 cn_data / cn_data_1min | universe 成分快照 | 物化因子（18 因子/price_941/涨跌停线） |
|---|---|---|---|
| **生产者** | 仓库外 cron 管道（kline-fetcher 服务 → qlib bin），**项目只读** | 本项目（iFinD 客户端） | 本项目（本地计算） |
| **获取路径** | 文件系统直读；数据经 `183.242.5.14:7778` kline-fetcher 落盘 | `universe.fetch_history_snapshots` 调 iFinD `p03473`（增量断点续拉、每 50 天落盘）→ `data/universe_snapshots.csv` → instruments 股池 | `materialize_minute.py` / `materialize.py`（`build_overlay` 内置）从 1min/日频 bin 算出，写 overlay 自有 bin |
| **更新时机** | **每交易日 15:30 自动**：crontab `30 15 * * 1-5` 跑 `~/qlib_data/scripts/cron_daily.sh`；实测 day.txt mtime 15:30:02、1min bin 15:48 | **每交易日 08:30 自动**（2026-09-11 起）：crontab `30 8 * * 1-5` 跑 `scripts/cron_update_universe.sh` → `update_universe.py`（增量拉名单 + 显式拉当日盘前快照，恰好 100 行校验门禁，flock 防并发，日志 `logs/cron_universe.log`）；交易日 09:00 S1 `universe` 子命令兜底 | **手动**：数据更新后 / 改因子公式后重跑 `build_overlay`（或单独 `materialize_minute.py`）。盘中生产**不用**物化（S4 实时组装），回放实时算因子也不依赖，仅重训延伸窗口需要 |
| **当前截止** | 2026-09-11（当日已更新） | 2026-09-11（cron 已接管，原瓶颈解除） | 2026-09-04（上次重建时点）← 唯一剩余手动项 |

一天的数据时间线：

```text
08:30  cron 拉当日成分快照     → universe 到 T 日（2026-09-11 起）
15:00 收盘
15:30  cron 落日频 bin         → cn_data 到 T 日
~15:48 cron 落 1min bin        → cn_data_1min 到 T 日
手动    build_overlay          → 重物化因子到行情最新（改公式/延伸重训窗口时才需要）
手动    retrain / replay       → 用上新数据（前向回放需 --allow-unreferenced-scores）
次日 09:31–09:41 生产链路      → 用实时 bar，不依赖物化层
```

两个必须记住的坑：

1. ~~重跑 `build_overlay` 不带参数不会扩展 universe~~ **已由 08:30 cron 解决**
   （2026-09-11）：名单由 `cron_update_universe.sh` 自动维护；`build_overlay`
   默认窗口 `DUMP_END="2026-07-02"` 只影响其自带的拉取步骤（此时已无可拉），
   instruments 始终按**全量缓存**重写，无需再传 `--end`。
2. **物化因子落后于行情源是常态且无害**——生产用实时 bar、回放实时算因子都不
   依赖物化层；只有滚动重训需要延伸物化窗口时才手动重跑。

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
| universe 新日期快照 | **已自动化**：08:30 cron（`scripts/cron_update_universe.sh`）增量拉取；手动备用 `python -m scripts.update_universe --end <日期>`；token 走 `QLIB_IFIND_TOKEN_FILE`（默认 `~/qlib_data/.ifind_token`，refresh 自动续期） | ✅ cron 维护 |
| 日频/1min 新日期行情 | **仓库内无更新脚本**——只读源由仓库外的数据管道维护（实测它几乎每日在更新），项目红线"不生产行情数据" | 外部自动 |
| 盘中实时分钟 bar | kline-fetcher 服务（路径 C） | 仅真实交易日 09:31–09:41 |
| T 日 label | 不是下载问题而是时间边界：**T+1 收盘后才存在** | 永远晚一天 |
| 已有但不用 | 当日 09:42 后的 229 根分钟线；883926 指数本体 bin（close 字段异常） | — |

## 5. 想把流程跑到新日期（操作顺序）

```bash
# ① 确认外部管道已更新行情（日历最后一行 ≥ 目标日）
tail -1 ~/.qlib/qlib_data/cn_data/calendars/day.txt
# ② 重物化因子（universe 已由 08:30 cron 自动维护，无需 --end）
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
