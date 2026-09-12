---
layout: default
title: 代码导读
nav_order: 7.6
has_children: true
---

# 代码导读

帮助新开发者快速建立对仓库代码的全局认知。数据口径看[数据源与读取](../data-access.md)、
配置看[配置说明](../configs.md)、命令行看[回测与脚本解析](../backtest-scripts.md)——
本节只回答一个问题：**代码在哪里、每个模块干什么、按什么顺序读**。

## 目录树（代码区，2026-09-12 清理后，共 ~4,900 行）

```text
qlib_ifind_beta/            可复用领域库（3,445 行）
├── config.py               ★ 全部路径/常量/冻结合同的单一来源（159 行）
├── binio.py                qlib .bin 格式读写（53 行）
├── overlay.py              symlink 叠加层构建（103 行）
├── universe.py             883926 时变股池：拉取/缓存/分段（226 行）
├── materialize.py          涨跌停线等 3 个日频衍生字段（89 行）
├── minute_factors.py       ★ 18 因子纯函数（131 行，实时/物化共用）
├── materialize_minute.py   因子+辅助字段+label v2 两腿 → day.bin（399 行）
├── highbeta_handler.py     Alpha158 子类 + Dropna 前视护栏（73 行）
├── minute_enhanced_handler.py  ★ Champion Handler：只改特征清单（23 行）
├── model_ensemble.py       HFLGB/XGB 门控与混合（237 行）
├── evaluation.py           TD0 手动回测（retrain 门控用，125 行）
├── td0_strategy.py         TopkDropoutStrategyTD0（shift=0，181 行）
├── ifind.py                iFinD HTTP 客户端 + token 续期（225 行）
├── live/                   盘中生产核心
│   ├── intraday.py         ★ 六步生产全部纯函数（412 行）
│   └── historical_replay.py  生产链路的历史行情适配器（163 行）
└── realtime/
    ├── data_fetch.py       kline-fetcher 实时采集（465 行）
    └── signal.py           实时因子组装+内存推理（277 行）

scripts/                    CLI 编排（12 个入口）
qrun/                       run.py + Champion yaml（唯一现役配置）
tests/                      离线测试（10 文件 76 用例，全部 mock 网络）
```

## 依赖方向（谁 import 谁）

```text
config.py ←─────────── 被所有模块引用（零自身依赖）
binio ← overlay/materialize*/live.historical_replay
minute_factors ← materialize_minute / realtime.signal / tests（公式单一来源）
materialize(_minute) ← overlay ← build_overlay（数据基建线）
highbeta_handler ← minute_enhanced_handler ← qrun 训练（研究线）
model_ensemble / evaluation / td0_strategy ← retrain / replay（治理与回测线）
live.intraday ← intraday_production / replay（生产线，不依赖 Handler）
```

## 三条阅读路线（按目标选）

| 你想懂什么 | 路线 | 预计 |
|---|---|---|
| 数据从哪来 | `config → binio → overlay → universe → materialize_minute`（[模块精读](modules.md) 上半） | 1 小时 |
| 模型怎么训 | `minute_factors → handler 家族 → qrun/run.py → model_ensemble → evaluation/td0_strategy`（[模块精读](modules.md) 下半） | 1 小时 |
| 盘中怎么跑 | `live/intraday.py 逐函数 → scripts/intraday_production.py 子命令 → realtime/* → replay`（[生产代码](production-code.md)） | 1.5 小时 |

## 测试地图（改哪块跑哪个）

| 测试文件（用例数） | 守护的模块 | 关键断言 |
|---|---|---|
| test_minute_factors (7) | 18 因子公式 | 手算值逐位一致 |
| test_materialize_minute (13) | 物化层 | 22 bin 齐全、与日频 bin 对齐 |
| test_highbeta_handler (7) | Handler | Dropna 前视护栏行为 |
| test_td0_strategy (4) | TD0 策略 | shift=0 语义、n_drop 约束 |
| test_model_ensemble (8) | 门控 | fail-closed、防篡改校验 |
| test_intraday_production (10) | 生产六步 | 幂等 upsert、两阶段现金 |
| test_realtime_signal (12) | 实时推理 | 与物化 bin 零漂移 |
| test_retrain (9) | 滚动重训 | 窗口切分、embargo |
| test_ifind_token_loader (2) / test_1min_format (4) | 外设 | token 缓存、bin 格式 |

全量：`conda run -n qlib_ifind_beta python -m pytest -q`（当前 82 用例，
须 100% 通过，网络一律 mock）。

## 子页

1. [核心库精读](modules.md)——`qlib_ifind_beta/` 逐模块
2. [生产代码与脚本](production-code.md)——`live/`、`realtime/`、`scripts/`、`qrun/`
