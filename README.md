# qlib_ifind_beta

> 基于 [qlib](https://github.com/microsoft/qlib) 的 A 股因子挖掘 MVP。
> 标的：**883926（同花顺高贝塔值指数）成分股**；形态：日频 baseline（`Alpha158`，IC≈0）→ **champion = enhanced(18)@n_drop=15**（14 个 T 日 9:30-9:40 分钟因子 + 4 extra，9:41 成交），全链路（因子 → 模型 → 回测 → 报告）。详见 [backtest-log §22](docs/backtest-log/2026-07-06-l1-full-backtest.md)。
> 数据：**只读消费** [`/home/zxh/qlib_data`](../../qlib_data)（7 字段 × 26 年，日频）+ [`/home/zxh/cn_data_1min`](../../cn_data_1min)（1min，分钟因子源），不生产行情数据。

**状态**：MVP 已端到端跑通 → 已演进到 **champion = enhanced(18)@n_drop=15**（test 2026-04→07 +158.86% w/cost / IC 0.0545 / ICIR 5.12 / drawdown −5.44%，详见 backtest-log §22）；本地 git 初始化（`feat/minute-factors` 分支，未接远端）。

---

## 快速开始

> ⚠️ CLAUDE.md 硬约束：**conda-only**。所有命令必须 `conda run -n qlib_ifind_beta` 前缀；直接 `python` 会落 base env 报 `No module named 'qlib'`。

```bash
# 0. 验证环境
conda run -n qlib_ifind_beta python -c "import qlib; print(qlib.__version__)"   # → 0.9.7

# 1. 构建 overlay（一次性，幂等可重跑）—— symlink qlib_data + 物化涨跌停衍生字段 + 拉 883926 时变成分股（p03473 每日快照 T-1 lag）
conda run -n qlib_ifind_beta python -m scripts.build_overlay

# 2. 烟雾测试（2025 子窗口，快速跑通全链路）
conda run -n qlib_ifind_beta python qrun/run.py qrun/workflow_smoke.yaml

# 3. 全量 MVP（train 2024-01-01→2025-12-31 / valid 2026-Q1 / test 2026-04-01→2026-07-02）
conda run -n qlib_ifind_beta python qrun/run.py qrun/workflow.yaml
#    产物 → mlruns/<experiment_id>/<recorder_id>/{pred.pkl, label.pkl, IC, nav, …}

# 4. 【champion 复现】物化分钟因子（14 baseline + 4 extra + price_941/change_941）→ 跑 enhanced
conda run -n qlib_ifind_beta python scripts/materialize_minute.py
conda run -n qlib_ifind_beta python qrun/run.py qrun/workflow_minute_enhanced.yaml   # n_drop=15
```

---

## 它是什么 / 不是什么

| ✓ 是 | ✗ 不是 |
|---|---|
| qlib 全链路因子挖掘 MVP（数据工程 + 因子 + 模型 + 回测 + 报告） | 行情数据生产者（行情由独立的 `qlib_data` 项目维护） |
| 只读消费 qlib_data + 最小 overlay 叠加 | 拷贝/重建 qlib_data |
| `qlib.contrib` 全链路 + 因子/策略子类化 | 自定义 Exchange 子类（涨跌停用原生 `LT_TP_EXP` 表达式，零子类化） |

---

## 数据源

| 来源 | 用途 | 访问 |
|---|---|---|
| `/home/zxh/qlib_data` | 行情（7 字段 × 6419 天）、instruments、calendars | 文件系统，只读，symlink 复用 |
| iFinD `quantapi.51ifind.com` | 883926 时变成分股（`data_pool p03473` 每日快照，T-1 lag） | HTTPS + token；token 复用 `qlib_data/.ifind_token`（secret） |
| `data/qlib_root/`（生成本项目） | overlay provider_uri | build_overlay 产出，`.gitignore` |

**7 字段**：`open / high / low / close / volume / factor / vwap`（后复权；无 `$turn/$amount/$change/$pct_chg/$pre_close`）。
**overlay 每股 30 bins** = 7 base（symlink qlib_data）+ 3 衍生（`change / limit_up / limit_down`，供 Exchange 涨跌停）+ 14 分钟因子 + 4 enhanced extra（`vol_vs_yest_t2/t3/t5` + `overnight_gap`）+ `price_941`（9:41 成交价）+ `change_941`（9:41 涨跌幅，涨跌停 buy 表达式）。

---

## 目录速览

```
3.qlib_ifind_beta/
├── qlib_ifind_beta/               # 数据工程 + 因子/策略包（自研）
│   ├── config.py                  #   集中配置（路径/字段/分钟 slot 映射）
│   ├── overlay.py                 #   symlink farm
│   ├── materialize.py             #   衍生字段物化（change/limit_up/limit_down）
│   ├── materialize_minute.py      #   14 分钟因子 + 4 extra + price_941/change_941 物化
│   ├── minute_factors.py          #   T 日 9:30-9:40 分钟因子定义
│   ├── universe.py                #   883926 时变成分股（iFinD p03473，T-1 lag）
│   ├── highbeta_handler.py        #   HighBetaAlpha158（Alpha158 子类，v3 rolling + L1 护栏）
│   ├── minute_only_handler.py     #   MinuteOnlyHandler（m14 实验分支）
│   ├── minute_enhanced_handler.py #   MinuteEnhancedHandler（★ champion，18 因子）
│   ├── td0_strategy.py            #   TopkDropoutStrategyTD0（shift=1→0，9:41 成交）
│   ├── ifind.py / binio.py / dump_index.py   #  iFinD client / bin 读写 / index dump
├── scripts/build_overlay.py       # 一次性编排：overlay 端到端
├── scripts/materialize_minute.py  # 仅重物化分钟因子（改公式后免重拉 universe）
├── scripts/make_report.py         # 回测报告汇总
├── qrun/                          # 全链路入口
│   ├── workflow.yaml              #   MVP（Alpha158）
│   ├── workflow_minute_enhanced.yaml   # ★ champion（enhanced(18)@n_drop=15）
│   ├── workflow_smoke.yaml        #   烟雾测试
│   └── run.py                     #   qrun 等价入口（绕本机坑）
├── docs/                          # 架构 / 技术方案 / backtest-log
├── data/qlib_root/                # 生成的 overlay（.gitignore）
└── mlruns/                        # qlib 实验产物（.gitignore）
```

---

## 核心设计（详见 [docs/technical-design.md](docs/technical-design.md)）

- **`qlib.contrib` 全链路 + 因子/策略子类化**：模型/执行/记录仍原生（`LGBModel` / `SimulatorExecutor` / `SignalRecord-SigAnaRecord-PortAnaRecord`）；因子层 `Alpha158 → HighBetaAlpha158 → MinuteEnhancedHandler`（champion，含 L1 前视护栏）、策略层 `TopkDropoutStrategy → TopkDropoutStrategyTD0`（9:41 成交）。Exchange 用原生 `LT_TP_EXP` 不子类化。
- **Overlay symlink farm**：只读 qlib_data 之上逐文件 symlink 7 base bin + 自有目录写 23 个衍生/分钟 bin（每股 30 bins），最小写入面。
- **板块分级涨跌停**：原生 `LT_TP_EXP` 表达式 tuple（`$change >= $limit_up` / `<= $limit_down`）+ 按代码前缀物化阈值（主板 0.095 / 创·科 0.195 / 北交所 0.295），不子类化 Exchange。
- **A 股 T+1**：`TopkDropoutStrategyTD0(hold_thresh=1, forbid_all_trade_at_limit=True)`，daily 模式原生强制。
- **Label + 撮合价**（**冻结口径，禁止修改**）：label = `Ref($close, -1) / $price_941 - 1`（T+1 收盘 / T 日 9:41 价 − 1，~1.5 天 horizon）；回测 `deal_price=["$price_941","$close"]`——买入 9:41 价[T]、卖出 close[T+1]，`TopkDropoutStrategyTD0` shift=1→0 实现 T 日 9:41 成交，与 label 完全对齐。
- **Benchmark**：`SH000300`（复用 qlib_data 干净 bin；883926.TI 因 iFinD history_data 序列不连贯暂搁置，第一性原理 probe 详见技术方案 §2 D5）。

---

## 文档

- 📐 [docs/architecture.md](docs/architecture.md) — 系统架构、模块划分、数据流、依赖图
- 🔄 [docs/data_flow.md](docs/data_flow.md) — 数据流程 + 全流程图（股池 / overlay / 因子 / 训练 / 回测 / 报告），通俗版
- 🧭 [docs/technical-design.md](docs/technical-design.md) — 技术选型、关键设计决策、已知妥协、本机坑绕过、演进路径
- ⚙️ [CLAUDE.md](CLAUDE.md) — 工程约束（conda-only / context7 / sequential-thinking / 只读消费 / secret 纪律）

---

## 工程约束（摘要）

1. **conda-only**：`conda run -n qlib_ifind_beta <cmd>`。
2. **设计前查 qlib 文档**：涉及 qlib 用法/扩展先用 context7 查最佳实践。
3. **sequential-thinking 分析**：拆解/根因排查类问题先理清思路。
4. **只读消费 qlib_data**：不生产、不拷贝。
5. **secret 纪律**：iFinD token、API key、`.claude/settings.local.json` 凭证绝不硬编码/打印/入笔记/进 git。
