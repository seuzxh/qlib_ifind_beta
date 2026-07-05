# qlib_ifind_beta

> 基于 [qlib](https://github.com/microsoft/qlib) 的 A 股因子挖掘 MVP。
> 标的：**883926（同花顺高贝塔值指数）成分股**；形态：日频 `Alpha158` 全链路（因子 → 模型 → 回测 → 报告）。
> 数据：**只读消费** [`/home/zxh/qlib_data`](../../qlib_data)（7 字段 × 26 年），不生产行情数据。

**状态**：搭建/设计阶段，MVP 已端到端跑通（已本地 git 初始化，master，未接远端）。

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
```

---

## 它是什么 / 不是什么

| ✓ 是 | ✗ 不是 |
|---|---|
| qlib 全链路因子挖掘 MVP（数据工程 + 因子 + 模型 + 回测 + 报告） | 行情数据生产者（行情由独立的 `qlib_data` 项目维护） |
| 只读消费 qlib_data + 最小 overlay 叠加 | 拷贝/重建 qlib_data |
| 纯 `qlib.contrib` 原生类全链路 | 自定义 Handler/Strategy/Exchange 子类（MVP 零子类化） |

---

## 数据源

| 来源 | 用途 | 访问 |
|---|---|---|
| `/home/zxh/qlib_data` | 行情（7 字段 × 6419 天）、instruments、calendars | 文件系统，只读，symlink 复用 |
| iFinD `quantapi.51ifind.com` | 883926 时变成分股（`data_pool p03473` 每日快照，T-1 lag） | HTTPS + token；token 复用 `qlib_data/.ifind_token`（secret） |
| `data/qlib_root/`（生成本项目） | overlay provider_uri | build_overlay 产出，`.gitignore` |

**7 字段**：`open / high / low / close / volume / factor / vwap`（后复权；无 `$turn/$amount/$change/$pct_chg/$pre_close`）。
**衍生 3 字段**（本项目物化，供 qlib Exchange 涨跌停拦截）：`change / limit_up / limit_down`。

---

## 目录速览

```
3.qlib_ifind_beta/
├── qlib_ifind_beta/        # 数据工程包（自研，唯一业务代码）
│   ├── config.py           #   集中配置（路径/字段/代码/URL）
│   ├── overlay.py          #   symlink farm
│   ├── materialize.py      #   衍生字段物化（涨跌停依赖）
│   ├── universe.py         #   883926 时变成分股（iFinD p03473，T-1 lag）
│   ├── ifind.py            #   iFinD HTTP client + token
│   ├── binio.py            #   .day.bin 原始读写
│   └── dump_index.py       #   SH883926 dump（当前未启用，见技术方案 §5）
├── scripts/build_overlay.py  # 一次性编排：建 overlay 端到端
├── qrun/                    # 全链路入口
│   ├── workflow.yaml       #   全量配置
│   ├── workflow_smoke.yaml #   烟雾测试
│   └── run.py              #   qrun 等价入口（绕本机坑）
├── docs/                    # 架构 / 技术方案文档
├── data/qlib_root/          # 生成的 overlay（.gitignore）
└── mlruns/                  # qlib 实验产物（.gitignore）
```

---

## 核心设计（详见 [docs/technical-design.md](docs/technical-design.md)）

- **纯 `qlib.contrib` 原生类**：`Alpha158` / `LGBModel` / `TopkDropoutStrategy` / `SimulatorExecutor` / `SignalRecord-SigAnaRecord-PortAnaRecord`，零自定义子类。
- **Overlay symlink farm**：只读 qlib_data 之上逐文件 symlink 7 base bin + 真实目录写 3 衍生 bin，最小写入面。
- **板块分级涨跌停**：原生 `LT_TP_EXP` 表达式 tuple（`$change >= $limit_up` / `<= $limit_down`）+ 按代码前缀物化阈值（主板 0.095 / 创·科 0.195 / 北交所 0.295），不子类化 Exchange。
- **A 股 T+1**：`TopkDropoutStrategy(hold_thresh=1, forbid_all_trade_at_limit=True)`，daily 模式原生强制。
- **Label + 撮合价**：label = `Ref($close, -2) / Ref($open, -1) - 1`（T+1 开盘买、T+2 收盘卖，用户指定）；回测 `deal_price=["$open","$close"]`（qlib 原生支持买卖不同价）——买入 open[T+1]、卖出 close[T+2]，与 label 完全对齐。
- **Benchmark**：`SH000300`（复用 qlib_data 干净 bin；883926.TI 因 iFinD history_data 序列不连贯暂搁置，第一性原理 probe 详见技术方案 §2 D5）。

---

## 文档

- 📐 [docs/architecture.md](docs/architecture.md) — 系统架构、模块划分、数据流、依赖图
- 🔄 [docs/data_flow.md](docs/data_flow.md) — 数据流程（股池准备 / overlay 落盘 / 数据过滤 / 因子计算），通俗版
- 🧭 [docs/technical-design.md](docs/technical-design.md) — 技术选型、关键设计决策、已知妥协、本机坑绕过、演进路径
- ⚙️ [CLAUDE.md](CLAUDE.md) — 工程约束（conda-only / context7 / sequential-thinking / 只读消费 / secret 纪律）

---

## 工程约束（摘要）

1. **conda-only**：`conda run -n qlib_ifind_beta <cmd>`。
2. **设计前查 qlib 文档**：涉及 qlib 用法/扩展先用 context7 查最佳实践。
3. **sequential-thinking 分析**：拆解/根因排查类问题先理清思路。
4. **只读消费 qlib_data**：不生产、不拷贝。
5. **secret 纪律**：iFinD token、API key、`.claude/settings.local.json` 凭证绝不硬编码/打印/入笔记/进 git。
