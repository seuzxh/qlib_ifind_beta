# CLAUDE.md

This file guides Claude Code when working in this repository.

# qlib\_ifind\_beta

基于 qlib 的 A 股因子挖掘项目。行情数据直接消费 [`/home/zxh/qlib_data`](../../../qlib_data)（日频）与 [`/home/zxh/cn_data_1min`](../../../cn_data_1min)（1min）两个已建好的 qlib bin 目录 —— **只读消费，不负责数据生产**。这两个目录由独立的数据项目维护（`qlib_data` 用 kline-fetcher API 拉取 + iFinD 交叉验证，工作日 15:30 cron 增量同步）。

> 注意：`ths_qlib_data`（iFinD 直连，116 天）是**另一套**并行数据基础设施，**不是**本项目的数据源。本项目用 `/home/zxh/qlib_data`（26 年深度）。

> 项目状态：**MVP 已端到端跑通**（数据工程 overlay + 时变 universe + Alpha158 + 回测 + 报告）；已本地 git 初始化（master，未接远端）。

## 工程约束（用户指定，硬性）

1. **conda 是唯一环境管理方式**。所有命令一律 `conda run -n qlib_ifind_beta <cmd>` 前缀，**禁止**直接 `python`（会落到 base env 缺 qlib，报 `No module named 'qlib...'`）。
2. **设计前先查 Qlib 文档**。任何涉及 qlib 用法/扩展的设计决策前，先用 **context7** 查 qlib 最佳实践，确保不偏离 qlib 原生扩展机制（Handler / Processor / DatasetH / Model / Strategy / Executor / Record Template）。辅以 **sequential-thinking** 做多步分解。

** 必须遵循的 qlib 分层 **

设计或修改任何模块时，先定位到 qlib 官方分层，再给出实现：

| 层级 | qlib 官方组件 | 本项目对应文件 |
|---|---|---|
| Infrastructure / DataProvider | `qlib.data.data.FeatureProvider`<br>`qlib.data.data.InstrumentProvider`<br>`qlib.data.data.CalendarProvider` | `data/ifind_provider.py` |
| Workflow / DataLoader | `qlib.data.dataset.loader.QlibDataLoader`<br>`qlib.data.dataset.loader.StaticDataLoader`<br>`qlib.data.dataset.loader.NestedDataLoader` | `data/data_loader.py` |
| Workflow / DataHandler | `qlib.contrib.data.handler.Alpha158` 等 | `data/data_loader.py` 中 DataHandler 适配 |
| Workflow / Dataset | `qlib.data.dataset.DatasetH` | 当前未实现，未来接入 |
| Workflow / Feature Engineering | `qlib.data.dataset.processor.Processor`<br>Resample1minProcessor（highfreq 示例） | `factors/` 下各因子模块 |
| Workflow / Model | `qlib.contrib.model.gbdt.LGBModel`<br>`qlib.contrib.model.highfreq_gdbt_model.HFLGBModel` | ✅ 原生 `HFLGBModel`（§50 从 LGBModel 升级，binary loss） |
| Workflow / Strategy | `qlib.contrib.strategy.TopkDropoutStrategy`<br>`qlib.contrib.strategy.EnhancedIndexingStrategy` | ✅ 子类化 [TopkDropoutStrategyTD0](qlib_ifind_beta/td0_strategy.py)（shift=1→0，T 日 9:41 成交） |
| Workflow / Backtest | `qlib.contrib.evaluate.backtest` | ✅ `SimulatorExecutor` + `PortAnaRecord`（qrun yaml `port_analysis_config`） |
| Interface / Recorder | `qlib.workflow.recorder` | `factors/recorder.py` |
| Interface / Workflow | `qrun` / `qlib.workflow` | `qrun/workflow*.yaml` |

> 注：上表「本项目对应文件」列为 MVP 早期设计映射，部分路径（Infrastructure/DataLoader 等）随实现演进已变化。**真实文件清单以 [technical-design §9](docs/technical-design.md) 为准**（overlay `data/qlib_root/` + [qlib_ifind_beta/](qlib_ifind_beta/) + [qrun/](qrun/)）。

3. **使用 sequential-thinking 做问题分析**。遇到需要拆解、推演、根因排查或多步反思的问题时，先用 **sequential-thinking** MCP 工具理清思路再行动，不要凭直觉跳步。

## 环境

| 项        | 值                                                                               |
| -------- | ------------------------------------------------------------------------------- |
| Conda 根  | `/home/zxh/miniconda3`                                                          |
| 环境名      | `qlib_ifind_beta`（Python 3.12.13 + pyqlib 0.9.7，从 `qlib` env clone）             |
| 关键依赖     | lightgbm 4.6.0 / scikit-learn 1.8.0 / pandas 2.3.3 / numpy 2.4.6 / Cython 3.2.4 |
| editable | `qlib_monitor` ← `/home/zxh/quant_projects/qlib`                                |
| 关联 env   | `qlib`（同源公共基础）/ `ifind`（**数据生产**专用，本项目不直接用）                                     |

```bash
# 基本验证
conda run -n qlib_ifind_beta python -c "import qlib; print(qlib.__version__)"
```

## 数据源

直接消费 `/home/zxh/qlib_data`（日频）与 `/home/zxh/cn_data_1min`（1min）两个 qlib bin 目录，**只读**。字段口径、复权、instruments 语义详见 [`/home/zxh/qlib_data/CLAUDE.md`](../../../qlib_data/CLAUDE.md)。

| 频率   | provider\_uri            | 日历                                                                    | 字段（7，日频/1min 一致）                                     |
| ---- | ------------------------ | --------------------------------------------------------------------- | ---------------------------------------------------- |
| day  | `/home/zxh/qlib_data`    | `calendars/day.txt`：`2000-01-04` → `2026-07-02`（**6419 个交易日，\~26 年**） | `open / high / low / close / volume / factor / vwap` |
| 1min | `/home/zxh/cn_data_1min` | `calendars/1min.txt`（1,553,640 行，每日 242 槽）                            | 同上                                                   |

- **价格后复权**（除以 `factor` 还原不复权价；指数 `factor=1.0`）。**无** **`$turn / $amount / $change / $pct_chg / $pre_close`**——因子设计只能基于这 7 个字段。
- **股票池**（`instruments/*.txt`，TSV：`code\tstart_date\tend_date`，`end_date` 为 bin 真实末日期、非 `9999` 哨兵，与 qlib 官方对齐）：
  - `all`（5748 只全 A）、`csi300`（811 行，**时变成分**）、`csi500`、`csi100`、`csi1000`
  - `concept_tzt_*`（\~400 个同花顺概念板块，`9999-12-31` 在此表示"当前仍在板块内"，是 qlib `IndexBase` 语义、合法）
- **基准指数**（features 层有 bin，可直接作 `benchmark`；qlib 大小写不敏感，写 `SH000300` 即可）：`SH000300` 沪深300 / `SH000905` 中证500 / `SH000852` 中证1000 / `SZ399001` 深证成指 / `SZ399006` 创业板指 / `SH000001` 上证综指。（`SH000016` 上证50 **缺失**。）
- **数据落点**：通过 overlay symlink farm `data/qlib_root/`（可写叠加层）只读消费 `/home/zxh/qlib_data`——symlink 复用 7 base bin + `calendars/` + `instruments/all.txt` + `SH000300` 基准，自有 `highbeta883926` 时变池 + 3 衍生 bin（`change/limit_up/limit_down`）；`provider_uri: data/qlib_root/`。详见 [universe.py](qlib_ifind_beta/universe.py) + [scripts/build\_overlay.py](scripts/build_overlay.py)。

## MCP servers

| 来源                                        | Server                | 用途                                                 |
| ----------------------------------------- | --------------------- | -------------------------------------------------- |
| 插件 `context7@claude-plugins-official`（全局） | `context7`            | 查 qlib 等库最新文档（`resolve-library-id` → `query-docs`） |
| 用户级 `~/.claude.json`（全局）                  | `sequential-thinking` | 多步问题分解与反思                                          |

**context7 排障记录**：曾因死代理（mihomo 已卸载）时代 `npx` 拉不到 `@upstash/context7-mcp`，失败被误判为 needs-auth 并缓存于 `~/.claude/mcp-needs-auth-cache.json`。直跑 `npx -y @upstash/context7-mcp` 实测正常（Context7 v3.2.2）。修复 = 清空该 cache 后重启 Claude Code。

## 工作流

沿用 superpowers：`brainstorming` → spec（`docs/superpowers/specs/`）→ plan（`docs/superpowers/plans/`）→ 实现。姊妹项目 `ths_qlib_data` 为同套流程的参考样板。

## 记忆与笔记（getnote）— 用户指定规则

**「得到大脑 / Get笔记」是本项目的记忆管理工具**。已安装 `getnote` 技能（`~/.claude/skills/getnote/`），凭证在 `~/.claude/settings.json` 的 `env`（`GETNOTE_API_KEY` / `GETNOTE_CLIENT_ID`，client\_id 为 Get笔记预注册的固定值）。

**内容落点**：项目相关信息一律记录到 Get笔记 **「同花顺高贝塔值指数增强策略」**（`note_id=1914664125624050528`，int64 当字符串处理）。

**何时主动记录**（写入不必每次询问，但写前在对话里说明一句话即可）：

- 关键设计决策、方案选型、接口/数据口径结论
- 重要进展里程碑、阶段性成果
- 调试得到的关键结论、踩坑与根因
- 明确的待办 / 下一步

**怎么记录**：

- 默认 **追加**：先 `GET /open/api/v1/resource/note/detail?id=1914664125624050528` 拉最新全文，在文末按 `## [YYYY-MM-DD] 标题` 追加条目，再 `POST /open/api/v1/resource/note/update` 写回。
- **不破坏原有正文与图片**（图片是 OSS 临时 URL，失效后无法恢复）；整段改写/删除需用户明确要求。
- **secret 不入正文**（ifind access\_token、API key 等只留在本地配置/对话，不写进笔记）。
- `save` 接口仅新建笔记，**不能用于编辑**这条主文档。

**读取记忆**：需要项目上下文时直接调 detail 接口取最新全文，**状态以 API 返回为准，不依赖记忆/上下文**。技能详细用法见 `~/.claude/skills/getnote/references/`。

## 已定

> 最新状态以 [docs/technical-design.md](docs/technical-design.md) 为准（source of truth）；本段为快速索引。

- **路径**：手写 qlib 因子（非 RDAgent）
- **范围**：全链路（因子 → 模型 → 回测 → 报告）
- **因子来源**：MVP 原生 `Alpha158`（IC≈0，日频因子与 ~1.5 天 label 尺度错配，详见 technical-design §D6 基线段）；2026-07-06 起子类化 → champion = [MinuteEnhancedHandler](qlib_ifind_beta/minute_enhanced_handler.py)（18 = 14 个 T 日 9:30-9:40 分钟因子 + 4 extra，纯分钟族 + 隔夜跳空，无日频 Alpha158）。详见 technical-design §D6 + backtest-log §22
- **架构**：`qrun` YAML + `qlib.contrib` 原生类全链路（`HFLGBModel` / `SimulatorExecutor` / `SignalRecord-SigAnaRecord-PortAnaRecord`）；2026-07-06 起因子层子类化（`Alpha158 → HighBetaAlpha158 → MinuteEnhancedHandler`，含 §3.5 L1 前视护栏）、策略层子类化（`TopkDropoutStrategy → TopkDropoutStrategyTD0`，shift=1→0 实现 T 日 9:41 成交）。模型 §50 从 LGBModel 升级为 HFLGBModel（binary loss 横截面 alpha 二分类），执行/记录仍原生。详见 technical-design §D1 标注
- **Universe**：`highbeta883926` 时变成分股池（iFinD p03473 每日快照，T 日盘前更新无前视：T 日观察池 = 883926 的 T 日在册集；2026-07-10 起生效，此前为 T-1 lag。详见 [universe.py](qlib_ifind_beta/universe.py) + technical-design §3）
- **Benchmark**：`SH000300`（883926.TI 因 iFinD `history_data` 序列不连贯暂搁置，见 technical-design §2 D5）
- **切分**：train 2024-01-01→2025-12-31 / valid 2026-01-01→2026-03-31 / test 2026-04-01→2026-07-02（仅用 2024-2026 \~2.5 年，不用 26 年全段）
- **频率**：日频 baseline（Alpha158）+ T 日 9:30-9:40 分钟因子（物化为 day.bin、Handler 层不混频）；champion = enhanced(18)@topk10/nd8（§33 策略层 sweep 双窗双赢晋升自 20/15，含成本超额 +191%）

## 待定

- ~~自定义因子起步集最终清单~~ → ✅ 已完成（champion = enhanced(18)@topk10/nd8，详见 technical-design §D6 + backtest-log §22/§33）
- 扩段评估（用 26 年全数据）/ 追加新分钟因子族 —— 待用户决策（champion 已达成「一套有效 min 因子组合」目标）


## 禁止事项

1. **禁止**在因子计算中引入未来信息
2. **禁止**把 T+1 收益作为特征输入
3. **禁止**在因子阶段做买入/卖出决策
4. **禁止**引入与 qlib 框架无关的自定义抽象层
5. **禁止**为一次性操作创建通用工具类
6. **禁止**硬编码 token、密码等敏感信息到代码中
7. **禁止**在未读现有代码的情况下直接修改模块

## 修改前 Checklist

修改任何模块前，按此检查：

- [ ] 已读取该模块现有代码
- [ ] 已确认该改动属于哪个 qlib 分层
- [ ] 已判断是否能用 qlib 内置组件实现
- [ ] 已判断是否需要新增测试
- [ ] 已确认不会引入未来信息
- [ ] 已确认不会改变因子输出 index 结构 `(instrument, datetime)`

## 测试要求

- 每次代码修改后必须运行测试
- 新增因子必须包含：正常场景、空数据、单只股票、边界时间
- 分钟因子测试必须验证 K 线数量（开盘 10 根、尾盘 20 根）
- 所有测试必须能在无网络环境下通过（ifind API 调用必须可 mock 或回退）

## 输出格式约定

当用户要求设计方案时，回复必须包含：

1. **Qlib 官方方案**：引用具体类名/模块/示例
2. **当前项目适配方案**：基于项目既有代码的改动点
3. **偏离点**：当前实现与 qlib 原生的差异
4. **未来对齐路径**：如果要切回 qlib 原生流程，需要哪些步骤
