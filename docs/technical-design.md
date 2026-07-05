# 技术方案文档 · qlib_ifind_beta

> 配套 [architecture.md](architecture.md)（as-is 架构）。本文聚焦**技术选型理由、关键设计决策、qlib 原生机制对齐、工程约束、已知妥协与本机坑绕过、演进路径**。
> 全部决策均有代码或源码出处佐证；标 ⚠️ 的为已接受妥协或待办。

---

## 1. 技术选型

| 维度 | 选型 | 理由 |
|---|---|---|
| 因子框架 | **qlib 0.9.7（pyqlib）+ `qlib.contrib`** | CLAUDE.md 指定；最成熟的 A 股量化全链路；contrib 提供开箱即用 Alpha158/LGBModel/TopkDropout。 |
| 模型 | **LGBModel**（lightgbm 4.6） | GBDT 是 qlib 默认强基线；表格因子 + 日频场景的工业标配。 |
| 数据底座 | **只读 `/home/zxh/qlib_data`** | 26 年深度、7 字段、instruments/calendars 齐备，由独立数据项目维护；本项目只消费不生产。 |
| 数据接入策略 | **overlay symlink farm**（非拷贝/非 symlink farm 全量） | 只读源不可写，又需追加 3 个衍生 bin；逐文件 symlink 复用 base bin（零拷贝、与源同步）+ 真实目录写衍生 bin，是改动最小、最不易腐化的形态。 |
| 外部行情源 | **iFinD `quantapi`**（仅取成分股 p03473） | 883926 成分股 qlib_data 无；iFinD token 复用 qlib_data 既有刷新链路，零额外凭证。 |
| 环境 | **conda env `qlib_ifind_beta`**（Python 3.12.13） | CLAUDE.md 硬约束；从 `qlib` env clone，保证 pyqlib 版本一致。 |
| 标的范围 | **883926 成分股 + SH000300 benchmark** | 用户指定标的；benchmark 选择见 §2 D5。 |
| 频率 | **日频** | qlib 最成熟、分析层无坑；1min 留作后续高频扩展。 |

---

## 2. 关键设计决策

### D1 · 纯 `qlib.contrib` 原生类，零自定义子类
**决策**：因子（`Alpha158`）、模型（`LGBModel`）、策略（`TopkDropoutStrategy`）、执行（`SimulatorExecutor`）、记录（`SignalRecord/SigAnaRecord/PortAnaRecord`）全部用 contrib 原生类，MVP 唯一自研代码是数据工程脚本。

**理由**：MVP 目标是「跑通全链路、产出 IC/回测/报告」，而非因子创新。原生类覆盖了 Handler/DatasetH/Model/Strategy/Executor/Record 全部扩展点，零子类化 = 零维护面、零版本耦合风险。

**qlib 原生机制对齐**（CLAUDE.md「不偏离原生扩展」凭证）：

| qlib 扩展点 | 本项目用的原生类 | module_path |
|---|---|---|
| Handler | `Alpha158` | `qlib.contrib.data.handler` |
| Dataset | `DatasetH` | `qlib.data.dataset` |
| Model | `LGBModel` | `qlib.contrib.model.gbdt` |
| Strategy | `TopkDropoutStrategy` | `qlib.contrib.strategy.signal_strategy` |
| Executor | `SimulatorExecutor`（PortAnaRecord 默认） | `qlib.backtest` |
| Exchange 涨跌停 | `LT_TP_EXP`（原生表达式型） | `qlib.backtest.exchange` |
| Record | `SignalRecord` / `SigAnaRecord` / `PortAnaRecord` | `qlib.workflow.record_temp` |

### D2 · Overlay symlink farm（最小写入面）
**决策**：建 `data/qlib_root/` 作 provider_uri；calendars 整目录 symlink；instruments 目录里 `all.txt` symlink + 自有 market 文件真实；features 逐文件 symlink 7 base bin + 真实写 3 衍生 bin。

**理由**：qlib_data 只读，不能往里塞衍生 bin；全量拷贝浪费且与源脱钩；纯 symlink farm 又无法满足衍生 bin 的写入需求。逐文件粒度的混合方案是「最小可写面」——只对必须改的字段开真实目录，其余零拷贝复用。

**鲁棒性依据**（qlib 源码确认）：`FileFeatureStorage.__getitem__` 缺失 bin 返回空 Series（全 NaN）不崩；`FileInstrumentStorage.check()` 缺 market 文件抛 ValueError → market 文件必须先就位（build_overlay step 3 保证）。

### D3 · 涨跌停用原生 `LT_TP_EXP` 表达式 tuple（板块分级，不子类化 Exchange）
**决策**：`exchange_kwargs.limit_threshold = ("$change >= $limit_up", "$change <= $limit_down")`，配合每只票按板块物化的常量阈值 `limit_up/limit_down`，实现主板 ±10%、创业板/科创板 ±20%、北交所 ±30% 的分级涨跌停拦截。

**理由**：qlib 原生 `limit_threshold` 支持三种形态——全局 float（`LT_FLT`）、表达式 tuple（`LT_TP_EXP`，`exchange.py:46/258-292`）。A 股板块档位各异，全局 float 不适用；`LT_TP_EXP` 经 `D.features` 求值为 per-instrument 布尔列（True = 拦截），**无需子类化 Exchange**，是原生支持的最优解。

**阈值取「略低于名义限」**（0.095 而非 0.10）：与 `LT_FLT` 用 `.ge()` 一致，规避限价处的浮点边界（命中涨停时 `$change≈+0.099…`，`>= 0.095` 为 True → 拦截）。

**buy/sell 语义**：buy 表达式 True（`$change >= $limit_up`，封涨停）→ 禁买；sell 表达式 True（`$change <= $limit_down`，封跌停）→ 禁卖。

⚠️ **ST/*ST 未区分**（±5% → 0.045）：7 字段数据无 ST 标记；升级路径需 iFinD ST 状态接入，把常量阈值升级为按日 bin（见 §8）。

### D4 · T+1 用 `TopkDropoutStrategy(hold_thresh=1)` 原生强制
**决策**：`hold_thresh=1`（daily 模式单位 = 天）+ `forbid_all_trade_at_limit=True`。

**理由**：daily 模式下 `hold_thresh` 单位是天，`hold_thresh=1` 表示持仓至少 1 天才能卖，即 A 股 T+1 的原生实现；无需自定义 Strategy。`forbid_all_trade_at_limit` 依赖 D3 的 `$change + 板块阈值` 实现封板禁交易。

**参数**：`topk=20`（883926 约 100 只成分股，取 20）；`n_drop=5`（⚠️ 换手偏高，跑完按 PortAnaRecord 换手率再调，可降到 2-3）。

### D5 · Benchmark 从 SH883926 切到 SH000300
**决策**：`BENCHMARK = "SH000300"`，复用 qlib_data 干净 bin（`link_stock + materialize`），不再 dump 883926.TI。

**理由**（第一性原理 probe，2026-07-05）：iFinD `history_data` 返回的 `883926.TI` 序列**不连贯**——
- `close` 4 年内从 ~20k 漂移到 ~2.76M，2026-05-25 当日 −99.96% 跳变；
- `vwap` 稳定在 4~88 正常区间，与 `close` 相差 2-3 个数量级（decoupled）；
- 排除参数因素：`functionpara` 的 `CPS`（0/1/省略）三组返回**逐字节相同**，证明非 CPS 致错。

这是 `883xxx` 指数在 iFinD 该端点上的字段级属性（"883926 是指数，所以值是正确的"——但该接口对 883xxx 的 close/vwap 返回与可消费的 benchmark 口径不符）。用户决策：benchmark 暂用 SH000300（603 个干净交易日，无病态）。883926.TI 的 dump 逻辑保留在 [dump_index.py](../qlib_ifind_beta/dump_index.py) 待后续。

### D6 · 7 字段约束下的因子设计
**决策**：MVP 用原生 `Alpha158`（仅依赖 `$open/$high/$low/$close/$volume/$vwap/$factor`，与 qlib_data 7 字段完全兼容，零自定义因子）。

**理由**：qlib_data 无 `$turn/$amount/$change/$pct_chg/$pre_close`；Alpha158 默认因子集不依赖这些缺失字段，开箱即用。自定义因子起步集（围绕 7 字段、剔除上述依赖）作为 MVP 验证后的后续方向（见 §8）。

⚠️ **基线诚实结果**：默认 Alpha158 在 883926 高贝塔成分股池（时变，每日约 100 只、池子每日换 ~90%）上，test 窗口 IC≈0（略负），模型在 iter 5/200 早停。这是 baseline 的客观表现，非 bug；因子增强是后续方向（§8）。

---

## 3. Label 与切分

**Label**（用户指定，2026-07-05）：`Ref($close, -2) / Ref($open, -1) - 1`，即 **T+1 开盘买入、T+2 收盘卖出**的收益。这是 Alpha158 默认 label（`Ref($close,-2)/Ref($close,-1)-1`，close[T+1]→close[T+2]）的变体——分母从 `$close[T+1]` 换成 `$open[T+1]`，更贴近「T 日收盘出信号 → T+1 开盘成交」的实际交易节奏。在 `data_handler_config.label` 以 list 形式传入（context7 确认 Alpha158 原生支持）。

**deal_price 口径**（用户 2026-07-05 确认；同日升级为二元）：`exchange_kwargs.deal_price=["$open","$close"]`——qlib `Exchange` 原生支持买卖不同价（`exchange.py:44` 签名 `Union[str, Tuple[str,str], List[str]]`，`L157-164` 把 `tuple/list` 拆成 `(buy_price, sell_price)`；非 RL 专属，daily `SimulatorExecutor` 同样支持）。买入 `open[T+1]`、卖出 `close[T+2]`，与本 label **完全对齐**：T 日收盘出信号 → T+1 开盘成交买入，持仓 1 天（`hold_thresh=1`）→ T+2 收盘成交卖出。

✅ **卖价已对齐**（2026-07-05 升级解决）：此前 `deal_price` 为单值时卖价为 `open[T+2]`，与 label 卖价 `close[T+2]` 存在残余不一致（曾记为「固有差异、需自定义 Executor」）。源码核查推翻该判断——qlib `Exchange` 原生支持二元 `deal_price=(buy, sell)`，daily 模式同样适用；改为 `["$open","$close"]` 后，买卖两端均与 label 严格一致，零子类化。

**切分**（用户指定，仅用 2024-2026，不用 26 年全段）：

| 段 | 区间 | 用途 |
|---|---|---|
| 数据总窗 | 2024-01-01 → 2026-07-02（~620 交易日） | Alpha158 fetch/fit 范围 |
| train | 2024-01-01 → 2025-12-31（~2 年） | 最大化训练样本 |
| valid | 2026-01-01 → 2026-03-31 | LGBModel 早停 |
| test | 2026-04-01 → 2026-07-02 | out-of-sample 回测 + IC |
| backtest | 2026-04-01 → **2026-07-01** | ⚠️ 比 test 少 1 天，见 §6 坑 3 |

**时变池已消除幸存者偏差**（2026-07-05 升级）：[universe.py](../qlib_ifind_beta/universe.py) 按每日 p03473 快照构建 T-1 lag 时变 instruments（T 日观察池 = 883926 的 T-1 在册集），缩窗 2.5 年不再含 hindsight（见 §5 C1）。

---

## 4. 工程约束（用户硬性，CLAUDE.md）

1. **conda-only**：所有命令 `conda run -n qlib_ifind_beta <cmd>` 前缀；禁直接 `python`（落 base env 报 `No module named 'qlib'`）。
2. **设计前查 qlib 文档**：涉及 qlib 用法/扩展前用 context7 查最佳实践，确保不偏离原生扩展机制。
3. **sequential-thinking 做问题分析**：需拆解/推演/根因排查时先理清思路。
4. **只读消费 qlib_data**：不负责数据生产、不做 symlink farm 全量拷贝。
5. **secret 纪律**：iFinD `refresh_token`/`access_token`、API key、`.claude/settings.local.json` 内凭证**绝不硬编码/打印/入笔记/进 git**；token 经 `load_refresh_token()`/`get_access_token()` 运行时获取。

---

## 5. 已知妥协（flag，MVP 接受）

| 编号 | 妥协 | 影响 | 触发条件/升级路径 |
|---|---|---|---|
| ~~C1~~ | ~~幸存者偏差~~ → **已于 2026-07-05 解决** | — | ✅ p03473 历史 `iv_date` 实测通过 + [universe.py](../qlib_ifind_beta/universe.py) 时变 T-1 lag instruments 已落地（每日快照→连续段→+1 交易日 shift），静态池 hindsight 已消除。注：883926 是每日重平衡高贝塔榜（每日 ~80-90% 换手），时变池每日约 100 只、池子每日换血 |
| C2 | **ST/*ST 涨跌停未区分**（±5%） | ST 股按各自板块 ±10/20/30% 处理 | 接入 iFinD ST 状态 → `limit_up/down` 升级为按日 bin |
| C3 | **Alpha158 baseline IC≈0** | 策略 alpha 不足 | 自定义因子增强（§8） |
| C4 | **`dump_index.py` 当前 dead code** | benchmark 走 SH000300 后未被引用 | 883926.TI 数据问题解决后可复活 |
| C5 | **`n_drop=5` 换手偏高** | 交易成本侵蚀 | 跑完按 PortAnaRecord 换手率调到 2-3 |
| C6 | **git 未初始化** | 无版本追踪 | 时机待用户定 |

---

## 6. 本机环境坑与绕过

### 坑 1 · `LT_TP_EXP` 要 Python tuple，YAML 给的是 list
- **现象**：`Exchange._get_limit_type`（`exchange.py:264`）用 `isinstance(limit_threshold, tuple)` 判分支；YAML 的 `[a,b]` 被 `ruamel.yaml` 加载成 `list` → 落 `NotImplementedError`。
- **绕过**：[run.py](../qrun/run.py) 的 `_coerce_limit_threshold()` 在 `qlib.init` 前把 `PortAnaRecord.kwargs.config.backtest.exchange_kwargs.limit_threshold` 从 `list` 转 `tuple`。

### 坑 2 · mlflow 3.12.0 file-store maintenance mode
- **现象**：本机 mlflow 3.12 把 file store 列 maintenance，`task_train` 落 mlflow 抛 `MlflowException`。
- **绕过**：[run.py](../qrun/run.py) 在 `import qlib` 前预置 `MLFLOW_ALLOW_FILE_STORE=true`；并把 `exp_manager.kwargs.uri` 指向 `file:<cwd>/mlruns`。

### 坑 3 · qlib 末日历越界（`get_step_time` IndexError OOB）
- **现象**：回测 `end_time = 2026-07-02`（`calendars/day.txt` 最后一行）时，qlib `utils.py:131` 的 `get_step_time` 访问 `calendar[index+1]` → `IndexError: index 6419 out of bounds for axis 0 with size 6419`。qlib 期望日历末尾有「未来缓冲日」。
- **绕过**：[workflow.yaml](../qrun/workflow.yaml) 把回测 `end_time` 收到 `2026-07-01`（少最后 1 天）；dataset `test` 段仍到 `2026-07-02`，pred/IC 覆盖全窗，仅回测模拟少 1 天。烟雾测试（2025-12-31，非末日历）不触发。
- **更彻底的备选**（未采）：扩展日历加未来缓冲日——侵入性大，待用户定。

### 坑 4 · `MLflowRecorder` 属性名
- **现象**：`run.py` 末尾 `recorder.recorder_id` 抛 `AttributeError`（pipeline 已在此前完成，纯 cosmetic）。
- **绕过**：`getattr(recorder, "recorder_id", None) or getattr(recorder, "id", None)`。

### 坑 5 · `D.features()` 关键字参数名
- **现象**：`BaseProvider.features()` 用 `start_time/end_time`，不是 `start/end`。
- **绕过**：调用方一律用 `start_time/end_time`。

---

## 7. qrun 入口为什么是 `run.py` 而非裸 `qrun`
两个本机坑（§6 坑 1、坑 2）必须在 qlib 初始化前后夹击处理，裸 `qrun` 命令没有插桩点。`run.py` 等价于 `qlib.cli.run.workflow`（`qlib_init → task_train`），仅在两端各加一道修正：

```
设 MLFLOW_ALLOW_FILE_STORE → import qlib → 加载 YAML → limit_threshold list→tuple
→ qlib.init(exp_manager uri=file:./mlruns) → task_train → save_objects(config)
```

---

## 8. 演进路径（后续方向，非当前实现）

| 方向 | 触发条件 | 改动面 |
|---|---|---|
| ~~时变成分股池~~（去幸存者偏差） | ✅ **已完成 2026-07-05** | [universe.py](../qlib_ifind_beta/universe.py)：每日 p03473 快照→连续段→T-1 shift→时变 instruments（每日 ~80-90% 换血，详见 §5 C1） |
| **ST 涨跌停时变阈值** | 接入 iFinD ST 状态 | [materialize.py](../qlib_ifind_beta/materialize.py)：`limit_up/down` 从常量升级为按日 bin |
| **自定义因子起步集** | MVP baseline 验证后 | 子类 `Alpha158`（如 `Alpha158HighBeta(Alpha158)`）追加围绕 7 字段的因子（MOM/波动/形态等），剔除 `$turn/$amount` 依赖 |
| **883926 benchmark 复活** | `883926.TI` 数据口径问题解决 | 启用 [dump_index.py](../qlib_ifind_beta/dump_index.py)，`BENCHMARK` 改回 `SH883926` |
| **分钟频因子** | 日频 pipeline 成熟后 | 新增 1min provider_uri（消费 `/home/zxh/cn_data_1min`），高频 Handler/Strategy |
| **换手率调参** | 全量回测产出 PortAnaRecord 换手指标 | `workflow.yaml` 调 `n_drop` 5→2/3 |
| **git 初始化** | 用户指定时机 | 整库 `git init`（确认 `.gitignore` 已盖 `data/`、`mlruns/`、`.claude/settings.local.json`） |
| **末日历越界彻底修复** | 需找回 2026-07-02 回测日 | 扩展 `calendars/day.txt` 加未来缓冲日（侵入 qlib_data 口径，需评估） |
| **Path A · 衍生字段表达式化**（消 materialize） | overlay 维护成本上升时 | 自定义 `Limit` 算子（按代码前缀返回阈值）+ `$change` 走原生表达式；**未 probe、未定**，详见 §8.1 |

### §8.1（可选）Path A · 衍生字段表达式化

> 状态：**未 probe、未定**。当前 [materialize.py](../qlib_ifind_beta/materialize.py) 工作正常，这是「锦上添花」级简化思路，不阻塞 MVP。

**现状**：3 个衍生字段（`$change / $limit_up / $limit_down`）由 [materialize.py](../qlib_ifind_beta/materialize.py) 预算好，物化成 `.day.bin` 放进 overlay（D2/D3）。回测时 qlib 像读普通行情一样读它们——「提前洗好切好放冰箱」。

**Path A 的想法**：不物化，让 qlib 取数时按表达式**现算**——「现洗现切」。

- `$change` 是纯表达式：`($close/$factor) / Ref($close/$factor, 1) - 1`。qlib 原生算子（Ref + 四则运算）直接能算，零自定义。
- `$limit_up / $limit_down` 需要**自定义算子** `Limit`：按当前票的代码前缀返回阈值（主板 0.095 / 创·科 0.195 / 北交所 0.295）。qlib 内置 `ChangeInstrument` 算子是先例（同样是「按当前 instrument 返回常量」）。
- 关键依据：qlib 的 `LocalExpressionProvider.expression(...)` 是**逐票求值**的——算子执行时天然知道当前是哪只票，所以 `Limit()` 能返回该票对应板块的阈值。注册走 `OpsWrapper` + `register_all_ops`。

**收益**：

1. 删掉 [materialize.py](../qlib_ifind_beta/materialize.py)，3 个衍生 bin 不再需要；
2. overlay 进一步瘦身——每只票的 `features/<code>/` 真实目录可全部去掉（7 个 base 直接指 qlib_data），overlay 只剩 `instruments/highbeta883926.txt` 一个真实文件；
3. 口径永远最新——不会出现「qlib_data 更新了但衍生 bin 没重跑」的漂移。

**限制**：

- **消不掉 overlay 本身**——自有 instruments 文件还得放，qlib `provider_uri` 单根约束还在。Path A 只把 overlay 从「目录级」缩到「单文件级」。
- **打破 D1（零自定义）**——需要注册自定义算子 `Limit`。这是 Path A 唯一引入的自定义代码。
- **关键技术风险未验证**：`limit_threshold` 的 tuple 表达式是 Exchange 在每只票上求值的，自定义算子能否在那条路径上正确拿到 instrument **未 probe**。若不行，Path A 对 `$limit_up/$limit_down` 不成立（`$change` 仍可单独表达式化）。

**判定**：deferred。优先级低于 ST 时变阈值、自定义因子起步集。若日后 probe 通过且 overlay 维护成本上升，再回头评估。

---

## 9. 关键文件速查

| 主题 | 文件 |
|---|---|
| 全部配置 | [qlib_ifind_beta/config.py](../qlib_ifind_beta/config.py) |
| overlay 构建 | [qlib_ifind_beta/overlay.py](../qlib_ifind_beta/overlay.py) + [scripts/build_overlay.py](../scripts/build_overlay.py) |
| 衍生字段（涨跌停依赖） | [qlib_ifind_beta/materialize.py](../qlib_ifind_beta/materialize.py) + [qlib_ifind_beta/binio.py](../qlib_ifind_beta/binio.py) |
| 成分股池 | [qlib_ifind_beta/universe.py](../qlib_ifind_beta/universe.py) |
| iFinD 客户端/token | [qlib_ifind_beta/ifind.py](../qlib_ifind_beta/ifind.py) |
| 全链路配置 | [qrun/workflow.yaml](../qrun/workflow.yaml)（全量）/ [qrun/workflow_smoke.yaml](../qrun/workflow_smoke.yaml)（烟雾） |
| 入口（坑绕过） | [qrun/run.py](../qrun/run.py) |
| 工程约束 | [CLAUDE.md](../CLAUDE.md) |
