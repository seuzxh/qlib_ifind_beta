---
nav_exclude: true
---

# 883926 高贝塔策略 · 分钟因子设计

> **状态**：定稿（v2 修订）
> **日期**：2026-07-06
> **范围**：在现有日频 Alpha158 baseline 之上，叠加 T 日 9:30~9:40 分钟频因子，与日频因子共同训练；信号 T 日 9:40 产出、**T 日 9:41 撮合买入**。
>
> **v2 修订（2026-07-06）**：v1 隐含假设"T 日能直接用 `pred[T]` 撮合"，但 qlib `TopkDropoutStrategy.generate_trade_decision` 硬编码 `shift=1`（[signal_strategy.py:142](file:///home/zxh/miniconda3/envs/qlib_ifind_beta/lib/python3.12/site-packages/qlib/contrib/strategy/signal_strategy.py#L142)），强制 `pred[T]→T+1 执行`。v1 的 label/deal_price 在 shift=1 下会错位一天。v2 三处修复：
> 1. **特征 lag**：158 个 Alpha158 field 全包 `Ref(...,1)` → 日频特征压到 T-1（v1 Handler 代码示例漏写，已修正）。
> 2. **shift=0 子类**：新增 `TopkDropoutStrategyTD0`（[§撮合时序](#撮合时序shift0-strategy-子类)），让 `pred[T]→T 日执行`。
> 3. **9:41 涨跌停**：buy 用 `$change_941`（9:41 时刻涨跌幅，无前视），sell 用 `$change`（T+1 收盘全天已知），跌停不拦买入。

---

## TL;DR

- 在 Alpha158（158 日频特征，**全部 `Ref(*,1)` lag 到 T-1**）之上新增 **14 个分钟频衍生因子**（T 日 9:30~9:40，不 lag），围绕高贝塔成分股"开盘启动"特点设计。
- 分钟因子 + `$price_941` + `$change_941` 由独立物化脚本从 `cn_data_1min` 计算，落成 day.bin 进 overlay；qlib 仍只消费日频 overlay，**不挂双频**。
- Handler 子类 `HighBetaAlpha158` 返回 `[Ref(f,1) for f in 158] + 14 个 $minute`。
- **撮合时序靠 `TopkDropoutStrategyTD0`（shift=0 子类）实现**：`pred[T]`（T 日 9:40 产出）→ T 日 9:41 买入 `$price_941[T]` → T+1 收盘卖出 `$close[T+1]`；label = `Ref($close,-1)/$price_941 - 1`。
- **涨跌停**：buy 表达式 `$change_941 >= $limit_up`（9:41 已封涨停 → 禁买，无前视），sell 表达式 `$change <= $limit_down`（全天封跌停 → 禁卖）。
- 全程无前视：特征 ≤ T 日 9:40，撮合价 T 日 9:41。

---

## Context

现有 baseline（纯日频 Alpha158 + LGBModel）test 段 IC ≈ 0、超额为负（见 [data_flow.md](../../data_flow.md) 全流程图快照）。根因：Alpha158 默认周期与日频 label + 高贝塔池"每日重平衡、开盘启动"特性不匹配，信噪比低。

883926 是每日重平衡的高贝塔榜，成分股特点是**开盘后短期加速、启动强势**。日频因子捕捉不到这种日内启动形态。本设计叠加 T 日 9:30~9:40 的 1min K 线因子，与日频因子共同训练，以 T 日 9:41 close 为买入信号价。

---

## 已定决策

| 项 | 决策 | 备注 |
|---|---|---|
| 时序处理 | 混频：分钟用 T 日、日频用 T-1（`Ref(*,1)` lag1） | 两者都 ≤ T 日 9:40，T 日 9:41 撮合无前视 |
| 因子维度 | 4 维全选：启动动量 / 加速度（避追高）/ 拉升形态 / 量能放量 | + 整段基准 + 跨日量能 = 14 个 |
| 周期 | 1min / 3min / 5min 滑动窗口 | 落在 9:30~9:40 共 10 根真实 bar（slot 1-10）上 |
| 实现方案 | A：物化 day.bin + 子类 Alpha158 | 不用 NestedDataLoader（详见 §物化架构） |
| 模型 | LGBModel（不变） | HFLGBModel 非多频 model，留作后续可选 |
| **撮合时序** | **`TopkDropoutStrategyTD0`（shift=0 子类）** | **v2 命门**：打破 qlib 默认 shift=1 的 T+1 执行，实现 pred[T]→T 日 9:41 买入（[§撮合时序](#撮合时序shift0-strategy-子类)）|
| 买入价 | T 日 9:41 close（`$price_941`） | 物化为 day.bin |
| 卖出价 | T+1 收盘（`$close`） | A 股 T+1 最早 T+1 卖 |
| label | `Ref($close,-1)/$price_941 - 1` | T 日 9:41 买 → T+1 收盘卖；shift=0 下与实收益一一对齐 |
| deal_price | `["$price_941","$close"]` | buy/sell 分开，qlib 原生支持（[exchange.py:61-65](file:///home/zxh/miniconda3/envs/qlib_ifind_beta/lib/python3.12/site-packages/qlib/backtest/exchange.py#L61)）|
| **涨跌停** | **buy `$change_941 >= $limit_up`、sell `$change <= $limit_down`** | **v2**：buy 用 9:41 时刻涨跌幅（无前视），sell 用全天（T+1 收盘已知）；跌停不拦买入（撮合可行性，见 [§涨跌停](#涨跌停941-时刻-buy--全天-sell)）|
| 切分 | 不变：train 2024-01-01→2025-12-31 / valid 2026-Q1 / test 2026-04-01→2026-07-02 | |

---

## 数据基础

### cn_data_1min 结构（第一性原理 probe 验证）

- 路径：`/home/zxh/cn_data_1min`
- 1min 日历：`calendars/1min.txt`，1,553,640 行 = 6420 日 × **242 槽**，**标记 start-time**（`YYYY-MM-DD HH:MM:00`）：slot 0 = 09:30、slot 11 = 09:41、slot 241 = 15:00（probe 实证 2026-07-06）。
- 字段：7 个 `.1min.bin`（close/open/high/low/volume/factor/vwap），与日频一致。
- **日内槽位（关键，2026-07-06 probe 更正）**：每日日历 242 槽，但**真实 bar 只有 240 根**——slot 0（09:30 集合竞价占位）与 slot 121（13:00 午休边界占位）**全池全日 NaN**（8 票 × 604 日：slot 0 与 slot 121 non-NaN 全为 0；per-day bar-count median = 240.0 吻合）。slot 241（15:00）**有真实数据**（602/604 日非空），不是占位。第一根真实 bar = **slot 1**（覆盖 [09:30,09:31)，开盘连续竞价首根 bar）——`daily_open == minute_open[slot 1]` 8/8 日精确相等。故 `vol_vs_yest` 分母 `/240` 恰好等于全天真实 bar 数（非近似）。
  - **slot 1~10 = 09:30~09:40（10 根真实 bar，因子输入）**
  - **slot 11 = 09:41**（**买入价 bar，不参与因子计算**，真实有数 600/604 日）
- **日历网格对齐（影响物化）**：全池所有票的 1min.bin `start_index` 均为 1407473 = 2024-01-02 09:31（slot 1）。物化必须按**绝对日历行散射**、不可 `reshape(242)`（会逐日偏移）；特征窗口取 **slot 1-10**（跳过全 NaN 的 slot 0）、买入价取 slot 11。引用 slot 0 的因子不再存在（窗口已后移），DropnaLabel 只处理真实停牌/缺格。
- **volume 单位差异（probe 实证，影响 vol_vs_yest）**：cn_data_1min 分钟 volume 与 qlib_data 日频 volume 单位/复权不一致，跨票比值 1.0~192.7（科创板≈1、茅台 5.84、平安银行 192.7），约等于逐票累计复权因子。凡跨源混用 volume 的因子必须**两边都用 cn_data_1min 分钟量**。
- day 日历：`calendars/day.txt`（与 qlib_data 一致）。
- instruments：`all.txt`（与 qlib_data 一致）。

### 池覆盖率

highbeta883926 时变池 5116 只 unique codes，在 cn_data_1min 覆盖 5040 只（98.5%），76 只缺失（与日频 bin 缺失集一致，已退市/停牌早）。缺失票物化时写 NaN，qlib DropnaLabel 自动处理。

### 符号约定

记 T 日分钟 bar（index 已按"窗口后移 1 格"重编号，2026-07-06 更正）：
- `c_i / o_i / vol_i / h_i / l_i` = index i 的分钟 close/open/volume/high/low（i = 0..10）
- index i = slot (i+1)：**index 0..9 = slot 1..10 = 09:30~09:40（因子输入，10 根真实 bar）**，**index 10 = slot 11 = 09:41（买入价）**
- `o_0` = slot 1 open = 日开盘价（probe 实证 == daily_open）
- `mean(vol_a..b)` = index a 到 b 的 volume 均值；`max(h_a..b) / min(l_a..b)` 同理

---

## 因子集（14 个）

### A. 启动动量（一阶累计收益）—— 开盘后涨了多少

| 因子 | 公式 | 语义 |
|---|---|---|
| `startup_mom_1m` | `c9/c8 - 1` | 最后 1 分钟涨幅（尾动量） |
| `startup_mom_3m` | `c9/c6 - 1` | 最后 3 分钟涨幅 |
| `startup_mom_5m` | `c9/c4 - 1` | 最后 5 分钟涨幅 |
| `startup_total` | `c9/o0 - 1` | 整段基准：开盘（o0=slot1 open）→ 9:40 累计涨幅 |

`startup_total` 区分"冲高回落"（total 正、mom 负）与"尾盘发力"（两者都正）两种形态。

### B. 加速度（二阶，避追高核心）—— 后段收益 − 前段收益

段收益用「段首 open → 段末 close」：

| 因子 | 前段 | 后段 | 公式 |
|---|---|---|---|
| `accel_1m` | index0 (slot1) | index9 (slot10) | `(c9/o9-1) - (c0/o0-1)` |
| `accel_3m` | index0-1 (slot1-2) | index7-9 (slot8-10) | `(c9/o7-1) - (c1/o0-1)` |
| `accel_5m` | index0-3 (slot1-4) | index5-9 (slot6-10) | `(c9/o5-1) - (c3/o0-1)` |

`>0` → 加速赶顶（追高风险）；`<0` → 减速（健康）。特征不预判方向，交给模型学。

### C. 拉升形态（收盘在窗口高位还是低位）

`close_pos_W = (c9 - low_W) / (high_W - low_W)`，越接近 1 越收在最高位：

| 因子 | 窗口 |
|---|---|
| `close_pos_1m` | index9（slot10）单根的 h/l |
| `close_pos_3m` | index7-9（slot8-10）的 max(h)/min(l) |
| `close_pos_5m` | index5-9（slot6-10）的 max(h)/min(l) |

### D. 量能放量（后段均量 / 前段均量）

| 因子 | 公式 |
|---|---|
| `vol_ratio_1m` | `vol9 / mean(vol0-8)`（vol0-8 = slot1-9） |
| `vol_ratio_3m` | `mean(vol7-9) / mean(vol0-1)`（back slot8-10 / front slot1-2） |
| `vol_ratio_5m` | `mean(vol5-9) / mean(vol0-3)`（back slot6-10 / front slot1-4） |

### E. 跨日量能（绝对放量）

`vol_vs_yest = sum(vol0-9) / (prev_day_full_minute_vol / 240)`

T 日前 10 分钟（slot 1-10）分钟 volume 之和 ÷ T-1 日全天 240 根分钟 volume 之和折算到每分钟。**分子分母均来自 cn_data_1min 分钟量**（避免与 qlib_data 日频 volume 的单位/复权差异，probe 实证跨票比值 1.0-192.7），T-1 < T **无前视**。

---

## 物化架构

### 方案 A（采用）：独立读 1min.bin，qlib 不挂双频

物化脚本（`materialize_minute.py`）独立用 numpy 解析 cn_data_1min 的 1min.bin（probe 已验证读法：bin 头部 float32 `start_index` + float32 数组），按 **1min 日历的 slot 网格对齐**取数：预计算全局 `min_slots[cal_row]`（0..241），把每只票的 bin 按 `start_index` 散射到全局 `(day, slot)` 网格，**取 slot 1-10 / slot 11**（09:30-09:41，跳过全 NaN 的 slot 0）算 14 因子 + 9:41 close → 写 day.bin 进 overlay。**为什么不 `reshape(242)`**：全池 bin 普遍从 slot 1 起步（slot 0 全池全日 NaN，见 §数据基础），`arr.reshape(n,242)` 会逐日整体偏移一格、首日错位；日历网格散射按绝对 cal_row 取值，对任意 bin 头洞/缺格鲁棒（缺处 NaN）。

qlib 保持 `provider_uri=data/qlib_root, freq=day`（不变），完全不知道分钟数据存在。新增 **16 个 day.bin/票**（14 因子 + `$price_941` + `$change_941`）与现有 `change/limit_up/limit_down` 同模式。

**`vol_vs_yest` 特殊处理**：分子分母**均来自 cn_data_1min 分钟 bin**（probe 实证 qlib_data 日频 volume 与分钟 volume 单位/复权不一致，跨票比值 1.0-192.7，不可混用）。物化脚本只读 cn_data_1min：分子 = T 日 slot 1-10 volume 之和，分母 = T-1 日全天 240 根 volume 之和 / 240。

**`$change_941` 物化（v2 新增）**：与 `$price_941` 同源同生命，在 `materialize_minute.py` 内一并算（不放 `materialize.py`——[build_overlay.py:51-55](file:///home/zxh/projects/3.qlib_ifind_beta/scripts/build_overlay.py#L51) 是 derived 先、minute 后，`materialize.py` 跑时 `$price_941.bin` 还不存在）。物化时新增读 daily `factor.bin`，公式：

```
change_941[T] = (price_941[T] / factor[T]) / (close[T-1] / factor[T-1]) - 1
```

即 9:41 不复权价 相对 T-1 不复权收盘 的涨跌幅（口径与 [`compute_change`](file:///home/zxh/projects/3.qlib_ifind_beta/qlib_ifind_beta/materialize.py#L52) 同源，只把"全天 close"换成"9:41 close"）。必须不复权——除权日 factor 跳变，后复权会跳空被误判成涨跌停。首日无昨收 → NaN。

### 为什么不用 qlib 原生混频（NestedDataLoader / QlibDataLoader freq dict）

context7 查证：qlib 原生 `QlibDataLoader(freq={group:"day", group:"1min"})` 和 `NestedDataLoader(dataloader_l=[...])` 支持"读两个 freq"，但混出来的是**分钟级行**（一天 242 行）。本设计要的是**日级二维表**（送 LGBModel + 日级回测）。从分钟序列到"每日一个因子标量"的聚合 qlib 无现成类，自定义 Loader 不如物化省事。聚合放数据层（最轻、与项目已验证物化模式一致），不碰 Handler 层混频。

### 不物化方案对比（已否决）

| | 方案 A 物化（采用） | 方案 B 自定义 Loader（否决） |
|---|---|---|
| 聚合位置 | 数据层（预计算 day.bin） | Handler 层（Loader.load 内 groupby） |
| 训练 IO | 只读日频（快） | 每次 fit 重拉分钟（慢） |
| 改公式 | 重跑脚本（~1-2 分钟） | 改代码立即生效 |
| 偏离原生 | 低（与 change/limit_up 同模式） | 中（~100 行自定义 Loader） |
| 项目先例 | ✅ | ❌ |

---

## 时序模型 + 无前视核查

| 时刻 | 事件 | 数据 |
|---|---|---|
| T-1 收盘及更早 | 日频特征就绪 | qlib_data（`Ref(*,1)` lag1）|
| T 日 9:30~9:40 | 分钟特征就绪 | cn_data_1min slot 1-10（index 0-9）|
| T 日 9:40 | 信号产出（model.predict → `pred[T]`）| 特征到齐（158 lag + 14 分钟）|
| **T 日 9:41** | **撮合买入**（`TopkDropoutStrategyTD0` shift=0 拉 `pred[T]`）| `$price_941[T]`（index 10 = slot 11 close）|
| T+1 收盘 | 撮合卖出 | `$close[T+1]` |

**无前视核查**：
- 分钟特征最晚 index 9（slot 10 = 9:40）< 撮合价 index 10（slot 11 = 9:41）✓
- 日频特征 lag1（T-1）< T 日 9:41 ✓
- label 用 T+1 数据，但 label 只作训练 target，**不进特征** ✓
- shift=0：`pred[T]` 只依赖 ≤ T 日 9:40 的特征 → T 日 9:41 用它撮合，**无前视** ✓
- 实收益对齐：T 日 9:41 买 `price_941[T]`、T+1 收盘卖 `close[T+1]` → 1 日收益 = `close[T+1]/price_941[T]-1` = `label[T]` ✓

---

## 撮合时序：shift=0 Strategy 子类（v2 命门）

### 问题：qlib 默认 shift=1 强制 T+1 执行

`TopkDropoutStrategy.generate_trade_decision`（[signal_strategy.py:138-265](file:///home/zxh/miniconda3/envs/qlib_ifind_beta/lib/python3.12/site-packages/qlib/contrib/strategy/signal_strategy.py#L138)）第 142 行硬编码：

```python
pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
```

`get_step_time(step, shift=1)`（[utils.py:102-124](file:///home/zxh/miniconda3/envs/qlib_ifind_beta/lib/python3.12/site-packages/qlib/backtest/utils.py#L102)）取**前一 bar** 的信号：执行日 N 拉 `pred[N-1]`。即默认"昨晚出信号、今早执行"。本设计要"今早 9:40 出信号、9:41 执行"，必须让执行日 T 拉 `pred[T]` → `shift=0`。

### 为什么只能子类化（否决其他路径）

- **shift=1 + 改 label 偏移**：`pred[T]` 配 `label[T+1]`，SigAnaRecord 的 IC 错位一天，否决。
- **EnhancedIndexingStrategy**：[signal_strategy.py:352](file:///home/zxh/miniconda3/envs/qlib_ifind_beta/lib/python3.12/site-packages/qlib/contrib/strategy/signal_strategy.py#L352) 同样硬编码 `shift=1`，没省事。
- **index 技巧**（平移 pred 序列）：破坏 SigAnaRecord 的 pred↔label index 对齐，否决。

### 实现：TopkDropoutStrategyTD0

整段重写 `generate_trade_decision`（复制 [signal_strategy.py:138-265](file:///home/zxh/miniconda3/envs/qlib_ifind_beta/lib/python3.12/site-packages/qlib/contrib/strategy/signal_strategy.py#L138) 全文），**唯一改动**：第 142 行 `shift=1` → `shift=0`。这是 qlib 标准扩展模式（copy + 一行改），无 hook/参数可注入。

```python
# qlib_ifind_beta/td0_strategy.py
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy

class TopkDropoutStrategyTD0(TopkDropoutStrategy):
    """shift=0 版本：pred[T] → T 日执行（默认 shift=1 是 T+1 执行）。
    供 9:41 撮合设计使用：T 日 9:40 出信号、9:41 买入。其余逻辑同父类。"""
    def generate_trade_decision(self, execute_result=None):
        # ... 完整复制父类 138-265 ...
        # 唯一改动行：
        pred_start_time, pred_end_time = \
            self.trade_calendar.get_step_time(trade_step, shift=0)
        # ... 其余原样 ...
```

YAML 里 `strategy.class: TopkDropoutStrategyTD0`（+ `module_path: qlib_ifind_beta.td0_strategy`）。

---

## Handler 子类

```python
class HighBetaAlpha158(Alpha158):
    def get_feature_config(self):
        fields, names = super().get_feature_config()          # 原生 158
        lag_fields = [f"Ref({f}, 1)" for f in fields]         # v2: 全部 lag 到 T-1
        min_fields = [f"${n}" for n in MINUTE_FACTOR_FIELDS]  # 14 个，T 日当天不 lag
        return lag_fields + min_fields, names + min_fields
```

- 总特征 158 + 14 = 172，name 不冲突（`Ref` 只改 field 表达式、不改 name）。
- v1 代码示例曾漏写 `Ref` 包裹（与决策表"日频 lag1"不一致），已修正。

label 经 Alpha158 的 `label` kwarg 传入（YAML 配置，见下）。

---

## Label + deal_price

- **label**：`Ref($close, -1) / $price_941 - 1`（T 日 9:41 买 → T+1 收盘卖；shift=0 下与实收益对齐）
- **deal_price**：`["$price_941", "$close"]`（buy/sell 分开，[exchange.py:61-65](file:///home/zxh/miniconda3/envs/qlib_ifind_beta/lib/python3.12/site-packages/qlib/backtest/exchange.py#L61) 原生支持二元 list：第 0 个 buy、第 1 个 sell）

> **shift=0 依赖**：此 label/deal_price 的时序对齐**依赖 `TopkDropoutStrategyTD0`（shift=0）**。在默认 shift=1 下，buy 会变成 `price_941[T+1]`、sell 变成 `close[T+2]`，与 label 错位一天——这正是 v1 的 bug。

---

## 涨跌停：9:41 时刻 buy + 全天 sell（v2）

`Exchange` 的 `limit_threshold` 二元 tuple（[exchange.py:285-286](file:///home/zxh/miniconda3/envs/qlib_ifind_beta/lib/python3.12/site-packages/qlib/backtest/exchange.py#L285)）：第 0 个 = **buy 表达式**（True 禁买）、第 1 个 = **sell 表达式**（True 禁卖），买卖可分别用不同字段。

```yaml
limit_threshold: ["$change_941 >= $limit_up", "$change <= $limit_down"]
```

- **buy**：`$change_941`（T 日 9:41 时刻涨跌幅，[§物化架构](#物化架构)）判涨停 → 9:41 已封板 → 买不进 → 禁买。**无前视**（9:41 价已知）。
- **sell**：`$change`（全天涨跌幅）判跌停 → T+1 收盘卖出，那时 T+1 全天已知 → **无前视**。
- **跌停不拦买入**：从撮合可行性第一性原理，跌停 = 卖一档堆满卖单，挂买单立即成交 → **跌停买得进**，无需拦。涨停才买不进。若想在跌停也不买（规避崩盘票），属策略层 score 筛选，不放在 Exchange 撮合可行性这里。

---

## YAML 改动点（相对当前 workflow.yaml）

| 段 | 改动 |
|---|---|
| `qlib_init` | 不变（`provider_uri: data/qlib_root`）|
| `handler.class` | `Alpha158` → `HighBetaAlpha158`（+ `module_path: qlib_ifind_beta.highbeta_handler`）|
| `handler.kwargs.label` | `Ref($close,-2)/Ref($open,-1)-1` → `Ref($close,-1)/$price_941-1` |
| `exchange.deal_price` | `["$open","$close"]` → `["$price_941","$close"]` |
| **`strategy.class`**（v2）| `TopkDropoutStrategy` → `TopkDropoutStrategyTD0`（+ `module_path: qlib_ifind_beta.td0_strategy`）|
| **`exchange.limit_threshold`**（v2）| `["$change >= $limit_up", "$change <= $limit_down"]` → `["$change_941 >= $limit_up", "$change <= $limit_down"]` |
| 其余（model/executor/record/segments）| 不变 |

---

## 验证计划

### A. 工程正确性

| # | 验什么 | 怎么验 | 通过标准 |
|---|---|---|---|
| A1 | 物化正确性 | 抽 3 票（主板/创业板/科创板各一）× 3 日，读原 1min.bin 手算 6 个代表因子（`startup_mom_1m / accel_3m / close_pos_5m / vol_ratio_3m / vol_vs_yest / price_941`），对比 day.bin 值 | 误差 < 1e-5 |
| A1b（v2）| `$change_941` 口径 | 抽 1 票 × 3 日，手算 `(price_941/factor) / (prev_close/prev_factor) - 1` 对比 bin | 误差 < 1e-5；除权日不复权基准正确 |
| A2 | 缺失容错 | 76 只缺 bin 票 + 停牌段跑物化脚本 | 不崩；缺失时段写 NaN（DropnaLabel 处理）|
| A3 | 日历对齐 | 1min.txt 日期 vs day.txt 日期 | 交易日集合一致（1min 是 day 的细化）|
| A4 | 时序无前视 | 抽 T 日一行核查各列时点 | 分钟列 ≤ 9:40、`$price_941` = 9:41、日频列 lag1、`$change_941` = 9:41 时刻 |
| A5 | 混频拼表 | HighBetaAlpha158 fetch shape + 列名 | 158 + 14 = 172 feature + 1 label；NaN 率抽样合理 |

### B. 信号有效性

| # | 验什么 | 怎么验 | 通过标准 |
|---|---|---|---|
| B1 | 回测链路跑通 | `qrun workflow.yaml` 全量（2024-2026） | pred/label/IC/nav 全产出；quote_df 含 `$price_941`/`$change_941` 列无异常 NaN |
| B2 | 撮合生效（v2 强化）| 看回测成交记录 + limit_buy | (a) **shift=0**：买单 `trade_date` == `pred` index 当天（非次日）；(b) 买入价 == T 日 `$price_941`；(c) 9:41 已封涨停（`$change_941 >= $limit_up`）的票 `limit_buy=True`、买单被拦 |
| B3 | baseline IC 对比（核心判据）| 纯日频 Alpha158 baseline（IC=-0.0078）vs 加 14 分钟因子版 | IC / Rank IC / ICIR 三项提升 |
| B4 | 特征重要性 | LGBModel `feature_importance` top-30 | 14 个分钟因子里至少几个进 top-30 |

### 验证顺序

1. 物化脚本单测（A1-A3 + A1b）—— 不进 qlib，纯数据层核验
2. 烟雾测试 2025 子窗口（train 2025-01~09 / test 2025-10~12）跑通全链路（A4-A5 + B1-B2）
3. 全量回测 2024-2026（B3-B4）

### 关键判据

- **B2(a) shift=0 是 v2 命门**：必须确认成交日 = 信号日（而非次日），否则 shift=0 子类失效、label 错位。
- **B3 是信号命门**：分钟因子加进去 IC 必须比 baseline 提升，否则设计失败、回炉。
- **B4 定生死**：如果 14 个因子都没进 top-30，说明这套因子设计没抓到信号，要重想公式（不是调参）。

---

## 已知妥协 / 不在本 spec 范围

- **HFLGBModel 不切换**：label 保持回归（连续收益率），与 baseline IC 可直接对比；HFLGBModel 二分类方向预测留作后续可选。
- **卖出价固定 T+1 收盘**：未做卖出价优化（T+1 开盘 / T+1 9:41 对称等候选见 brainstorming 记录）。
- **ST 涨跌停未区分**：仍按板块分级（继承现有 MVP 妥协）。
- **跌停买入不拦**：撮合可行性上跌停买得进，本设计只拦涨停；跌停规避留作策略层后续。
- **1min 数据只取 9:30-9:41**：其余 230 槽未利用（高频因子扩展留后续）。

---

## qlib 原生依据（CLAUDE.md "不偏离原生扩展" 凭证）

- `Alpha158` 子类化（覆写 `get_feature_config`，field 包 `Ref(*,1)`）—— 原生扩展点，`Ref` 嵌套合法（test_ref.py 验证）。
- **`TopkDropoutStrategyTD0`（shift=0 子类）** —— v2：qlib 无 shift 参数/hook，整段重写 `generate_trade_decision` 改一行 `shift=0` 是标准扩展模式（[signal_strategy.py:142](file:///home/zxh/miniconda3/envs/qlib_ifind_beta/lib/python3.12/site-packages/qlib/contrib/strategy/signal_strategy.py#L142) 硬编码 `shift=1`，源码确认）。
- `Exchange(deal_price=List[str])` —— 买卖不同价原生支持（[exchange.py:61-65](file:///home/zxh/miniconda3/envs/qlib_ifind_beta/lib/python3.12/site-packages/qlib/backtest/exchange.py#L61)）。
- `Exchange(limit_threshold=Tuple[str,str])` 即 `LT_TP_EXP` —— buy/sell 表达式分别求值（[exchange.py:285-286](file:///home/zxh/miniconda3/envs/qlib_ifind_beta/lib/python3.12/site-packages/qlib/backtest/exchange.py#L285)），tuple 表达式字段经 `D.features` 自动订阅。
- 物化 day.bin —— `FileFeatureStorage` 缺 bin 鲁棒（返回空 Series 不崩）。
- `NestedDataLoader` / `QlibDataLoader(freq=dict)` —— context7 查证为 qlib 原生混频入口，但仅混"频级行"，本设计不走此路（聚合放数据层）。
