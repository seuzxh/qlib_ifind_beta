# 883926 高贝塔策略 · 分钟因子设计

> **状态**：定稿待审
> **日期**：2026-07-06
> **范围**：在现有日频 Alpha158 baseline 之上，叠加 T 日 9:30~9:40 分钟频因子，与日频因子共同训练；信号 T 日 9:41 撮合买入。

---

## TL;DR

- 在 Alpha158（158 日频特征）之上新增 **14 个分钟频衍生因子**（4 维度 × 3 周期 + 整段基准 + 跨日量能），围绕高贝塔成分股"开盘启动"特点设计。
- 分钟因子由独立物化脚本从 `cn_data_1min` 计算，落成 day.bin 进 overlay；qlib 仍只消费日频 overlay，**不挂双频**。
- Handler 子类 `HighBetaAlpha158` 把 14 个新字段追加进 feature 集。
- **买入价改为 T 日 9:41 close**（`$price_941`），卖出 T+1 收盘；label = `Ref($close,-1)/$price_941 - 1`。
- 全程无前视：特征 ≤ T 日 9:40，撮合价 T 日 9:41，日频特征 lag1（T-1）。

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
| 周期 | 1min / 3min / 5min 滑动窗口 | 落在 9:30~9:40 共 11 根 bar 上 |
| 实现方案 | A：物化 day.bin + 子类 Alpha158 | 不用 NestedDataLoader（详见 §物化架构） |
| 模型 | LGBModel（不变） | HFLGBModel 非多频 model，留作后续可选 |
| 买入价 | T 日 9:41 close（`$price_941`） | 物化为 day.bin |
| 卖出价 | T+1 收盘（`$close`） | A 股 T+1 最早 T+1 卖 |
| label | `Ref($close,-1)/$price_941 - 1` | T 日 9:41 买 → T+1 收盘卖 |
| deal_price | `["$price_941","$close"]` | buy/sell 分开，qlib 原生支持 |
| 切分 | 不变：train 2024-01-01→2025-12-31 / valid 2026-Q1 / test 2026-04-01→2026-07-02 | |

---

## 数据基础

### cn_data_1min 结构（第一性原理 probe 验证）

- 路径：`/home/zxh/cn_data_1min`
- 1min 日历：`calendars/1min.txt`，1,553,640 行 = 6420 日 × **242 槽**，**标记 start-time**（`YYYY-MM-DD HH:MM:00`）：slot 0 = 09:30、slot 11 = 09:41、slot 241 = 15:00（probe 实证 2026-07-06）。
- 字段：7 个 `.1min.bin`（close/open/high/low/volume/factor/vwap），与日频一致。
- **日内槽位**（关键，start-time 标记下）：
  - slot 0~10 = 9:30~9:40（11 根 bar，**因子输入**）
  - **slot 11 = 9:41**（**买入价 bar，不参与因子计算**）
- **数据集头洞（实证，影响物化）**：全池所有票的 1min.bin `start_index` 均为 1407473 = 2024-01-02 09:31（slot 1），`arr.size % 242 = 241` —— 即 **2024-01-02 的 9:30 那根 bar 全池普遍缺失**，其余每日完整 242 根。物化必须按**日历网格对齐**（见 §物化架构）：按绝对日历行散射、不可 `reshape(242)`（会逐日偏移一格）。slot 0 缺失处写 NaN，引用 slot 0 的因子（`startup_total / accel_* / vol_ratio_*` 等）当日为 NaN → DropnaLabel 处理（仅 1 个训练日，可忽略）。
- day 日历：`calendars/day.txt`（与 qlib_data 一致）。
- instruments：`all.txt`（与 qlib_data 一致）。

### 池覆盖率

highbeta883926 时变池 5116 只 unique codes，在 cn_data_1min 覆盖 5040 只（98.5%），76 只缺失（与日频 bin 缺失集一致，已退市/停牌早）。缺失票物化时写 NaN，qlib DropnaLabel 自动处理。

### 符号约定

记 T 日分钟 bar：
- `c_i / o_i / vol_i / h_i / l_i` = index i 的分钟 close/open/volume/high/low（i = 0..11）
- index 0..10 = 9:30..9:40（因子输入），index 11 = 9:41（买入价）
- `mean(vol_a..b)` = index a 到 b 的 volume 均值；`max(h_a..b) / min(l_a..b)` 同理

---

## 因子集（14 个）

### A. 启动动量（一阶累计收益）—— 开盘后涨了多少

| 因子 | 公式 | 语义 |
|---|---|---|
| `startup_mom_1m` | `c10/c9 - 1` | 最后 1 分钟涨幅（尾动量） |
| `startup_mom_3m` | `c10/c7 - 1` | 最后 3 分钟涨幅 |
| `startup_mom_5m` | `c10/c5 - 1` | 最后 5 分钟涨幅 |
| `startup_total` | `c10/o0 - 1` | 整段基准：9:30 开盘 → 9:40 累计涨幅 |

`startup_total` 区分"冲高回落"（total 正、mom 负）与"尾盘发力"（两者都正）两种形态。

### B. 加速度（二阶，避追高核心）—— 后段收益 − 前段收益

段收益用「段首 open → 段末 close」：

| 因子 | 前段 | 后段 | 公式 |
|---|---|---|---|
| `accel_1m` | index0 | index10 | `(c10/o10-1) - (c0/o0-1)` |
| `accel_3m` | index0-2 | index8-10 | `(c10/o8-1) - (c2/o0-1)` |
| `accel_5m` | index0-4 | index6-10 | `(c10/o6-1) - (c4/o0-1)` |

`>0` → 加速赶顶（追高风险）；`<0` → 减速（健康）。特征不预判方向，交给模型学。

### C. 拉升形态（收盘在窗口高位还是低位）

`close_pos_W = (c10 - low_W) / (high_W - low_W)`，越接近 1 越收在最高位：

| 因子 | 窗口 |
|---|---|
| `close_pos_1m` | index10 单根的 h/l |
| `close_pos_3m` | index8-10 的 max(h)/min(l) |
| `close_pos_5m` | index6-10 的 max(h)/min(l) |

### D. 量能放量（后段均量 / 前段均量）

| 因子 | 公式 |
|---|---|
| `vol_ratio_1m` | `vol10 / mean(vol0-9)` |
| `vol_ratio_3m` | `mean(vol8-10) / mean(vol0-2)` |
| `vol_ratio_5m` | `mean(vol6-10) / mean(vol0-4)` |

### E. 跨日量能（绝对放量）

`vol_vs_yest = sum(vol0-10) / (Ref($volume,1)/240)`

前 10 分钟总量 ÷ 昨日全日量折算到每分钟。分子 T 日分钟、分母 `Ref($volume,1)` T-1 日频，**无前视**。

---

## 物化架构

### 方案 A（采用）：独立读 1min.bin，qlib 不挂双频

物化脚本（`materialize_minute.py`）独立用 numpy 解析 cn_data_1min 的 1min.bin（probe 已验证读法：bin 头部 float32 `start_index` + float32 数组），按 **1min 日历的 slot 网格对齐**取数：预计算全局 `min_slots[cal_row]`（0..241），把每只票的 bin 按 `start_index` 散射到全局 `(day, slot)` 网格，取 slot 0-10 / slot 11（9:30-9:41）算 14 因子 + 9:41 close → 写 day.bin 进 overlay。**为什么不 `reshape(242)`**：全池 bin 普遍从 slot 1 起步（2024-01-02 9:30 头洞，见 §数据基础），`arr.reshape(n,242)` 会逐日整体偏移一格、首日错位；日历网格散射按绝对 cal_row 取值，对任意 bin 头洞/缺格鲁棒（缺处 NaN）。

qlib 保持 `provider_uri=data/qlib_root, freq=day`（不变），完全不知道分钟数据存在。新增 **15 个 day.bin/票**（14 因子 + `$price_941`）与现有 `change/limit_up/limit_down` 同模式。

**`vol_vs_yest` 特殊处理**：分子 `sum(vol0-10)` 来自分钟 bin，分母 `Ref($volume,1)` 来自日频 bin（qlib_data）。物化脚本同时读 cn_data_1min（分钟）+ qlib_data（日频）两个源算好该因子，落 day.bin。

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
| T-1 收盘及更早 | 日频特征就绪 | qlib_data（`Ref(*,1)` lag1） |
| T 日 9:30~9:40 | 分钟特征就绪 | cn_data_1min index 0-10 |
| T 日 9:40 | 信号产出（model.predict） | 特征到齐 |
| **T 日 9:41** | **撮合买入** | `$price_941[T]`（index 11 close） |
| T+1 收盘 | 撮合卖出 | `$close[T+1]` |

**无前视核查**：
- 分钟特征最晚 index 10（9:40）< 撮合价 index 11（9:41）✓
- 日频特征 lag1（T-1）< T 日 9:41 ✓
- label 用 T+1 数据，但 label 只作训练 target，**不进特征** ✓
- Exchange 不校验撮合时刻 ≥ 信号时刻，只读 `$price_941[T]` 字段值；数值上 9:41 ≥ 特征 9:40 即无前视 ✓

**语义偏离（可接受）**：传统 daily 回测 T 日开盘撮合、隐含 T-1 信号；本设计 T 日 9:41 撮合、T 日 9:40 信号。无前视是充要条件，偏离不影响回测有效性。

---

## Handler 子类

```python
class HighBetaAlpha158(Alpha158):
    def get_feature_config(self):
        fields, names = super().get_feature_config()  # 原生 158
        min_fields = [
            "$startup_mom_1m", "$startup_mom_3m", "$startup_mom_5m", "$startup_total",
            "$accel_1m", "$accel_3m", "$accel_5m",
            "$close_pos_1m", "$close_pos_3m", "$close_pos_5m",
            "$vol_ratio_1m", "$vol_ratio_3m", "$vol_ratio_5m",
            "$vol_vs_yest",
        ]
        return fields + min_fields, names + min_fields
```

label 经 Alpha158 的 `label` kwarg 传入（YAML 配置，见下）。

---

## Label + deal_price

- **label**：`Ref($close, -1) / $price_941 - 1`（T 日 9:41 买 → T+1 收盘卖）
- **deal_price**：`["$price_941", "$close"]`（buy/sell 分开，[exchange.py:157-164](file:///home/zxh/miniconda3/envs/qlib_ifind_beta/lib/python3.12/site-packages/qlib/backtest/exchange.py#L157) 原生支持）

---

## YAML 改动点（相对当前 workflow.yaml）

| 段 | 改动 |
|---|---|
| `qlib_init` | 不变（`provider_uri: data/qlib_root`） |
| `handler.class` | `Alpha158` → `HighBetaAlpha158`（需 import） |
| `handler.kwargs.label` | `Ref($close,-2)/Ref($open,-1)-1` → `Ref($close,-1)/$price_941-1` |
| `exchange.deal_price` | `["$open","$close"]` → `["$price_941","$close"]` |
| 其余（model/strategy/executor/record/segments） | 不变 |

---

## 验证计划

### A. 工程正确性

| # | 验什么 | 怎么验 | 通过标准 |
|---|---|---|---|
| A1 | 物化正确性 | 抽 3 票（主板/创业板/科创板各一）× 3 日，读原 1min.bin 手算 6 个代表因子（`startup_mom_1m / accel_3m / close_pos_5m / vol_ratio_3m / vol_vs_yest / price_941`），对比 day.bin 值 | 误差 < 1e-5 |
| A2 | 缺失容错 | 76 只缺 bin 票 + 停牌段跑物化脚本 | 不崩；缺失时段写 NaN（DropnaLabel 处理） |
| A3 | 日历对齐 | 1min.txt 日期 vs day.txt 日期 | 交易日集合一致（1min 是 day 的细化） |
| A4 | 时序无前视 | 抽 T 日一行核查各列时点 | 分钟列 ≤ 9:40、`$price_941` = 9:41、日频列 lag1 |
| A5 | 混频拼表 | HighBetaAlpha158 fetch shape + 列名 | 158 + 14 = 172 feature + 1 label；NaN 率抽样合理 |

### B. 信号有效性

| # | 验什么 | 怎么验 | 通过标准 |
|---|---|---|---|
| B1 | 回测链路跑通 | `qrun workflow.yaml` 全量（2024-2026） | pred/label/IC/nav 全产出；quote_df 含 `$price_941` 列无 NaN |
| B2 | deal_price 撮合生效 | 看回测成交记录买入价 | 买入价 == T 日 9:41 close（≈ 开盘附近，非 T 日 open/close） |
| B3 | baseline IC 对比（核心判据） | 纯日频 Alpha158 baseline（IC=-0.0078）vs 加 14 分钟因子版 | IC / Rank IC / ICIR 三项提升 |
| B4 | 特征重要性 | LGBModel `feature_importance` top-30 | 14 个分钟因子里至少几个进 top-30 |

### 验证顺序

1. 物化脚本单测（A1-A3）—— 不进 qlib，纯数据层核验
2. 烟雾测试 2025 子窗口（train 2025-01~09 / test 2025-10~12）跑通全链路（A4-A5 + B1-B2）
3. 全量回测 2024-2026（B3-B4）

### 关键判据

- **B3 是命门**：分钟因子加进去 IC 必须比 baseline 提升，否则设计失败、回炉。当前 baseline IC≈0 是因为"Alpha158 默认周期与日频 label + 高贝塔池不匹配"——分钟因子正是为高贝塔启动特点设计，理应改善。
- **B4 定生死**：如果 14 个因子都没进 top-30，说明这套因子设计没抓到信号，要重想公式（不是调参）。

---

## 已知妥协 / 不在本 spec 范围

- **HFLGBModel 不切换**：label 保持回归（连续收益率），与 baseline IC 可直接对比；HFLGBModel 二分类方向预测留作后续可选。
- **卖出价固定 T+1 收盘**：未做卖出价优化（T+1 开盘 / T+1 9:41 对称等候选见 brainstorming 记录）。
- **ST 涨跌停未区分**：仍按板块分级（继承现有 MVP 妥协）。
- **1min 数据只取 9:30-9:41**：其余 230 槽未利用（高频因子扩展留后续）。

---

## qlib 原生依据（CLAUDE.md "不偏离原生扩展" 凭证）

- `Alpha158` 子类化（覆写 `get_feature_config`）—— 原生扩展点，追加字段零风险。
- `Exchange(deal_price=List[str])` —— 买卖不同价原生支持（[exchange.py:44/157-164/178/201-213](file:///home/zxh/miniconda3/envs/qlib_ifind_beta/lib/python3.12/site-packages/qlib/backtest/exchange.py#L44)，已源码确认 `$price_941` 这种 materialized field 可作 deal_price）。
- `Ref(expr, 1)` 嵌套 —— test_ref.py 验证合法且等价直接 base lag。
- `TopkDropoutStrategy(hold_thresh=1)` —— daily 模式原生 T+1。
- 物化 day.bin —— `FileFeatureStorage` 缺 bin 鲁棒（返回空 Series 不崩）。
- `NestedDataLoader` / `QlibDataLoader(freq=dict)` —— context7 查证为 qlib 原生混频入口，但仅混"频级行"，本设计不走此路（聚合放数据层）。
