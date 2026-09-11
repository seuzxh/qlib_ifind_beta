---
layout: default
title: overnight_gap 口径 A/B 与数据源 factor 逐日漂移发现
parent: "验证与研究"
nav_order: 4
---

# overnight_gap 口径 A/B 与数据源 factor 逐日漂移发现

> 日期：2026-09-11  
> 状态：**研究结论已存档，Champion 不变**；衍生出一个待立项问题（label 基准噪声）  
> 脚本：`scripts/validate_gap_adjusted.py`；变体 recorder `b7db9b9b`（实验
> `gap_adjusted_variant`）；基线 recorder `f364dcb3`（与冻结 Champion 零漂移）

## 1. 背景

overnight_gap 现役为"名义口径"：`(open[T]/f[T]) / (close[T-1]/f[T-1]) - 1`（每天
各除各的复权因子）。质疑：除权配股的机械跳空不应进入因子，标准做法应当用后复权
`open[T]/close[T-1]-1` 消除。本实验做单变量对照：唯一变量是 gap 口径，其余
17 因子、label、HFLGB 超参、切分、TD0 策略、成本全部与
`workflow_minute_enhanced_tk10_nd8.yaml` 逐字段相同。

## 2. 发现：数据源 factor 序列逐日漂移（非阶梯函数）

正常复权因子应只在除权除息日跳变。实测本数据源（`cn_data` 日频 factor bin）：

- test 段 5,870 / 6,187 个样本 `f[T] ≠ f[T-1]`，日漂移幅度中位 0.07%、p99 2.45%；
- **铁证（一字板日）**：某日 `open == prev_close`（一字涨跌停），后复权口径 gap
  精确等于 0，名义口径却显示 +0.12%——恰好等于当日因子漂移量。

推论：**本数据源里 `close/factor` 不是交易所名义价**（带逐日基准噪声），后复权
序列才是内部自洽的"真价"序列。物化 bin 忠实实现了名义公式（逐样本与手工重算
一致），问题在数据源的 factor 定义，不在物化实现。

## 3. A/B 结果（test 2026-04→07，含成本）

| 指标 | Champion（名义 gap） | Variant（后复权 gap） |
|---|---|---|
| IC / Rank IC | **0.0548 / 0.0612** | 0.0194 / 0.0323 |
| ICIR | 0.506 | 0.188 |
| 策略累计 | +73.30% | **+74.28%** |
| 超额累计 | +56.05% | **+56.75%** |
| IR（日频年化） | 5.00 | 4.93 |
| 最大回撤 | -13.47% | **-11.16%** |
| 与 Champion Top10 重合 | — | 0/62 天 |

## 4. 取证：Champion 名义口径的 IC 优势 = 与 label 共享的基准噪声

单因子日均 IC（全窗口 2024-01→2026-07）：

| gap 口径 | 现役 label（日频 adj close ÷ 1min adj price_941，混合基准） | 干净 label（纯日频 adj 序列 close-to-close） |
|---|---|---|
| 名义 | **+0.0701** | +0.1273 |
| 后复权 | +0.0557 | **+0.1301** |

结论：**对干净 label 两种口径信息量几乎相同（后复权还略优）**。名义 gap 对现役
label 的领先来自共享的因子漂移噪声（label 混用日频与 1min 两套复权基准）。换
干净 gap 后模型无法利用这份噪声 → 测得 IC 崩塌；但噪声本不可交易 → 组合收益
持平、回撤略改善。

## 5. 三个结论与一个待立项

1. **不切换**：后复权口径无收益损失、信息量持平，但生产 09:40 拿不到当日
   factor，名义口径仍是研究/生产两路径唯一可逐位复算的选择——Champion 冻结合同
   不变，切换收益为零而验证成本高。
2. **回测稳健性警示**：IC 从 0.055 崩到 0.019、Top10 每天 0 重合，组合收益却
   持平——本窗口收益主要由池 beta + 其余 17 因子驱动，对头部排序质量不敏感。
   评估任何因子/模型改动时不能只看回测收益，必须看 IC 与 Top10 稳定性。
3. **文档纠错**：此前"复权会抵消真实跳空"的表述不准确（对本数据源，后复权才
   是自洽序列）。factors.md、pipeline-walkthrough.md、materialize_minute.py 注释
   已随本存档同步修正。
4. **待立项（更大发现）**：现役 label `Ref($close,-1)/$price_941-1` 混用日频与
   1min 两套复权基准，在 factor 逐日漂移的数据源下 label 本身可能带 ±0.1%~0.3%/日
   的基准噪声，影响全部因子的 IC 测量与训练目标。后续可用同源一致 label
   （如 1min close 序列自建）重训对照验证。

## 6. 产物

- 脚本：`scripts/validate_gap_adjusted.py`（含单因子 IC、全流程训练、对照报告）
- 实验 `gap_adjusted_variant` / recorder `b7db9b9b9fca4b71b6becff474121b58`
- 基线：`minute_enhanced_tk10_nd8` / recorder `f364dcb3`（2026-09-06 重跑，
  与冻结 Champion pred 逐位相同）
