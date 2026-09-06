---
nav_exclude: true
---

# T-1/T-2 开盘因子族设计 — 破 IC→超额墙的「同 regime 滞后」假设

> 日期：2026-07-08（overnight autonomous，用户已授权「仔细挖掘…明早告我结论…需要决策的记录下来」）
> 线索：goal「换个思路可以扩展不同区间的 1min 因子特征，提高 IC 和超额收益」
> 前置：§23/§24/§25 三重墙定论（champion enhanced(18)@n_drop=15 的 18 因子近 topk=20 边界最优）

---

## 1. 问题与假设

**三重墙复盘**（机制各异、全败）：

| 实验 | 改动 | 信号 IC | 超额含成本年化 | 失败机制 |
|---|---|---|---|---|
| §23 Step B | +12 日频/指数→30 | 0.0611（+12%）↑ | +159.27% ≈ 平 | IC↑ 不传导头部 |
| §24 CSRank | 18 横截面归一化 | 0.0526 ↓ | +143.14% ↓ | 边界摧毁（平坦/倒挂） |
| §25 tail | +2 正交尾盘→20 | 0.0457 ↓ | +139.60% ↓ | 边界腰斩 + 尖峰压平 |

共同结构结论：champion 超额来自 topk=20 边界放得极锐（边界内缘−外缘 Δ=1.03%/日），18 因子已在边界近最优——**任何方向都重排并退化该边界**。

**§25 失败的关键洞察**：T-1 尾盘（slot 232-241）是**不同 regime**（尾盘 = 收盘前主力最终态度），与 T 日早盘开盘动量**正交**（|corr|<0.07）→ 正交因子重排 topk=20 边界、把 champion 头部赢家挤出 → 典型「加因子稀释」。

**新假设（本设计核心）**：T-1 **开盘**（slot 1-10，与 T 日开盘**同 regime**、仅时间平移 1 天）。若高贝塔开盘动量**跨日持续**（真启动/共振/惯性），则 T-1 开盘与 T 日开盘**共线强化**（reinforcing，非正交重排）→ 可能**不重排边界而是锐化它**，从而破墙。若跨日**反转**，则 T-1 开盘正交 → 大概率复刻 §25 失败。

→ **Exp1 = T-1 开盘是「高贝塔开盘动量是否跨日持续」的科学检验**（无论胜败都有结论）。

## 2. 用户领域心法对齐

用户 /goal（prior，preserved）：「分钟频只能观测 T 日的开盘阶段，所以我需要结合指数（大势）和 **T-1 之前的 K 线**来判断个股是否是**共振、启动、高潮或者惯性冲高**」。

→ regime 检测框架：用 T-1/T-2 开盘动量判断 T 日开盘是真启动/共振/惯性（可买）还是高潮/衰竭（回避）。树模型（LGBM）给齐 T 日 + T-1 开盘因子后，自动通过 splits 学**交互**（T-day × T-1 开盘 = 共振强度）。故「加 T-1 开盘因子」= 给模型 regime 检测原料，方向与用户心法一致。

## 3. 因子族（一次物化，IC 校准后挑组合）

物化 2 族（复用 champion 早盘 5 公式 + shift）：

**Fam A — T-1 开盘**（slot 1-10 shift1，5 bin/股）：
- `startup_mom_5m_t1` = shift1(c[9]/c[4]-1)
- `startup_total_t1` = shift1(c[9]/o[0]-1)
- `accel_5m_t1` = shift1((c[9]/o[5]-1)-(c[3]/o[0]-1))
- `close_pos_5m_t1` = shift1((c[9]-l5)/(h5-l5))
- `vol_ratio_5m_t1` = shift1(v[5:10].mean/v[0:4].mean)

**Fam B — T-2 开盘**（slot 1-10 shift2，5 bin/股）：同 5 公式 shift2，测动量持续 2 日 vs 1 日（更深 regime）。

**Fam D — 持续比/交互**（Handler 层 qlib 表达式，**不新增 bin**，复用 Fam A + champion bin）：
- `opening_persistence` = `$startup_total / $startup_total_t1`（T 日开盘涨幅 / T-1 开盘涨幅 = 共振强度）
- 等 IC 校准后按需加。

## 4. 前视论证（CLAUDE.md 硬约束 + label 冻结）

T-1/T-2 开盘 shift 在 **min-cal 空间**：min-cal 日 d 的开盘 → shift k → 存进 min-cal 日 d 的 fac → scatter 到 day-cal 行 day_rows[d]。等价：day-cal 行 T 存 min-cal 日 d-k 的开盘 = **T-k 的 9:31-9:40**。

时序：T-1 9:40 数据在 T-1 9:40 产生 ≪ T 日 9:41 决策。**无前视** ✓。与 §25 尾盘（T-1 15:00 → T 9:41）前视论证完全同构，由 test_materialize_minute 的 shift 判别测试覆盖模式。

**全冻结**：label（`Ref($close,-1)/$price_941-1`）、deal_price（`$price_941/$close`）、涨跌停拦截、策略（TopkDropoutStrategyTD0 n_drop=15 topk=20）、benchmark（SH000300）、切分、模型超参。唯一变量 = feature 集。

## 5. 实验矩阵与决策判据

baseline = champion IC 0.0545 / 超额含成本年化 +158.86% / 回撤 −5.44% / IR≈4.6 / 边界锐度 Δ≈1.03。

| 变体 | feature | 假设 |
|---|---|---|
| **V1** | 18 + top-2 T-1 开盘 = 20 | §25 同构头对头（同 count、同 shift、仅 regime 不同）→ 隔离「同 regime 滞后 vs 不同 regime」 |
| **V2** | 18 + top-2 T-1 开盘 + 1 持续比 = 21 | regime 交互（用户心法） |
| **V3**（若时间） | 18 + top-2 T-2 开盘 = 20 | 更深 lag 持续性 |
| **V4**（若 V1 墙） | 16（丢 2 弱 champion）+ top-2 T-1 开盘 = 18 | replacement 非 add（测「不同 18 是否更优」） |

**判据**：
- **promote**：超额 > champion+10pp（≥+169%）**且** IC > champion **且** 回撤不恶化 >2pp。三者全满足。
- **marginal**：超额 ±10pp 内 → 不 promote，记「同墙内扰动」。
- **reject**：超额 < champion−10pp → 第 4 次墙确认。
- 边界锐度 Δ 作机制诊断（champion ≈1.03；§25 腰斩至 0.50）。

## 6. 实现（仅 ADD，champion 14 baseline 冻结）

- `config.py`：加 `MINUTE_FACTOR_OPENING_T1_FIELDS` / `MINUTE_FACTOR_OPENING_T2_FIELDS`（各 5）。
- `materialize_minute.py`：复用 champion 早盘 5 公式（已在 fac），copy 到 _t1/_t2 key → 进 shift 循环（TAIL+OPENING_T1 shift1，OPENING_T2 shift2）→ 加 `_scatter_fields`。**不改既有公式行**。
- `tests/test_materialize_minute.py`：加 `test_opening_t1_t2_factors_crosscheck`（shift 判别 + 公式逐字对齐）+ 更新 bin 计数 25→35。
- handler 变体：`MinuteEnhancedOpeningT1Handler(MinuteEnhancedHandler)`，`OPENING_T1_PICK = (top-2 by IC)`，仅 override get_feature_config（mirror MinuteEnhancedTailHandler）。
- workflow：`workflow_minute_enhanced_opening_t1.yaml`（copy tail workflow，换 handler/recorder）。

## 7. 明早需用户决策的（记录）

1. 若 V1 marginal/reject：接受「墙」结论转向 OOS walk-forward（sliding 设计已批、paused），还是继续挖第 N+1 族？
2. 若 V1 promote 候选但回撤恶化：接受回撤销换 IC/超额？（回撤门槛待定）
3. V4 replacement 若胜：是否正式替换 champion 因子集（影响 §D6 冻结口径）？
4. 若因子族全败：是否考虑调 topk（§24 hint：CSRank 在 topk=10 可能赢）？策略层始终冻结，只记录不擅动。
