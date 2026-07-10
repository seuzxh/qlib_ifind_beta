# 日频情绪因子 + 上证指数共振因子 设计（2026-07-07）

> 状态：**已实现 + 已证伪（§39，2026-07-09）**。原设计承接旧 champion=enhanced(18)@n_drop=15；当前 champion 已 §33 晋升 tk10/nd8。index/日频因子在当前 champion 上的权威 A/B（shadow + W2 OOS）= W1 正向（IC +12%）→ **W2 决定性证伪（IC −15% / 超额 −51pp）**，§27 窗口过拟合。详见 backtest-log §39。本文保留作设计记录。
> 遵循 superpowers brainstorming；本文为 spec，确认后走 writing-plans → TDD 实现。

## 1. 目标（用户原话）
1. 添加日频因子，分析个股情绪阶段（启动/发酵/高潮），避免拥挤度过高接盘，辅助分钟因子加强决策。
2. 添加上证指数相关因子，辅助判断个股&指数共振，根据上证指数情绪（冰点/沸点）辅助判断高 beta 类个股风险。

## 2. 技术前提确认（第一性原理 + 实测，不联想）

### 2.1 数据可得性
- ✅ SH000001（上证综指）在 `/home/zxh/qlib_data/features/sh000001/` 有完整 8 字段（open/high/low/close/volume/vwap/amount/factor）。overlay 现无 → 需补 symlink。
- ✅ 个股原生 7 字段（open/high/low/close/volume/factor/vwap），**无 amount/turn** → 拥挤度只能用 volume 量比代理（champion 已有分钟版 vol_ratio/vol_vs_yest，日频版新建）。

### 2.2 跨 instrument 引用（任务3 核心）— qlib 原生 ChangeInstrument
- `qlib.data.ops.ChangeInstrument`（ops.py:64，pyqlib 0.9.7 **实测可用**）。
- 实测 `D.features(['sh600519'], ["ChangeInstrument('SH000300', $close)"])` 返回 SH000300 真实指数值（3386.35）；复合表达式（指数日收益、个股超额）正确解析求值。
- 语义 `ChangeInstrument(instrument, feature)`，instrument 第一参数；load 忽略传入 instrument、用指数。
- 零自定义抽象层，符合 CLAUDE.md「不偏离原生扩展」。

### 2.3 前视对齐硬约束（最易出错，定死）
- champion 撮合 = T 日 **9:41**（TopkDropoutStrategyTD0 shift=0）。T 日 9:41 决策时 T 日全天日频数据未产生。
- → 所有日频/指数因子必须基于「截至 T-1 收盘」数据 = 表达式套 `Ref(expr, 1)`：T 行的 Ref(expr,1) = expr 在 T-1 的值，T 日 9:41 已知。
- 与 champion 分钟因子区别：分钟因子 T 日 9:30-9:40 物化、当天已知、**不 lag**；日频/指数因子 **lag1**。两类在 9:41 决策点都已知，无前视。

## 3. 任务2：日频情绪阶段因子

### 3.1 第一性原理：三阶段可观测特征
- **启动**：动量由负转正、量能温和放大、价格近均线（乖离小）
- **发酵**：连涨、动量加速、量价齐升、乖离扩大未极端
- **高潮（拥挤）**：严重超买（高乖离）、极度放量（高量比）、近涨停、RSV 极高

### 3.2 设计原则
提供**连续可观测因子**让 LGBM 自己学阶段非线性（模型吃连续特征 > 规则硬分类；符合「让模型学，不预设」+ qlib 原生特征层）。

### 3.3 候选因子（全部 Ref(...,1) 防前视）
| 因子 | 表达式（T-1 口径） | 语义 |
|---|---|---|
| bias_5 | `Ref(($close-Mean($close,5))/Mean($close,5),1)` | 5 日乖离，超买 |
| bias_20 | `Ref(($close-Mean($close,20))/Mean($close,20),1)` | 20 日乖离 |
| vol_ratio_20 | `Ref($volume/Mean($volume,20),1)` | 量比，拥挤度代理 |
| run_up_5 | `Ref($close/Ref($close,5)-1,1)` | 5 日动量累积 |
| rsv_9 | `Ref(($close-Min($low,9))/(Max($high,9)-Min($low,9)),1)` | 经典超买超卖 |
| dist_to_limit | `Ref($change/$limit_up,1)` | 封板强度（高潮极致） |
| accel_mom | `Ref(($close/Ref($close,3)-1)-(Ref($close,3)/Ref($close,6)-1),1)` | 动量加速度 |

7 个连续因子覆盖启动→发酵→高潮。

## 4. 任务3：上证指数共振 + 冰点沸点因子（ChangeInstrument）

设 `idx_close = ChangeInstrument('SH000001', $close)`（同理 idx_high/low/volume）。

### 4.1 个股&指数共振（系统性暴露）
| 因子 | 表达式 | 语义 |
|---|---|---|
| beta_20 | `Ref(Cov(stock_ret,idx_ret,20)/Var(idx_ret,20),1)` | 个股对指数 beta |
| corr_20 | `Ref(Corr(stock_ret,idx_ret,20),1)` | 同向程度 |

`stock_ret = $close/Ref($close,1)-1`；`idx_ret = idx_close/Ref(idx_close,1)-1`。

### 4.2 指数情绪（冰点=超卖恐慌 / 沸点=超买狂热）
| 因子 | 表达式 | 语义 |
|---|---|---|
| idx_bias_20 | `Ref((idx_close-Mean(idx_close,20))/Mean(idx_close,20),1)` | 沸点高/冰点低 |
| idx_rsv_9 | `Ref((idx_close-Min(idx_low,9))/(Max(idx_high,9)-Min(idx_low,9)),1)` | 经典冰点沸点 |
| idx_run_5 | `Ref(idx_close/Ref(idx_close,5)-1,1)` | 短期过热/过冷 |
| idx_vol_ratio_20 | `Ref(idx_volume/Mean(idx_volume,20),1)` | 情绪激烈度 |

「高 beta 股在沸点风险高」：不硬编码交互，分别给共振+情绪因子，让模型学交互（更原生）。

共 6 个指数因子。任务2+3 合计 13 个。

## 5. 架构（qlib 分层对齐 + 最小改动）

### 5.1 expression 不物化
13 个因子全是日频滚动统计 + ChangeInstrument，qlib expression 直接算（Cov/Var/Corr/Mean/Ref/Max/Min 原生）。→ 不写 materialize 脚本、不新增 bin（除 SH000001 symlink）。最 native。

### 5.2 数据层
build_overlay.py 加 SH000001 symlink（7 字段复用 qlib_data，同 sh000300 模式）。指数 factor=1、不交易、无需 change/limit 衍生。

### 5.3 handler（见 Q1 待定）
- **选项 A（合体）**：`EnhancedWithDailyIndex(MinuteEnhancedHandler)`，get_feature_config = 18 分钟($field 不lag) + 13 日频/指数(expression Ref1) = 31 因子。保留 champion 全部分钟 alpha + 新维度。
- **选项 B（单独）**：`IndexDailyHandler` 纯 13 因子，先验证日频/指数因子自身有效性，再决定合体。

### 5.4 workflow
`workflow_daily_index.yaml`（复制 workflow_minute_enhanced.yaml，换 handler）。label=`Ref($close,-1)/$price_941-1`、deal_price=`["$price_941","$close"]`、TopkDropoutStrategyTD0、n_drop=15 **全冻结不动**。

## 6. 实验设计 + Gate（不污染 champion）
1. valid(2026-01→03) OOS：IC/ICIR vs champion(0.0545/5.12)
2. test(2026-04→07) 回测：超额/收益/回撤 vs champion(+158.86%/-5.44%)
3. valid+test 双优才升级；否则保留 champion、记录负结果（full(85) 教训）

### 6.1 风险 flag
- full(85) 前车：日频可能稀释 9:41 分钟信号。mitigation = 情绪/共振语义（非均线族）+ 实验 gate。
- 共振 beta/corr 在高 beta 池内方差可能小（都高 beta）。mitigation = 仍提供、模型定权重、消融精简。
- 多重共线（bias_5/20, vol_ratio 同窗口）：LGBM 鲁棒，后续消融。

## 7. 待确认决策点（governance：不确定需确认）
- **Q1**：handler 走合体(31) 还是单独(13) 先验证？（影响实现路径）
- **Q2**：13 个因子清单是否认可，要增减？
- **Q3**：上证指数 = SH000001（用户语义「上证指数」），确认？（vs 当前 benchmark SH000300）

## 8. qlib 原生依据（不偏离扩展凭证）
- `ChangeInstrument`（ops.py:64）— 跨 instrument 引用指数
- `Cov/Var/Corr/Mean/Ref/Max/Min/Std` — 原生滚动算子
- `get_feature_config` 返回 expression — 原生 DataLoader 特征层
- handler 子类化 — Alpha158→HighBetaAlpha158→MinuteEnhancedHandler 链
- label/deal_price/TopkDropoutStrategyTD0 冻结不动
