---
layout: default
title: qlib-factor-zoo 因子库匹配度研究方案（分钟域移植：当日 + 多日）
nav_exclude: true
---

# qlib-factor-zoo 因子库匹配度研究方案（分钟域移植：当日 + 多日）

- 研究分支：`research/factor-zoo-20260924`（worktree `.worktrees/factor-zoo-research`，基线 `2a54281`）
- 上游因子库：<https://github.com/JustinF8/qlib-factor-zoo.git>（MIT，Qlib fork；本地克隆 `/tmp/factor-zoo`，2026-09-24 调研）
- 立项日期：2026-09-24
- 状态：**已完结（2026-09-25）**。四轨道全部闭环 + 替代挑战战役（C1 穿
  purged 后被 TD0 涨停墙否决）+ 多日族 alpha 全量终审（3,322 评 0 幸存，
  §8）——无可替代方案成立，Champion 不变。全部结论见
  `docs/backtest-log/2026-09-24-factor-zoo-screen.md` §0–§8；
  运行资产已两轮清理（fz 研究 bin、全部可再生 pkl 采样/中间部件 ~38G、
  worktree 本地 mlruns 实验、/tmp 脚本、诊断用 diag_portana_deadlock.py），
  仅保留结果文件（16 个 CSV/JSON/digest ~1MB）、dayok universe 与
  validate_factor_zoo_* 全套复现脚本（可确定性重跑再生一切）。
- 修订：v1.1（2026-09-24）用户澄清——原表述"隔日"实为**多日**（T-1/2/3/5），
  且多日族首要用途是判断个股**是否已经加速**、已加速时控制尾部风险；
  §0/§1.2/§1.3/§三/§四/§五/§六/§八 已同步修订；
  v1.2（2026-09-26）多日轨道改为仅关注 alpha（风险框架退出），§8 记录

## 0. 摘要

用户方向：探索 qlib-factor-zoo（六大因子库，约 1013 个因子）中是否有与现有
09:41 分钟级策略**最匹配**的因子；策略是 min 级，除**当日**盘初分钟因子外，
还要求关注**多日**分钟因子——用 T-1/T-2/T-3/T-5 等历史日全天分钟形态判断
个股是否**已经加速**，已加速时注意尾部风险（用户 2026-09-24 澄清，原表述
"隔日"不确切）。

调研核心结论：该因子库**全部为日频表达式**（handler 一律 `freq="day"`，文档
§7.5 明示无分钟因子），不能直接喂给分钟级策略。因此本研究的主假设不是
"从库里挑日频因子"，而是**把库的表达式移植到分钟域**，形成两个全新家族：

1. **当日族（B1）**：zoo 表达式实例化在 T 日 09:31–09:40 十根 1min K 线上
   （与现役 18 因子同窗口、同决策时点）；
2. **多日族（B2）**：zoo 表达式实例化在 T-k（k=1/2/3/5）各日全天 240 根
   1min K 线上，物化为 T 日可用的 `day.bin` 特征（09:41 前可得，无前视）。
   该族定位以**风险识别**为主：构造"是否已经加速"的状态证据，供尾部风险
   层软惩罚/拦截使用，兼作 alpha 模型的候选特征。

同时以"库原生日频 T-1 滞后"作为对照路线（路线 A）。全部筛选必须先过
**graveyard 排雷**（本项目 2026-09-12 前后已证伪大量同类方向），幸存者进入
**冻结 Champion 协议**的增量验证；任何因子不得自动晋升（AGENTS.md 红线）。

## 一、研究问题与"最匹配"的定义

### 1.1 策略合同（现役 Champion，全部冻结）

| 要素 | 值 |
| --- | --- |
| 特征窗口 | T 日 slots 0–9（09:31–09:40 十根闭合 1min K 线） |
| 决策/买入 | slot 10（09:41 close = `price_941`） |
| label v2 | `Ref($close1500, -1) / $close0941 - 1`（T 09:41 买 → T+1 15:00 卖，两腿均 1min 原值） |
| 模型/组合 | HFLGB(binary)，Top10 / n_drop=8 |
| 现役因子 | 18 个（14 开盘分钟 + 4 extra），OOS IC 0.0676 |
| 股池 | `highbeta883926`：每日真实 100 只成分，相邻交易日重合率仅 ~15.9% |

### 1.2 术语约定（用户表述的落地）

- **当日 min 因子**：用 T 日盘初（slots 0–9）分钟数据计算，与 18 因子同时点。
- **多日 min 因子**：用 T-k（k=1/2/3/5）历史交易日**全天** 240 根分钟数据
  计算、在 T 日 09:41 可得的滞后特征（不含 T 日信息）。"多日"不指 T+1——
  T+1 属于 label 腿，任何特征不得使用（`vol_vs_yest_t2/3/5` 即多日族先例）。
- **加速状态与风险**：多日族的首要用途是判断个股在 T 日开盘前**是否已经
  加速**（连续拉升、涨停/炸板、尾盘拉升、高位放量、路径透支等）。已有证据
  （2026-09-12 研究文档）表明高动量样本的次日平均收益并不更低、但崩跌概率
  升高——因此该族按**尾部风险识别**验证（判据 b'），其次才是 alpha 增量；
  禁止退化为均匀负动量惩罚（已证伪，见 §三）。

### 1.3 "最匹配"判据（缺一不可）

因子 F 最匹配，当且仅当：

- **(a) 合同内**：T 09:41 前可得，无前视，无未来成分；
- **(b) 单因子有效**：对 label v2 的日度 Rank IC / IR 通过预设门槛，且在
  筛选段与第二窗口方向一致；
- **(b') 尾部风险判据（加速/风险族专用，可替代 b）**：在"已加速"状态
  子样本上，对 label v2 的下尾收益（bottom 分位收益 / 大亏概率）有稳定
  区分度（两窗口方向一致）；作为 Top10 软惩罚项时改善组合最大回撤 /
  Calmar 且不明显牺牲超额。走 b' 路线的因子不要求全体样本的普通 IC 为正。
- **(c) 增量性**：对 18 个现役因子线性残差化后的 residual IC 不显著衰减
  （≤50%），与现役因子及 graveyard 因子的最大 |corr| 受控；
- **(d) 非重复**：表达式与 graveyard 已证伪因子不同名不同形（自动比对 +
  人工复核）；
- **(e) 组合层增益**：18+k（k≤5）augment 在冻结协议（同 HFLGB 超参、同切分、
  同 Top10/nd8、同成本）下，IC / 超额 / 最大回撤 / Calmar 至少两项改善且
  无一项恶化，purged 19 折滚动验证不劣于 Champion。

## 二、因子库盘点（2026-09-24 调研结论）

### 2.1 库清单

| 库 | 因子数 | 来源 | 表达式位置 | 依赖自定义算子 |
| --- | --- | --- | --- | --- |
| Alpha360 | 360 | Qlib 原生（60 日回溯原始量价） | `loader.py` | 无 |
| Alpha158 | 158 | Qlib 原生（经典统计特征） | `loader.py` | 无 |
| Alpha101 | 101 | WorldQuant (Kakushadze 2015) | `loader_alpha101.py` | TsArgmax/TsArgmin |
| GTJA191 | 191 | 国泰君安 2014 研报 | `loader_gtja191.py` | SMA(通达信)/TsArgmax/TsArgmin/Amount |
| TDXGS | ~90 | 通达信/同花顺指标（MyTT 口径） | `handler.py` 内联 | 40+ 个 TDX 算子 |
| JQ110 | ~113 | 聚宽策略因子（动量/情绪/技术/风险/风格） | `handler.py` 内联 | TDX 算子 + 14 个 JQ 算子 |

合计约 1013 个。许可证 MIT，可 vendor 并注明出处。

### 2.2 关键事实（决定技术路线）

1. **纯日频**：六个 handler 全部硬编码 `freq="day"`；README/使用说明明示
   "仅支持日频数据，不包含分钟级因子"。→ 分钟级策略只能走**表达式移植**。
2. **表达式即资产**：因子都以 qlib 表达式字符串存在（非编译代码），窗口参数
   显式（`N` 可静态解析）→ 可批量改域、改窗重实例化。
3. **50+ 自定义算子**集中在 `custom_ops.py`（2296 行），基于 qlib ops 基类，
   理论上可在 pyqlib 0.9.7 注册，但需冒烟验证（fork 内核与 0.9.7 有差异风险）。
4. **与本项目重叠**：Alpha158 已有 85 个"昨日因子"被测且全灭（见 §三）→
   Alpha158/Alpha360 两库降权为"仅作对照，不期待产出"。
5. **未测空间**：Alpha101 / GTJA191 / TDXGS / JQ110（约 495 个）从未在本项目
   出现；全部 6 库的**分钟域实例化**是全新维度，graveyard 未覆盖。

### 2.3 数据可行性（已核实）

`cn_data_1min` 每股含 `open/high/low/close/volume/vwap/factor` 七个字段的
`1min.bin`，240 真实 bar/日（09:31–15:00）。zoo 全部表达式所需的字段
（close/open/high/low/vwap/volume）在分钟域齐备。`factor` 存在逐日微漂
（±0.1%~0.3%，见 2026-09-11 gap 口径 A/B），分钟域一律用 **1min 原值 +
名义口径**，与 label v2 / overnight_gap 决策保持一致。

## 三、graveyard：本项目已证伪方向（排雷清单）

以下方向已在历史验证中失败，**禁止换名重跑**；新因子候选必须先对照本清单
去重（表达式级比对，非名称级）：

| 已证伪方向 | 证据 |
| --- | --- |
| 85 个 Alpha158/昨日因子（19 折 purged 全灭） | `backtest-log/2026-09-12-alpha158-grid-and-yesterday-factors.md` |
| 14 个昨日 Champion 分钟字段 | 同上 + `docs/research/2026-09-12-...research.md` §1.2 |
| T-1/3/5/10 同一分钟窗口的量/动量/速度差 | 同上 §1.2 |
| 20 个跨日分钟对齐因子 | `backtest-log/2026-09-12-minute-aligned-cross-day-factors.md` |
| 09:41 bar 因子 | `backtest-log/2026-09-12-0941-bar-factors.md` |
| 均匀追高惩罚 / 简单反转（无条件负动量） | 研究文档 执行摘要、§1.2 |
| 日频情绪 + 指数共振（窗口 2 方向反转） | 研究文档 §1.2 |
| 1/3/5/10/20 日动量/波动/量波动替代 overnight_gap | 研究文档 §1.2 |

推论（指导筛选次序）：

- 路线 A（日频 T-1 lag）预期通过率最低，只作对照与算子冒烟；
- 分钟域移植的价值在**新信息形态**（多日全天路径形状、涨停/炸板与尾盘
  结构、分钟级技术指标、资金流/情绪结构），而不是再造与现役共线的
  MOM/STD/VSTD；
- 优先 JQ110（情绪/量能/风险组）与 TDXGS（短窗指标），其次 Alpha101 短窗
  子集，最后 GTJA191（SMA 递归族在 10 bar 内自由度存疑）。

## 四、技术路线

### 路线 A（对照）：库原生日频、T-1 滞后

全部 6 库在 day 数据上按原样计算，`Ref(·, 1)` 后作为 T 日特征。用途：

1. custom_ops 在 pyqlib 0.9.7 的兼容性冒烟（对照 zoo 自带
   `tools/check_six_handlers.py` 的口径）；
2. 给"分钟域是否带来增量"提供日频基线（预期失败，失败本身是结论）。

### 路线 B（核心）：分钟域移植

**B1 当日族**：表达式实例化于 T 日 slots 0–9。

- 静态解析每个表达式的窗口参数，仅保留**全部窗口 ≤ 8**（10 根 bar 留预热
  余量）的表达式；预估 Alpha101/GTJA191/TDXGS/JQ110 各出一小批短窗因子。
- 与 18 因子同槽位、同物化路径（`materialize_minute.py` 模式 → `<name>.day.bin`）。
- 现役因子已覆盖启动动量/加速度/收盘位置/量比/跨日量能/跳空六族；B1 的
  增量预期来自**非线性组合与截面算子**（Alpha101/GTJA191 的 Rank/Corr 结构），
  这是现役集合没有的形态。

**B2 多日族**：表达式实例化于 T-k（k=1/2/3/5）各日全天 slots 0–239，每日
独立一份。定位以**风险识别**为主：判断个股开盘前是否**已经加速**，已加速
则注意尾部风险；兼作 alpha 候选特征。

- 窗口上限 120（保留过半 bar 作有效样本）；两段式窗口（上午/下午）与
  尾盘/开盘分段作为特例观察（尾盘拉升是"隔夜加速"的高价值形态）。
- 物化为 T 日 `day.bin` 特征，命名 `zoo_<lib>_<name>_t{k}`（k∈{1,2,3,5}）。
- 停牌/缺 bar 处理与 bar gate 一致：T-k 日非完整 240 bar 则该日特征 NaN，
  交给 Handler `DropnaProcessor` 护栏，不得回填。
- **加速状态变量**：由多日分钟特征聚合成"已加速"代理（候选：T-1/T-2
  分钟动量水平、连涨天数、涨停/开板次数、尾盘拉升幅度、放量倍数、路径
  最大回撤）。Phase 1 先构造并描述性验证状态变量本身（状态内收益分布、
  状态覆盖率、状态持续性），再让 zoo 因子做交互/细化。
- **消费路径**：①尾部风险层——Top20 内对高加速概率软惩罚或拒绝，λ 只能
  由已结算样本确定、随状态平滑变化（参照 2026-09-12 研究文档 §二分层
  架构）；②alpha 模型特征。**禁止均匀追高惩罚**（graveyard 已证伪），
  必须是条件尾部概率框架。

**截面算子（CsRank/Rank）口径**：T-1 收盘截面用 T-1 成分集合；T 日盘初截面
用 T 日成分集合。前提是 T 日成分在开盘前可得（生产 intraday 依赖此事实），
Phase 0 需向 `update_universe` 的时间戳正式核实并写入审计记录。

### 路线 C（收敛原则）

不是独立路线，而是 B 的筛选次序与淘汰原则：优先新信息形态（情绪/资金流/
风险统计/路径形状），主动放弃与现役 18 因子或 graveyard 相关性 >0.7 的族。

## 五、分阶段任务

### Phase 0 — 基建与审计（本分支，不触碰 Champion）

| # | 任务 | 产出 |
| --- | --- | --- |
| 0.1 | vendor zoo 表达式与算子：拷贝 `loader_alpha101.py`、`loader_gtja191.py`、`custom_ops.py`、`handler.py`（TDXGS/JQ110 表达式部分）到 `qlib_ifind_beta/factor_zoo/`，附来源 commit、LICENSE 摘要 | `qlib_ifind_beta/factor_zoo/` |
| 0.2 | 算子兼容冒烟：pyqlib 0.9.7 注册 custom_ops，抽 ≥10 个表达式在 day 数据求值，与 zoo 自检口径对照 | 冒烟脚本 + 记录 |
| 0.3 | graveyard 清单结构化：解析 §三 引用的 4 份文档，产出因子名/表达式/失败原因的 `excluded.json` | `qlib_ifind_beta/factor_zoo/excluded.json` |
| 0.4 | 数据审计：T 日成分在 T-1 拥有完整 240 bar 的覆盖率；1min factor 漂移下统一名义口径；截面名单时点核实 | 审计小节（写入本文件附录） |
| 0.5 | 表达式静态解析器：从 6 库表达式表提取"窗口参数 ≤N 可实例化"的 B1/B2 候选全集及计数 | 候选清单 CSV |

### Phase 1 — 广谱廉价筛选（cheap falsify）

1. 路线 A 全库日频 T-1 lag 一张大表（`D.features` 一次性）。
2. B1/B2 分钟域批量计算：先 50 只 × 60 日样本验证实现正确性（与手工 numpy
   抽查对照），再全量。计算通道二选一（0.2 冒烟结果决定）：
   a. qlib 1min provider + 注册 zoo 算子；b. numpy 实现 operator kernel 的
   轻量求值器（贴近现有 materialize 管线，不依赖 zoo 内核）。
3. 指标（对 label v2，universe 内）**双轨**：
   - alpha 轨：日度 Rank IC 均值、IR、t 值、覆盖率、IC 半衰；**残差 IC**
     （对 18 因子线性投影后）；与现役 + graveyard 的最大 |corr|。
   - 风险轨（B2 加速/风险因子专用）：**已加速子样本**内的条件 IC 与收益
     分布；bottom-decile 收益和大亏概率（如 label < −5%，口径写入报告）的
     区分度（AUC / 分位单调性）；加速状态变量本身的分布画像。
4. 门槛：alpha 轨 |Rank IC| ≥ 0.015 且 IR ≥ 0.3（约当 Champion 单因子
   水平）、残差 IC 衰减 ≤50%、覆盖率 ≥95%；风险轨只要求尾部区分度稳定
   且两窗口方向一致（不要求普通 IC 为正）；两轨均须 graveyard 去重。
5. 产出：shortlist ≤ 40（alpha 轨 / 风险轨分别列示）+ 加速状态变量档案 +
   筛选报告 `docs/backtest-log/2026-10-xx-factor-zoo-screen.md`
   （报告须注明 screened 总数，供多重检验视角评估）。

### Phase 2 — 冻结协议增量验证（仅 shortlist）

1. 18+k（k≤5）augment：除新增特征外，与
   `qrun/workflow_minute_enhanced_tk10_nd8.yaml` **逐字段一致**
   （handler 类、HFLGB 超参、切分、Top10/nd8、成本、涨跌停约束）。
2. 验证：复用 `rolling_validate.py` 的 purged 19 折门槛 + 第二窗口；
   报告 IC / 超额 / 最大回撤 / Calmar / 换手。
3. 风险层验证（风险轨幸存者）：在冻结 Champion 分数上叠加加速状态软惩罚
   （Top20 内降序/拒绝），λ 只用 train 段已结算样本标定；对照满仓等权基线
   报告最大回撤 / Calmar / 超额的净变化，并验证非加速日不损伤收益。
4. 对照锚点：冻结 Champion 同窗成绩（`CHAMPION_RECORDER_ID`）。
5. 产出：`docs/backtest-log/2026-xx-xx-factor-zoo-augment.md`（alpha 轨
   augment）与 `2026-xx-xx-factor-zoo-overlay.md`（风险层软惩罚），含每个
   候选的失败模式分析（过拟合 / 覆盖率 / 与 18 因子交互 / λ 敏感性）。

### Phase 3 — 结论与晋升建议

1. 汇总研究报告（本目录）。
2. 幸存因子建立候选档案（表达式、窗口、物化口径、验证矩阵），标记
   `CANDIDATE/REVIEW`，**等待人工晋升**；无幸存者则把结论归档进 graveyard，
   明确"分钟域移植 zoo 表达式"这一方向的边界。

## 六、分支与目录布局

```
factor-zoo｜分钟因子匹配与替代挑战  # 原名 research/factor-zoo-20260924；基线 2a54281（= feat/paper-shadow tip，同 optimization-research 惯例）
├── qlib_ifind_beta/factor_zoo/         # vendored 表达式/算子 + excluded.json + 解析器
├── scripts/validate_factor_zoo_screen.py    # Phase 1（含加速状态变量构造）
├── scripts/validate_factor_zoo_augment.py   # Phase 2 alpha 轨
├── scripts/validate_factor_zoo_overlay.py   # Phase 2 风险层（加速软惩罚）
└── docs/research/2026-09-24-factor-zoo-minute-research-plan.md   # 本方案
```

- 不触碰：`qrun/` Champion 配置、`feat/paper-shadow` 工作树的未提交改动、
  生产 CSV、`data/`、`mlruns/`（运行资产不进 Git）。
- 主工作树（feat/paper-shadow）的影子模拟盘 cron 照常运行，互不干扰。

## 七、风险与红线

| 风险 | 对策 |
| --- | --- |
| 前视（T+1 信息、截面名单时点、复权基准） | §1.2 时点合同；0.4 审计；1min 原值名义口径 |
| 多重检验过拟合（千级因子筛选） | 报告 screened 总数；两窗口方向一致；purged 19 折；residual IC 增量门槛 |
| zoo fork 与 pyqlib 0.9.7 算子不兼容 | 0.2 冒烟前置；不兼容则走 numpy 求值器路线（1.2-b） |
| 与现役 18 因子共线（伪增量） | 残差 IC + max-correlation 双门槛 |
| 撞 graveyard（重复研究） | excluded.json 自动比对 + 表达式级人工复核 |
| 覆盖率陷阱（新股/停牌 NaN 回填） | NaN 直落 DropnaProcessor，禁止回填 |

红线（继承 AGENTS.md）：禁止未来数据；禁止自动晋升候选模型；测试全离线
（网络 mock）；只用 conda `qlib_ifind_beta` 环境。

## 八、里程碑

| 里程碑 | 内容 | 验收 |
| --- | --- | --- |
| M1 | Phase 0 完成 | vendored 代码入库、冒烟通过、excluded.json、B1/B2 候选计数表 |
| M2 | Phase 1 完成 | 筛选报告 + 双轨 shortlist ≤40 + 加速状态变量档案（或"无幸存"结论归档） |
| M3 | Phase 2 完成 | 冻结协议 augment / 风险层 overlay 报告，逐项对照判据 (e)/(b') |
| M4 | Phase 3 完成 | 汇总报告 + 候选档案 / graveyard 归档 |

## 附录 A：关键数字速查

- zoo：6 库 / ~1013 因子 / 纯日频 / MIT / fork of Microsoft Qlib
- 本策略：18 因子 / slots 0–9 / buy slot 10（09:41）/ label v2 / Top10/nd8 / OOS IC 0.0676
- 股池：每日 100 只真实成分，相邻日重合 ~15.9%，累计出现过 5140 只
- 1min 数据：240 真实 bar/日，7 字段（OHLCV+vwap+factor），`cn_data_1min`
- 多日族定位：加速状态识别 → 尾部风险层软惩罚（分层架构先例：2026-09-12 研究文档 §二）
- 未测因子空间：Alpha101/GTJA191/TDXGS/JQ110 ≈ 495 个 + 全部 6 库分钟域实例化
