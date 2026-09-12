---
layout: default
title: "数据流程（历史）"
nav_order: 12
---

# 数据流程

> ⚠️ **历史文档（MVP baseline 期）**：本文描述日频 Alpha158 158 因子基线的数据流，
> 项目已演进到 18 分钟因子 + 盘中生产链路。**数据流概述仍有参考价值，但具体模块
> 引用已过时**。现役数据流见 [盘中生产设计](superpowers/specs/2026-07-20-intraday-production-signal-design.md)。

> 本项目怎么从「原始行情 + 成分股名单」一步步变成「回测报告」。
> 配套文档：架构见 [architecture.md](architecture.md)，技术决策见 [technical-design.md](technical-design.md)。
>
> ⚠️ **本文是 MVP baseline 时期的通俗版叙述**（日频 `Alpha158` 158 因子 + 旧 label `Ref($close,-2)/Ref($open,-1)-1`，IC≈0 baseline）。项目已演进到 **champion = enhanced(18)@n_drop=15**（14 个 T 日 9:30-9:40 分钟因子 + 4 extra，test 2026-04→07 +158.86% w/cost / IC 0.0545 / ICIR 5.12），**label 冻结为 `Ref($close,-1)/$price_941-1`**（T+1 收盘 / T 日 9:41 价 − 1）、`deal_price=["$price_941","$close"]`、策略换 `TopkDropoutStrategyTD0`（9:41 成交）、overlay 每股 30 bins。下文带 ⚡ 的为演进后现状，完整 champion 流程见 [backtest-log §22](backtest-log/2026-07-06-l1-full-backtest.md)。

## 全链路一张图

```
   qlib_data (只读)                iFinD p03473
   7 字段 × 26 年                   883926 每日成分股快照
        │                                 │
        │ symlink 复用                    │ 拉 603 天（T 日盘前快照）
        ▼                                 ▼
   ┌──────── data/qlib_root（overlay，可写层）─────────────┐
   │  calendars/            ← 整目录链接 qlib_data         │
   │  instruments/all.txt   ← 链接                         │
   │  instruments/highbeta883926.txt  自有（时变股池）      │
   │  features/<每只票>/    7 base bin←链接 + 3 衍生 bin 自有│
   │  features/sh000300/    ← 链接（沪深300 基准）          │
   └──────────────────────────────────────────────────────┘
                           │
                           ▼  qlib.init(provider_uri = overlay)
              ┌─────────────────────────────┐
              │  Alpha158（qlib 内置 158 因子）│
              │  只用 7 字段，零自定义因子      │
              │  label = open[T+1] → close[T+2]│
              └─────────────────────────────┘
                           │
                           ▼
              LGBModel 训练 → 给每只票打预测分
                           │
                           ▼
              TopkDropoutStrategy 回测
              deal_price = ["$open","$close"]（与 label 对齐）
                           │
                           ▼
              mlruns/ 产物：IC / 净值曲线 / 换手率 / 报告
```

> ⚡ 上图是 **MVP baseline 视角**（Alpha158 + 旧 label `Ref($close,-2)/Ref($open,-1)-1` + `deal_price=[$open,$close]` + 每股 10 bin）。**champion 视角**：因子 = MinuteEnhancedHandler 的 18 个 T 日分钟因子、label = `Ref($close,-1)/$price_941-1`、deal_price = `[$price_941,$close]`、策略 = TopkDropoutStrategyTD0（9:41 成交）、overlay 每股 30 bins。详见 [backtest-log §22](backtest-log/2026-07-06-l1-full-backtest.md)。

下面按环节拆开讲。

---

## 1. 数据来源（两块，都是只读消费）

| 来源 | 内容 | 怎么访问 |
|---|---|---|
| `/home/zxh/qlib_data` | 7 字段（open/high/low/close/volume/factor/vwap）× 6419 天（2000-01-04 → 2026-07-02），全 A 股 + 主要指数 | 文件系统，**只读** |
| `/home/zxh/cn_data_1min` | ⚡ 1min 行情（分钟因子源，champion 用） | 文件系统，**只读** |
| iFinD `p03473` 接口 | 883926 每日的成分股名单（每日约 100 只） | HTTPS + token |

本项目不生产行情数据，行情全部来自 qlib_data。iFinD 只用来取 883926 的成分股名单。

---

## 2. 股池准备（highbeta883926 时变池）

这是数据流程的起点，也是最容易出错的一步。

### 2.1 为什么要专门做股池

883926（同花顺高贝塔值指数）是个**每日重平衡**的高贝塔榜——每天成分股都在变，每日换手率约 80-90%。这不是稳定的成分股集合（不像沪深300 半年调一次），而是每天一张新名单。

如果偷懒用「今天的成分股」回测整个 2024-2026，就会引入**未来函数**（幸存者偏差）：用 2026 年 7 月才知道的名单，去交易 2024 年的股票。所以必须按「历史上每一天当时的真实名单」来。

### 2.2 怎么拉历史名单

- 接口：iFinD `data_pool p03473`，参数 `iv_date=YYYYMMDD`（历史日期也支持）、`iv_zsdm=883926.TI`。
- 范围：2024-01-01 → 2026-07-02，共 **603 个交易日**。
- 实现：[universe.py](../qlib_ifind_beta/universe.py) 的 `fetch_history_snapshots` 逐日拉，缓存到 `data/universe_snapshots.csv`（断点续拉，每 50 天落盘一次）。
- 规模：603 天里累计出现过 **5116 只**不同的票，每只平均在池里约 12 天、分约 10 段进出。

### 2.3 T 日盘前更新（无前视，核心）

> **2026-07-10 修正**：经实测验证，883926 股池是 T 日盘前更新的（非此前假设的 T-1 日收盘后更新）。T 日开盘前即可获取当日最新成分股，无需 T-1 lag。

> **T 日能用的股池 = 883926 的 T 日在册集**

每一段成分股的进出区间 `[入场日 d_in, 离场日 d_out]`，直接写入 instruments 文件（不再右移）。qlib 在 T 日取数时返回 T 日在册集，盘前更新 ≪ 9:30 开盘 ≪ 9:41 决策 → 无前视。

> 历史：2026-07-05 ~ 07-09 使用 T-1 lag（shift_T1 右移 +1 交易日），基于「p03473 给的是 D 日收盘后名单」的假设。2026-07-10 验证推翻该假设后取消 shift。`shift_T1` 函数保留在 universe.py 中供参考。

### 2.4 产物

`data/qlib_root/instruments/highbeta883926.txt`，TSV 格式 `code \t start_date \t end_date`，每只票可能有多段（进出多次）。

### 2.5 验证

`/tmp/verify_t1.py` 做了端到端不变量检查：随机抽 6 个交易日 T，验证 `qlib 取到的 T 日股池 == 缓存里 T 日的快照`。结果 **6/6 全过**（无漏票、无多票）。

---

## 3. 数据落盘（overlay 叠加层）

### 3.1 为什么需要 overlay

qlib 只认一个数据根目录（`provider_uri`）。但 qlib_data 是只读的，而我们需要：
- 加 3 个衍生字段（涨跌停用，见第 5 节）；
- 放自有的股池文件（highbeta883926.txt）。

不能往只读的 qlib_data 里塞东西，又不能给 qlib 两个根。解法是搭一层**可写的叠加目录** `data/qlib_root/`：用软链接（symlink）把 qlib_data 的内容「指过来」，衍生字段和自有股池用真实文件写进去。qlib 只看到一个根，里面是「qlib_data 的视图 + 我们自己的东西」的合并。

类比 Docker 镜像层：下层 qlib_data 只读，上层 qlib_root 可写，叠在一起 qlib 看到完整画面。

### 3.2 结构

| 路径 | 类型 | 说明 |
|---|---|---|
| `calendars/` | symlink → qlib_data | 交易日历，整目录复用 |
| `instruments/all.txt` | symlink → qlib_data | 全 A 股名单，复用 |
| `instruments/highbeta883926.txt` | **真实文件** | 883926 时变股池（第 2 节产物） |
| `features/<每只票>/` | **真实目录** | ⚡ 每股 30 bins：7 base bin symlink 指回 qlib_data + 23 个真实 bin（3 衍生 + 14 分钟因子 + 4 extra + price_941 + change_941） |
| `features/sh000300/` | symlink → qlib_data | 沪深300 基准行情，整目录复用 |

为什么 `features/<票>/` 是真实目录、里面再 symlink？因为要往同一个目录里塞衍生 bin，目录必须可写；而 7 个 base bin 不重写，用 symlink 零拷贝复用。

### 3.3 构建

`conda run -n qlib_ifind_beta python -m scripts.build_overlay`，幂等可重跑。最近一次：5116 只票里 5040 只成功、76 只 qlib_data 缺 bin（已退市/停牌早，qlib 返回空不崩）。

---

## 4. 数据过滤（qlib 自动按股池 + 日期筛）

过滤不需要手写——qlib 的 `D.features(instruments, fields, start_time, end_time)` 会读 instruments 文件里的日期段，**自动只取该票在池期间的数据**。

时变股池的效果就体现在这：
- T 日取数时，qlib 只返回 T 日在册的那些票（第 2.3 节的 T 日盘前更新在这里生效）；
- 某只票不在池的日期，qlib 直接不返回它的数据——自动剔除「未来才入池」的票，无前视。

容错：76 只缺 bin 的票，qlib 对它们返回空 Series（全 NaN），不报错、不影响其他票。

---

## 5. 衍生字段（涨跌停判断用，**不是因子**）

这 3 个字段模型不用，是给**回测的交易规则**用的。

| 字段 | 含义 | 怎么算 |
|---|---|---|
| `change` | 当天涨跌幅 | 用**不复权价**算：`($close/$factor) / Ref($close/$factor, 1) - 1`。为什么用不复权？后复权价在除权日会跳，会误判涨停。 |
| `limit_up` | 这只票的涨停线 | 按板块分：主板 ±10%、创业板/科创板 ±20%、北交所 ±30%。物化时取略低于名义限（0.095/0.195/0.295）规避浮点边界 |
| `limit_down` | 跌停线 | 同上，取负值 |

回测配置里这行就用它们：
```yaml
limit_threshold: ["$change >= $limit_up", "$change <= $limit_down"]
```
意思是：当天涨跌幅碰到涨停线 → 禁止买入；碰到跌停线 → 禁止卖出。第一个表达式拦买入，第二个拦卖出。

实现：[materialize.py](../qlib_ifind_beta/materialize.py) 把这 3 个字段算好、物化成 bin 放进 overlay。

---

## 6. 因子计算（Alpha158）

> ⚡ **champion 已演进**：本节原写 MVP baseline（Alpha158 158 因子，IC≈0）。champion = [MinuteEnhancedHandler](../qlib_ifind_beta/minute_enhanced_handler.py) 的 **18 个 T 日 9:30-9:40 分钟因子**（14 baseline：startup_mom/accel/close_pos/vol_ratio ×{1,3,5} + vol_vs_yest；+ 4 extra：vol_vs_yest_t2/t3/t5 + overnight_gap），**label 冻结为 `Ref($close,-1)/$price_941-1`**（T+1 收盘 / T 日 9:41 价 − 1，~1.5 天 horizon），见下方 §6.2 历史口径对照。下文 6.1/6.2 保留 MVP 原文，champion 完整结果见 [backtest-log §22](backtest-log/2026-07-06-l1-full-backtest.md)。

### 6.1 用什么因子

qlib 内置的 **Alpha158**——158 个现成因子，全部只用 7 个基础字段（open/high/low/close/volume/factor/vwap）算出来。包含：

- K 线形态（KMID/KLEN/KUP/KLOW 等）
- 量价（成交量比、成交额相关）
- 滚动统计（N 日均值、标准差、分位数等，N=5/10/20/30/60）
- 均线、动量类（MA、MOM、ROCR 等）

**零自定义因子**——这是 MVP 的定位（先用基线跑通，自定义因子起步集留到 baseline 验证后再加，见 [technical-design.md §8](technical-design.md)）。

### 6.2 训练目标（label）

```yaml
label: [Ref($close, -2) / Ref($open, -1) - 1]
```

含义：T 日收盘出信号 → **T+1 开盘买入** → **T+2 收盘卖出** 的收益率。

这是 Alpha158 默认 label（close→close）的变体，分母改成 open，更贴近「信号出来第二天开盘才成交」的实际节奏。

---

## 7. 训练 + 预测 + 回测

### 7.1 切分（仅用 2024-2026 约 2.5 年）

| 段 | 区间 | 用途 |
|---|---|---|
| train | 2024-01-01 → 2025-12-31（~2 年） | 训练 |
| valid | 2026-01-01 → 2026-03-31 | 早停 |
| test | 2026-04-01 → 2026-07-02 | 样本外预测 + IC |

### 7.2 模型

`LGBModel`（LightGBM），标准 qlib contrib 类，零自定义。

### 7.3 回测撮合（与 label 对齐）

```yaml
deal_price: ["$open", "$close"]   # 买入用开盘、卖出用收盘
```

qlib 原生支持买卖不同价（[exchange.py:44/157-164](file:///home/zxh/miniconda3/envs/qlib_ifind_beta/lib/python3.12/site-packages/qlib/backtest/exchange.py)）。这样回测的买卖点和 label 完全一致：买在 open[T+1]、卖在 close[T+2]。

> ⚡ **champion 撮合口径（冻结，禁止修改）**：`deal_price=["$price_941","$close"]`——买入用 **T 日 9:41 价**、卖出用 **T+1 收盘**；策略换 [TopkDropoutStrategyTD0](../qlib_ifind_beta/td0_strategy.py)（`shift=1→0` 实现 T 日 9:41 成交），与冻结 label `Ref($close,-1)/$price_941-1` 完全对齐（注：label 已于 2026-09-12 升级 v2 分钟口径，见 backtest-log；deal_price 未变）。本节 `$open/$close` 口径是 MVP baseline 历史。

策略：`TopkDropoutStrategy(topk=20, n_drop=5, hold_thresh=1, forbid_all_trade_at_limit=true)`
- topk=20：每天选预测分最高的 20 只
- hold_thresh=1：持仓至少 1 天，原生强制 A 股 T+1
- forbid_all_trade_at_limit：涨跌停禁交易（依赖第 5 节的衍生字段）

### 7.4 产物

`mlruns/<experiment_id>/<recorder_id>/` 下：`pred.pkl`（预测分）、`label.pkl`、IC 指标、净值曲线、换手率、回撤等报告。

---

## 全流程图（本次全量跑，~17 秒）

> ⚡ **历史快照**（recorder `79d912a4`，2026-07-05 MVP baseline）——下图是 **Alpha158 + 旧 label（IC≈0）** 时期的实跑记录，保留作 baseline 对照。当前 champion（enhanced(18)@n_drop=15，test +158.86% w/cost / IC 0.0545 / ICIR 5.12）的完整流程与指标见 [backtest-log §22](backtest-log/2026-07-06-l1-full-backtest.md)。

> 上面「全链路一张图」是**数据视角**（数据怎么流进来）；下面这张是**配置执行视角**——`workflow.yaml` 从 init 到报告产物的完整执行链，并附本次实跑（recorder `79d912a4`，2026-07-05）的关键指标。

```
qlib.init(provider_uri = data/qlib_root)    ← overlay：qlib_data 只读视图 + 自有衍生字段/股池
      │
      ▼
┌─ Alpha158 Handler（qlib 原生，零子类）─────────────────┐
│  输入：7 字段 × highbeta883926 时变池（T 日盘前更新）    │
│  产出：158 特征 + 1 label                              │
│  label = Ref($close,-2) / Ref($open,-1) - 1            │
│        （T 日出信号 → T+1 开盘买 → T+2 收盘卖）         │
│  processors：DropnaLabel + CSZScoreNorm（仅 label 截面  │
│              标准化；特征不标准化——LightGBM 尺度无关）  │
└────────────────────────────────────────────────────────┘
      │
      ▼
┌─ DatasetH 切分（2024-2026，~2.5 年）──────────────────┐
│  train  2024-01-01 → 2025-12-31   训练（~2 年）        │
│  valid  2026-01-01 → 2026-03-31    早停依据            │
│  test   2026-04-01 → 2026-07-02    样本外              │
└────────────────────────────────────────────────────────┘
      │
      ▼
┌─ LGBModel（LightGBM）─────────────────────────────────┐
│  num_boost_round=200，early_stopping_rounds=20         │
│  本次：第 20 轮早停（valid 不再降）                    │
│  → 每只票 × 每天的 score（预测分）                      │
│  本次 test 段：pred / label 各 6198 行（dropna 后 5971）│
└────────────────────────────────────────────────────────┘
      │
      ▼   qlib Recorder 三段分析链（task.record）
┌────────────────────────────────────────────────────────┐
│ ① SignalRecord   → pred.pkl + label.pkl                │
│                                                         │
│ ② SigAnaRecord   → sig_analysis/{ic, ric}.pkl          │
│    （模型层）    本次 IC = -0.0078 ± 0.140  ICIR=-0.055 │
│                  Rank IC = -0.0072  胜率 48.4%（≈ 随机）│
│                                                         │
│ ③ PortAnaRecord  → portfolio_analysis/*.pkl            │
│    （组合层）    Strategy = TopkDropoutStrategy         │
│                    topk=20  n_drop=5  hold_thresh=1     │
│                    forbid_all_trade_at_limit            │
│                  Exchange  deal_price = [$open, $close] │
│                    limit_threshold =                    │
│                      [ $change ≥ $limit_up,             │
│                        $change ≤ $limit_down ]          │
│                    开 0.05% / 平 0.15% / 最小 5 元       │
│                  benchmark = SH000300                   │
│                  本次回测 61 天（test 62 天，回测       │
│                    end=2026-07-01 规避日历末日越界）     │
│                  换手均值 45.0% / 最大 90.2%            │
│                  基准年化 +44.85%  超额年化 -37%        │
│                  成本年化 -10.72%                       │
└────────────────────────────────────────────────────────┘
      │
      ▼   mlruns/264165997523727437/79d912a4…/artifacts/
┌─ 产物清单（全 .pkl）──────────────────────────────────┐
│  pred.pkl / label.pkl                                  │
│  sig_analysis/{ic, ric}.pkl                            │
│  portfolio_analysis/{report_normal, port_analysis,     │
│                     positions_normal}_1day.pkl         │
└────────────────────────────────────────────────────────┘
      │
      ▼   conda run -n qlib_ifind_beta python scripts/make_report.py
┌─ 报告（13 个 HTML，plotly 交互式，reports/）──────────┐
│  01_model_performance_* (6)  IC 时序 / 累积 IC / 月度  │
│                              热力图 / 分组收益 / QQ /  │
│                              自相关 / 换手             │
│  02_report (1)               净值 / 回撤 / 换手 /      │
│                              成本 / 超额               │
│  03_risk_analysis_* (5)      月度收益 / 年度对比 等    │
│  04_score_ic (1)             score IC 时序             │
│  （HTML 内嵌 plotly.js，离线可看）                     │
└────────────────────────────────────────────────────────┘
```

**本次全量跑耗时 ~17 秒**——快是因为数据量小：test 段每天约 100 只票，158 特征 × 6198 行，LightGBM 训练 + 61 天回测都很轻。

> ⚠️ 以上 IC≈0、超额为负是 **baseline 快照**，不是最终策略效果——Alpha158 默认周期与日频 label + 高贝塔池当前不匹配，信噪比低。改进方向见 [technical-design.md §8](technical-design.md)（自定义因子 / `n_drop` 调小 等）。

---

## 关键文件索引

| 文件 | 作用 |
|---|---|
| [config.py](../qlib_ifind_beta/config.py) | 集中配置：路径、字段、代码、URL、分钟 slot 映射 |
| [universe.py](../qlib_ifind_beta/universe.py) | 883926 时变股池（p03473 拉取，T 日盘前更新） |
| [overlay.py](../qlib_ifind_beta/overlay.py) | symlink 叠加层构建 |
| [materialize.py](../qlib_ifind_beta/materialize.py) | 3 个衍生字段物化（涨跌停用） |
| [minute_factors.py](../qlib_ifind_beta/minute_factors.py) | ⚡ 14 个 T 日 9:30-9:40 分钟因子（纯函数） |
| [materialize_minute.py](../qlib_ifind_beta/materialize_minute.py) | ⚡ 分钟因子 + price_941/change_941 物化 |
| [highbeta_handler.py](../qlib_ifind_beta/highbeta_handler.py) | ⚡ HighBetaAlpha158（Alpha158 子类 + L1 前视护栏） |
| [minute_enhanced_handler.py](../qlib_ifind_beta/minute_enhanced_handler.py) | ⚡ ★ champion Handler（18 因子） |
| [td0_strategy.py](../qlib_ifind_beta/td0_strategy.py) | ⚡ TopkDropoutStrategyTD0（shift=0，9:41 成交） |
| [ifind.py](../qlib_ifind_beta/ifind.py) | iFinD HTTP 客户端 + token |
| [scripts/build_overlay.py](../scripts/build_overlay.py) | 一次性编排：端到端建 overlay |
| [scripts/materialize_minute.py](../scripts/materialize_minute.py) | ⚡ 仅重物化分钟因子（改公式后免重拉 universe） |
| [qrun/workflow.yaml](../qrun/workflow.yaml) | MVP 全量回测配置（Alpha158） |
| [qrun/workflow_minute_enhanced_tk10_nd8.yaml](../qrun/workflow_minute_enhanced_tk10_nd8.yaml) | ⚡ ★ champion 配置（enhanced(18)@topk10/n_drop=8） |
| [qrun/workflow_minute_enhanced.yaml](../qrun/workflow_minute_enhanced.yaml) | legacy LGB / n_drop=15 配置 |
| [qrun/workflow_smoke.yaml](../qrun/workflow_smoke.yaml) | 烟雾测试配置（2025 子窗口） |
