# 特征组横截面 Rank 归一化（CSRankNorm 替换变体）— 设计 spec

> 日期：2026-07-07
> 状态：**已实现 + viability-gate FAIL（2026-07-07）**，负结果归档 backtest-log §24。横截面 rank 族在此构造下死；是否试 C（CSZScoreNorm）待用户决策。
> 上游决策链：`/goal`（优化因子提升超额与 IC）→ 三重证伪（gap 诊断 / 过滤层 / 头部可塑性）→ 用户选「横截面相对因子族」→ context7+源码重定性为 PROCESSOR 层变体 → 用户选方案 A → 实现 + 实验 → 全口径 FAIL

---

## 1. 动机与第一性原理

champion `enhanced(18)@n_drop=15` 的 18 个因子全是**绝对**开盘幅度（startup_mom / accel / close_pos / vol_ratio / overnight_gap 等）。highbeta883926 池每日成分换手 ~90%，同一个 +2% 开盘幅度在平静日是强信号、在沸腾日是弱信号——**日级 regime 混入因子值**，混淆了「个股相对强度」与「当日市场温度」。

假设：对特征做**日内池内横截面 rank 归一化**，强制模型只用「当日相对位置」，剥离日级 regime → 提纯相对强度 → 可能改善头部选股。

### 1.1 关键障碍（决定了无廉价预筛）
Spearman RankIC 对**日内单调变换不变**：CSRankNorm 是日内 rank（单调），故单因子每日 RankIC 与原始值**完全相同**。横截面假设本质是**多元组合问题**（改变 18 因子如何叠加），只能靠「训练带 CSRankNorm 特征的 GBDT」验，**无单因子预筛**。实现本身（一行处理器）即最廉价的判定实验。

### 1.2 必须坦白的风险（gap 诊断的墙）
gap 诊断已证 **IC↑≠头部↑**：Step B 把 test IC 从 0.0545 抬到 0.0611，头部收益反降。横截面归一化**机制不同**（重构同 18 因子，非加因子稀释），但**同样无保证** IC↑传导到 top-20 头部。故**首次训练即 viability gate**。

---

## 2. 重新定性：PROCESSOR 层变体，非新因子族

context7 + qlib 源码（`qlib/data/dataset/processor.py`）确认：qlib 横截面归一化**不在表达式层、不需新物化 bin**，在 **PROCESSOR 层**：

| Processor | 机制 | fit / 泄漏 |
|---|---|---|
| **CSRankNorm**（processor.py:326） | `groupby("datetime").rank(pct=True)` → `-0.5` → `*3.46` | **无 fit，每日独立** → 零跨段泄漏 |
| CSZScoreNorm（processor.py:300） | 每日横截面 z-score | 无 fit |
| RobustZScoreNorm（processor.py:262） | 时间序列 median/MAD z-score | **有 fit（train 段）** → 非横截面 |

champion 当前 processor 链（Alpha158 contrib 默认，`HighBetaAlpha158` 只覆盖 shared 未动 infer/learn）：
- **特征** = `_DEFAULT_INFER_PROCESSORS` = `RobustZScoreNorm(feature, fit on train, clip)` + `Fillna` → **时间序列** z-score，**非**横截面
- **label** = `_DEFAULT_LEARN_PROCESSORS` = `DropnaLabel` + `CSZScoreNorm(label)` → **已**横截面（每日）

→ 「日内池内 rank 归一化」= 把横截面归一化也施加到 **FEATURE 组**。即用 `CSRankNorm(feature)` **替换** `RobustZScoreNorm(feature)`。这是未试的构造，且 = 改一行 `infer_processors` + 一个 yaml 变体，**零新 bin、零抽象层**（符合 CLAUDE.md「优先 qlib 内置组件」「禁止无关抽象层」）。

---

## 3. 方案选择（用户已选 A）

| 方案 | 做法 | 取舍 |
|---|---|---|
| **A（选定）** | `CSRankNorm(feature)` **替换** `RobustZScoreNorm` | 纯 rank→（label 已横截面）。零泄漏、对开盘幅度极端值最鲁棒、最贴合原意 |
| C | `CSZScoreNorm(feature)` 替换 | 保留日内间距量级（非纯 rank），中庸。A 有部分信号时的第二步 |
| B | 保留 `RobustZScoreNorm` + 追加 `CSRankNorm`（36 列） | GBDT 自选，但翻倍可能重蹈 Step B 稀释覆辙 |

---

## 4. 实现（隔离，不动 champion）

### 4.1 新增 `qlib_ifind_beta/minute_enhanced_csrank_handler.py`
```python
class MinuteEnhancedCSRankHandler(MinuteEnhancedHandler):
    """18 因子 + 特征组横截面 Rank 归一化（CSRankNorm 替换 RobustZScoreNorm）。"""
    _CS_INFER_PROCESSORS = [
        {"class": "CSRankNorm", "kwargs": {"fields_group": "feature"}},
        {"class": "Fillna", "kwargs": {}},
    ]
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("infer_processors", self._CS_INFER_PROCESSORS)
        super().__init__(*args, **kwargs)
```
- `get_feature_config` 不 override → 仍是 18 因子（继承 MinuteEnhancedHandler）。
- shared `DropnaProcessor(feature)` L1 前视护栏从 HighBetaAlpha158 继承（feature-NaN 行先 drop → CSRankNorm 看到完整横截面）。
- label 侧 CSZScoreNorm 不变（Alpha158 默认 learn_processor）。

### 4.2 新增 `qrun/workflow_minute_enhanced_csrank.yaml`
= `workflow_minute_enhanced.yaml` 逐字 clone，仅改：
- `experiment_name: minute_enhanced_csrank`
- handler `class: MinuteEnhancedCSRankHandler` + `module_path: qlib_ifind_beta.minute_enhanced_csrank_handler`
- 注释说明唯一变量 = 特征归一化时间序列→横截面

### 4.3 新增 `tests/test_minute_enhanced_csrank_handler.py`
结构校验（`__new__`，无数据依赖）：
- 18 因子不变（与 MinuteEnhancedHandler 同）
- `infer_processors` 实含 `CSRankNorm(feature)`、不含 `RobustZScoreNorm`
- L1 shared `DropnaProcessor(feature)` 仍继承

---

## 5. 冻结不变（label / 策略 / 切分 / 前视护栏全冻结）

18 因子表达式（`MINUTE_FACTOR_FIELDS` + `EXTRA`）、label `Ref($close,-1)/$price_941-1`、label 侧 CSZScoreNorm、shared `DropnaProcessor(feature)`、train 2024-01-01→2025-12-31 / valid 2026-01→03 / test 2026-04→07-02、`TopkDropoutStrategyTD0 n_drop=15 topk=20 hold_thresh=1 forbid_all_trade_at_limit=true`、benchmark SH000300、LGBModel 超参（lr 0.05 / depth 6 / leaves 64 / l1 5 / l2 10 / 200 round / early 20）、deal_price/limit_threshold/exchange_kwargs——**全部与 champion 逐字一致**。唯一变量 = 特征归一化方式。

---

## 6. 机制正确性（零泄漏证明）

- DataHandlerLP 流：`_learn_df = learn_proc(infer_proc(shared_proc(raw)))`，**infer_processors 对训练+推理都生效** → CSRankNorm 在 train 与 predict 两端都施加。✓
- CSRankNorm **无 `fit()`**，每日用当日横截面独立 rank → **零跨段泄漏、无前视**。✓
- shared `DropnaProcessor(feature)` 先于 CSRankNorm 跑 → feature-NaN 行先 drop → CSRankNorm 在完整横截面上 rank。✓
- label 侧 CSZScoreNorm 不变 → rank 特征 → 横截面 label，口径自洽。✓

---

## 7. Viability-gate 判定规则（首次训练即决断）

单次训练，test 段（2026-04→07）三口径对齐 champion：

| 口径 | champion baseline | 判定用途 |
|---|---|---|
| test IC / RankIC / ICIR | IC 0.0545 / RankICIR ~0.51 | 全局判别力是否变 |
| **头部收益**（top-20 等权实现收益） | **+1.148%/日**（gap 诊断基线） | **关键**：IC→超额传导 |
| 全回测（年化超额 / 回撤 / 换手 / ICIR，含成本） | +158.86% 含成本 | 实盘口径 |

| 结果 | 判定 | 后续 |
|---|---|---|
| 头部收益 ≥ champion **且** 全回测年化超额 ≥ champion（含成本） | **PASS** | 多种子确认 → 升级 champion 候选 |
| IC↑ 但头部收益 < champion（Step-B 墙重现） | **FAIL** | 归档负结果（backtest-log 新 §），族在此构造下死 |
| 头部收益 < champion（无论 IC） | **FAIL** | 同上；可选最后试 C(CSZScoreNorm) |

---

## 8. Ripple
- 新文件 3 个（handler / yaml / test），**不改既有文件**（champion 隔离）。
- `config.py` 不动（复用 `MINUTE_FACTOR_FIELDS` + `EXTRA`）。
- 文档：backtest-log 新增 §（CSRankNorm 实验，依结果 PASS/FAIL）；若 PASS 则 technical-design §D6 追加。
- Get笔记：结果后追加 `## [2026-07-07]`。

---

## 9. 待定
- 若 A FAIL：是否试 C（CSZScoreNorm）作为横截面族最后尝试，待用户决策。
- 若 A PASS：多种子（≥3 seed）确认稳定性后升 champion，待用户确认。
