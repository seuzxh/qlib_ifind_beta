# 日频情绪 + 上证指数共振因子 实现计划（2026-07-07）

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development 或 executing-plans。Steps 用 `- [ ]` 跟踪。
> 承接 spec：[docs/superpowers/specs/2026-07-07-daily-index-factors-design.md](../specs/2026-07-07-daily-index-factors-design.md)（13 因子表达式全文 + 前视口径 + 架构依据，本文不重复）。
> 用户 Q1 决策：**两步走**（Step A 单独 13 验证 → Step B 合体 31 对比 champion）。

**Goal:** 在 champion=enhanced(18)@n_drop=15 基础上，新增 7 日频情绪 + 6 上证指数共振因子，验证能否提升预测准确性。

**Architecture:** 13 因子全为 qlib expression（Ref1 lag1 防前视 + ChangeInstrument 跨 instrument），不物化、不新增 bin（仅 SH000001 symlink）。Step A 单独 handler 验证 IC，Step B 合体 31 对比 champion 回测。

**Tech Stack:** qlib 0.9.7 原生 ChangeInstrument/Cov/Var/Corr/Mean/Ref/Max/Min + Alpha158 子类化链 + LGBModel + TopkDropoutStrategyTD0。

**硬约束（冻结，不可改）：**
- label = `Ref($close,-1)/$price_941-1`
- deal_price = `["$price_941","$close"]`
- limit_threshold = `["$change_941 >= $limit_up", "$change <= $limit_down"]`
- strategy = TopkDropoutStrategyTD0, n_drop=15, topk=20, hold_thresh=1
- benchmark = SH000300（SH000001 仅作因子引用源，不替换 benchmark）
- 前视对齐：所有日频/指数因子必须 `Ref(expr,1)`（T 行 = T-1 数据算的值）；分钟因子不 lag（9:30-9:40 当天已知）

---

## File Structure

| 文件 | 责任 | 任务 |
|---|---|---|
| `qlib_ifind_beta/config.py` | 加 `INDEX_FACTOR_SOURCES=("SH000001",)` 常量 | T1 |
| `scripts/build_overlay.py` | step6 循环 `link_stock(idx)` link 指数源 | T1 |
| `qlib_ifind_beta/index_daily_handler.py` | **新建** IndexDailyHandler（13 expression，Step A） | T3 |
| `qlib_ifind_beta/enhanced_daily_index_handler.py` | **新建** EnhancedWithDailyIndex（18+13=31，Step B） | T6 |
| `qrun/workflow_daily_index.yaml` | Step A workflow（复制 enhanced 换 handler） | T4 |
| `qrun/workflow_enhanced_daily_index.yaml` | Step B workflow（合体） | T6 |
| `tests/test_index_factors.py` | **新建** 13 expression 正确性 + 前视测试 | T2 |

---

## Task 1: 数据层 — SH000001 link

**Files:** Modify `qlib_ifind_beta/config.py`, `scripts/build_overlay.py`

- [ ] **Step 1:** config.py 加 `INDEX_FACTOR_SOURCES = ("SH000001",)`（区别于 BENCHMARK=SH000300；注释说明"因子引用源，非 benchmark"）
- [ ] **Step 2:** build_overlay.py 在 step5 benchmark 后加 step6：`for idx in INDEX_FACTOR_SOURCES: overlay.link_stock(idx)`（仅 link 7 base bin，不 materialize——指数不交易）
- [ ] **Step 3:** 单独执行 link（不跑全量 build）：`conda run -n qlib_ifind_beta python -c "from qlib_ifind_beta import overlay; overlay.link_stock('SH000001')"`
- [ ] **Step 4:** 验证：`D.features(['sh000001'], ['$close','$volume','$high','$low'], start='2024-01-02', end='2024-01-10')` 非空连续
- [ ] **Step 5:** Commit `feat(daily-index): T1 SH000001 overlay link`

## Task 2: 因子表达式正确性 + 前视测试

**Files:** Create `tests/test_index_factors.py`（独立 pytest，直接 `D.features`，不走 handler）

spec §3.3 + §4 的 13 expression 全文。前视测试核心断言：**因子[T] == 用 ≤T-1 数据手算的值**。

- [ ] **Step 1:** 写 `test_sh000001_changeinst`：`D.features(['sh600519'], ["ChangeInstrument('SH000001',$close)"])` 按 datetime 对齐 == `D.features(['sh000001'], ['$close'])`
- [ ] **Step 2:** 写日频族正确性（手算 pandas rolling 对照，各测 1 个代表）：
  - `test_bias5_lag1`：bias_5[T] == (close[T-1]-mean(close[T-5..T-1]))/mean
  - `test_rsv9_lag1`：rsv_9[T] == (close[T-1]-min(low[T-9..T-1]))/(max(high[T-9..T-1])-min(low...))
  - `test_vol_ratio_20_lag1`、`test_run_up_5_lag1`、`test_accel_mom_lag1`、`test_dist_to_limit_lag1`、`test_bias20_lag1`
- [ ] **Step 3:** 写指数族正确性（含 ChangeInstrument 内联展开）：
  - `test_idx_bias20_lag1`：idx_bias_20[T] == (idx_close[T-1]-mean(idx_close[T-20..T-1]))/mean
  - `test_beta20_lag1`：beta_20[T] == cov(stock_ret, idx_ret, 20 ending T-1)/var(idx_ret, 20 ending T-1)（pandas cov/var 手算）
  - `test_corr_20_lag1`、`test_idx_rsv9_lag1`、`test_idx_run5_lag1`、`test_idx_vol_ratio_20_lag1`
- [ ] **Step 4:** 写前视 gate（最关键）：`test_no_lookahead`——取 sh600519 某段，对每个因子断言「mask T 日及之后所有数据，因子在 T 行的值不变」（Ref1 保证；等价于因子[T] 不依赖 ≥T 数据）
- [ ] **Step 5:** `conda run -n qlib_ifind_beta pytest tests/test_index_factors.py -v` 全绿
- [ ] **Step 6:** Commit `test(daily-index): T2 13 expression 正确性 + 前视 gate`

## Task 3: IndexDailyHandler（Step A 单独 13）

**Files:** Create `qlib_ifind_beta/index_daily_handler.py`

- [ ] **Step 1:** 写 `IndexDailyHandler(HighBetaAlpha158)`，override `get_feature_config` 返回 13 expression（fields）+ 13 name。expression 用 Python 变量 `idx_close="ChangeInstrument('SH000001', $close)"` 等拼字符串保持可读。继承 HighBetaAlpha158 复用 L1 feature-drop 护栏 + __init__
- [ ] **Step 2:** docstring 注明：① Step A 单独验证用；② 无分钟因子联动，p941 前视护栏弱于 champion（exchange 回退路径风险），主看 IC（不走 exchange，干净）；③ 13 expression 全 Ref1 lag
- [ ] **Step 3:** `test_index_daily_handler_config`：handler.get_feature_config() 返回 len==13、names 含 bias_5/beta_20 等、fields 全含 `Ref(...,1)`
- [ ] **Step 4:** Commit `feat(daily-index): T3 IndexDailyHandler（13 因子，Step A）`

## Task 4: workflow_daily_index.yaml（Step A）

**Files:** Create `qrun/workflow_daily_index.yaml`

- [ ] **Step 1:** 复制 workflow_minute_enhanced.yaml，换：experiment_name→daily_index、handler class→IndexDailyHandler、module_path→qlib_ifind_beta.index_daily_handler
- [ ] **Step 2:** 确认 label/deal_price/limit_threshold/strategy/n_drop=15/benchmark/segments 全与 enhanced 一致（冻结不动）
- [ ] **Step 3:** Commit `feat(daily-index): T4 workflow_daily_index.yaml`

## Task 5: Step A 评估

- [ ] **Step 1:** `conda run -n qlib_ifind_beta python qrun/run.py qrun/workflow_daily_index.yaml`
- [ ] **Step 2:** 读 SigAnaRecord 输出：valid + test 段 IC/ICIR vs champion(0.0545/5.12)
- [ ] **Step 3:** 决策 gate：IC>0 且 |ICIR|>0.3 → 进 Step B；否则保留 champion、记录负结果（full(85) 教训）
- [ ] **Step 4:** 若跑回测，注明"单独 handler p941 护栏弱于 champion，回测数字仅供参考，以 IC 为准"

## Task 6: Step B 合体 31 + 评估

**Files:** Create `qlib_ifind_beta/enhanced_daily_index_handler.py`, `qrun/workflow_enhanced_daily_index.yaml`

- [ ] **Step 1:** `EnhancedWithDailyIndex(MinuteEnhancedHandler)` override get_feature_config = 18 分钟($field 不lag) + 13 日频(expression Ref1) = 31。复用 champion 完整 L1 护栏
- [ ] **Step 2:** workflow_enhanced_daily_index.yaml（复制 enhanced 换 handler）
- [ ] **Step 3:** 跑 qrun，valid+test IC/ICIR + test 回测 vs champion(+158.86%/-5.44%)
- [ ] **Step 4:** Gate：valid+test 双优才升级 champion；否则保留
- [ ] **Step 5:** Commit

## Task 7: 记录

- [ ] **Step 1:** backtest-log 追加新 §（Step A/B 结果 + 因子清单 + 前视口径 + 决策）
- [ ] **Step 2:** Get笔记 note_id 1914664125624050528 追加 `## [2026-07-07] 日频情绪 + 上证指数共振因子（两步走实现）`
- [ ] **Step 3:** ripple 核查（CLAUDE.md/technical-design/因子表）

---

## Verification 总表

1. conda 强制：所有命令 `conda run -n qlib_ifind_beta`
2. SH000001 link：D.features 非空（T1）
3. 13 expression 正确性 + 前视 gate（T2，最关键）
4. handler get_feature_config 契约（T3）
5. Step A IC vs champion（T5 gate）
6. Step B 回测 vs champion（T6 gate）
