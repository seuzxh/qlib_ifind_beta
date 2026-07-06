# 2026-07-06 L1 后全量回测记录

> 详细流程 + 客观结果 + 异常观察。结论待用户决策（见末尾）。

## 1. 目的

验证 L1 前视护栏（`shared DropnaProcessor(feature)`，commit `985f5dc`）对**训练 + 回测**的端到端影响。此前 L1 只跑了 30 个单元测试 + test 段 NaN 抽样（误杀 4.392% / 防前视正确 drop 1.854%），**未重跑全量回测**——训练侧影响未评估。本次补跑。

## 2. 环境

| 项 | 值 |
|---|---|
| git HEAD | `985f5dc` (fix(highbeta): L1 前视护栏) |
| conda env | `qlib_ifind_beta` (qlib 0.9.7, Python 3.12) |
| yaml | `qrun/workflow.yaml` |
| 入口 | `qrun/run.py`（list→tuple 修正 + mlflow file-store 兜底） |

## 3. 运行流程（可复现）

```bash
conda run -n qlib_ifind_beta --no-capture-output \
  python qrun/run.py qrun/workflow.yaml
```

- **log**：`/tmp/backtest_l1_20260706_193334.log`
- **启动**：19:33:34 → **结束**：19:33:57（**23 秒**，exit=0）
- **recorder**：`5d9353762d8949a49eb34c259b3b3813`（experiment `264165997523727437`）

run.py 关键日志：
```
[run.py] limit_threshold coerce: PortAnaRecord: list(2) → tuple
Time cost: 11.703s | Loading data Done
Time cost: 0.066s | DropnaProcessor Done      ← L1 shared processor
Time cost: 0.023s | DropnaLabel Done          ← Alpha158 默认 learn processor
Time cost: 0.461s | CSZScoreNorm Done
Early stopping, best iteration is: [3] train's l2: 0.981635 valid's l2: 0.988429
```

## 4. 配置摘要（workflow.yaml）

- **handler**：`HighBetaAlpha158`（Alpha158 158 日频字段全 `Ref(...,1)` lag T-1 + 14 个 T 日分钟因子）
- **L1**：`shared_processors` 默认挂 `DropnaProcessor(feature)`（handler `__init__` setdefault；yaml 未显式覆盖 → 生效）
- **label**：`Ref($close,-1) / $price_941 - 1`（T 日 9:41 买入、T+1 收盘卖）
- **切分**：train 2024-01→2025-12 / valid 2026-01→2026-03 / test 2026-04-01→2026-07-02
- **回测**：2026-04-01→2026-07-01（避开 calendar 末日 `day.txt[6419]` 越界）
- **benchmark**：`SH000300`
- **strategy**：`TopkDropoutStrategyTD0`（topk=20, n_drop=5, hold_thresh=1, forbid_all_trade_at_limit=true）
- **exchange**：`deal_price=[$price_941, $close]`；`limit_threshold=[$change_941>=$limit_up, $change<=$limit_down]`（板块分级）
- **model**：`LGBModel`（lr=0.05, max_depth=6, num_leaves=64, lambda_l1=5, lambda_l2=10, num_boost_round=200, early_stopping_rounds=20）

## 5. 结果对比（L1 前 `17444139` → L1 后 `5d935376`）

### 信号 + 回测 metrics

| 指标 | L1 前 | L1 后 | 变化 |
|---|---|---|---|
| best iteration | 62 | **3** | 异常早停 |
| L2 train (best) | 0.9031 | 0.9816 | 退化（几乎没学） |
| L2 valid (best) | 0.9785 | 0.9884 | 退化 |
| IC | +0.0252 | **−0.0127** | 符号反转 |
| Rank IC | +0.0241 | +0.0126 | 仍正但弱 |
| ICIR | +0.2081 | −0.0870 | 反转 |
| 年化超额（含成本）| −88.71% | **+20.01%** | 巨变 |
| IR（含成本）| −4.278 | +0.761 | 反转 |
| max_drawdown | −25.63% | −14.17% | 改善 |
| 年化超额（无成本）| −77.65% | +30.81% | — |
| benchmark 年化（绝对）| N/A | +44.11% | test 段牛市 |

### pred.pkl 对比（决定性）

| | L1 前 | L1 后 |
|---|---|---|
| shape | 6198 | 5628（drop 9.2%） |
| pred std | **0.0827** | **0.0150**（压缩 5.5×） |
| pred range | [−0.82, +0.75] | [−0.20, +0.20] |
| unique 值 | 5370 | 282 |
| 截面 unique/日 | 2.4 | 1.9 |

## 6. 异常观察（客观数据）

1. **训练退化**：best iteration 62 → 3（early_stopping_rounds=20，第 3 轮 valid l2 触底后 20 轮不改善）。
2. **pred 幅度崩塌**：std 0.0827 → 0.0150，range ±0.82 → ±0.195，unique 5370 → 282。直接因果：best iter=3 → 3 棵树 × lr 0.05 → 幅度必然小。
3. **valid l2 升**：0.9785 → 0.9892（模型在 valid 上更差）。
4. **IC 符号反转**：+0.0252 → −0.0127（Pearson 反转，Rank IC 仍 +0.0126）。

## 7. 核心矛盾

**模型明显退化，但超额收益从 −88.71% 翻到 +20.01%。** 模型更差、收益却大幅变好——逻辑矛盾。

判断（基于 pred.pkl 数据）：+20% 超额是**退化模型的副产品**，非 L1 真实信号改善：
- pred 近乎无区分度（std 0.015，282 unique）→ 选股接近随机；
- test 段 benchmark SH000300 年化 +44%（牛市）；
- IC<0 与正超额本身矛盾，结果不可信。

## 8. 诊断结果（2026-07-06 补跑，只读脚本 `/tmp/l1_seg_drop.py`）

补测 train/valid 段 L1 drop（此前 §3.5 只测 test 段）。三段对比（market 时变池口径，与回测 universe 一致）：

| 段 | 总 stock-day | L1 drop | drop% | 防前视正确 (p941 NaN) | 误杀 (collateral) |
|---|---|---|---|---|---|
| train | 47,789 | 4,817 | **10.08%** | 413（0.86%） | 4,404（9.21%） |
| valid | 5,583 | 461 | **8.26%** | 26（0.47%） | 435（7.79%） |
| test  | 6,198 | 570 | **9.20%** | 116（1.87%） | 454（7.32%） |

**关键发现：**

1. **三段 drop% 相近（8–10%）**，并非 train/valid 段过度 drop。§3.5 此前测 test 段 6.246% 是按 flat codes 全集算的口径；按 market 时变池重测三段都在 8–10%，量级一致。**「整组 drop 在 train/valid 段误杀过大」假设不成立。**

2. **防前视正确 drop（p941 NaN）占比极低：train 0.86% / valid 0.47% / test 1.87%**。方案 A（整组 drop）的 **误杀 : 防前视 ≈ 9 : 1**——绝大多数 drop 是「p941 有值但某 feature 边界 NaN」的误杀，真正防住的 9:41 前视样本不到 1/10。

3. **drop 主因是分钟因子缺失**，非 p941 前视。train 段 drop 样本中：仅分钟 NaN 65.4% / 仅日频 NaN 26.3% / 两者皆含 8.3%。SH600519 抽样：2024 分钟 NaN 1.7%、2025/2026 为 0% → 分钟因子缺失集中在早期 2024 年（train 按月：2024-01 drop 20.8%、2024-04 19.8%，其余月份 5–15%）。

4. **best iter=3 退化根因仍未锁定**。drop 比例数据（10% 级、三段一致）无法直接解释 valid l2 为何从 0.9785 升到 0.9892。可能机制（未证实）：drop 移除了 2024 年初高波动期的高信号样本 → 训练集 SNR 下降 → lightgbm 学不到信号。**确认需补 per-iter l2 曲线 + drop 前后 label/feature 分布对比，本次诊断未做。**

## 9. 结论

**L1 防前视机制本身有效**（catch 到 train 0.86% / valid 0.47% / test 1.87% 的 p941 NaN 前视样本），**但方案 A（整组 drop）代价过大**：误杀 : 防前视 ≈ 9 : 1，且误杀样本集中在分钟因子刚上线的 2024 年初高信号期，**可能是 best iter=3 退化的诱因（未确证）**。+20% 超额收益不可信（退化模型副产品，见 §7）。待用户决策方向（见对话）。

## 10. 重跑可复现性确认（2026-07-06 20:06）

按用户要求重跑完整 训练-测试-回测 全流程，验证 `best iter=3` 非随机 fluke。

- **新 recorder**：`61dfe27374f94e86aa41a496c65b8438`
- **log**：`/tmp/backtest_l1_rerun_20260706_200608.log`，exit=0
- **best iteration**：`[3]`（train l2=0.981635 / valid l2=0.988429）—— 与首次 `5d935376` **逐位一致**
- **iter [20]**：train l2=0.950162 / valid l2=0.988924
- **回测**：年化超额（含成本）+20.01% / IR 0.761 / max_drawdown −14.17% / benchmark +44.11% —— 与首次一致

**结论**：`best iter=3` 是**确定性输出**（lightgbm seed 固定 + 同数据 → 同结果），非 fluke。

**第一性原理新洞察（根因方向修正）**：valid l2 从 iter 3（0.988429）到 iter 20（0.988924）**全程 ≥ 0.988 且不改善**，而 train l2 同期从 0.9816 降到 0.9502（模型在 train 上学，valid 不跟随）→ **valid 段信号缺失 + early stopping 在 iter 3 误触底**。真正根因疑点是 **valid 段（2026-01→03）模型零预测力**（valid l2 ≈ label 方差），需深挖 valid 段 label/feature 分布是否与 train 段偏移，而非「误杀样本」单一因素。

## 11. 根因坐实（2026-07-06 20:21）—— label 复权口径 bug（⚠️ 初版单点误判，见 §12）

> ⚠️ **本节基于 SH600608 单点采样，结论「整体不复权」是误判**（SH600608 恰为异常票）。系统验证后已修正，准确根因见 §12。本节保留作诊断痕迹，**勿据此结论决策**。

只读脚本 `/tmp/verify_1min_caliber.py` 直接对比两个 bin 目录的原始 close，**100% 坐实根因**：

**cn_data_1min 的 `$close` 是【不复权价】（factor 恒为 1.0），qlib_data day 的 `$close` 是【后复权价】（factor=累积复权因子）。** CLAUDE.md「day/1min 一致后复权」描述不准 —— 1min 实际无复权信息。

证据（SH600608 2024-07-25）：

| 来源 | close | factor |
|---|---|---|
| day (qlib_data) | **121.87** | 51.86 |
| 1min @09:41 (cn_data_1min) | **2.32** | **1.0** |
| day close / factor | 2.35 | — ≈ 1min close ✅ |

**传导链（第一性）：**

1. `materialize_minute.py:243` `price_941 = c[:,10]` 直接存 1min close 原值（不复权，无 `/factor` 转换）→ `$price_941` bin = **不复权**。
2. label `Ref($close,-1) / $price_941 - 1` = **后复权** close(T+1) / **不复权** p941(T) − 1 → 对高 factor 股爆炸（SH600608 factor=51.86 → label=+44.69）。
3. train 段 94 样本 `|label|>0.5`、20 样本 `>3.0` → l2 (MSE) 被 outlier 主导 → lightgbm 前 3 轮拟合 outlier → valid（outlier 少）不跟随 → early stop iter 3。
4. **factor IC 未崩**（valid mean|IC|=0.0245，62 因子 `|IC|>0.03`）→ 信号在，退化纯是 label-outlier/l2 假象。

**连带 bug（同一口径错配）：**

- **change_941 算错**：`materialize_minute.py:259` `raw_p941 = out_p941 / factor_d` 假设 price_941 后复权；实际不复权 → raw_p941 = 2.32/51.86 = 0.0447（应是 2.32）→ change_941 ≈ 0.0447/昨收 − 1 ≈ **−98%** → exchange limit_threshold buy 表达式 `$change_941 >= $limit_up` **永不成立 → 涨跌停买入拦截失效**。
- **deal_price 跨口径**：`[$price_941(不复权), $close(后复权)]` 撮合价也跨口径。

**结论**：best iter=3 退化、+20% 超额不可信、涨跌停买入拦截失效，**三者共同根因 = 1min close(不复权) 与 day close(后复权) 的口径错配**，物化层 price_941 未做后复权转换。修复方向待用户决策。

## 12. 根因修正（2026-07-06 20:25）—— 系统验证推翻 §11「整体不复权」

用户质疑「写入时本应是后复权」，扩大采样后 §11 单点结论被推翻。

**口径分布**（`/tmp/verify_1min_caliber_full.py`，highbeta 池 5116 票 + 知名 6 票 × 3 日 = 14878 stock-day）：

| 1min close 口径 | 占比 |
|---|---|
| 后复权（正常）| **95.8%** |
| 不复权（异常）| 2.3% |
| 无法判别（factor≈1，后复权≈不复权）| 1.9% |

**cn_data_1min 整体后复权**（用户没记错）；1min `$factor` 非恒 1.0（SH600653 1min_factor=8876.86 ≈ day factor）。§11「整体不复权」是 SH600608 单点误判，**作废**。

**异常票判据**（`/tmp/verify_abnormal_factor.py`，100% 干净）：

| 票 | day_factor | 1min_factor | 1min_close |
|---|---|---|---|
| SH600421 | 1.92 | **1.0** | 不复权 |
| SH600599 | 1.88 | **1.0** | 不复权 |
| SH600608 | 39.20 | **1.0** | 不复权 |
| SH600696 | 2.67 | **1.0** | 不复权 |
| SZ000638 | 4.42 | **1.0** | 不复权 |
| SH600519（对照）| 5.75 | 5.75 | 后复权 |

异常票的 **1min `$factor` 被错填为 1.0**（复权因子丢失）→ 1min close 未做后复权 → 与 day `$close`（后复权）跨口径 → label 爆炸。

**outlier 归属**（`/tmp/verify_label_outlier_source.py`，决定性）：

| 交叉 | 样本数 |
|---|---|
| 口径异常 ∩ label outlier（\|label\|>0.5）| **94 / 94（100%）** |
| 口径正常 但 label outlier | **0** |

label 的 94 个 outlier **全部**落在 3.03% 异常票（1434 stock-day / 1021 票）上。outlier top：SH600421(25)、SH600696(23)、SZ000638(12)、SH600636(10)。

**根因（修正版）**：cn_data_1min 中 ~3% 票的 1min `$factor` 字段错填为 1.0（**数据源 bug**，复权因子丢失）→ 这些票 1min close 不复权 → `$price_941`（materialize 存 1min 原值）与 day `$close`（后复权）跨口径 → label 局部爆炸 → l2 被 outlier 主导 → best iter=3。连带 bug（change_941、deal_price）**仅局限在 3% 异常票**，95.8% 正常票无碍。

**撤回**：§11 基于错误假设的「materialize ×factor」「label 改不复权」方案会破坏 95.8% 正常票（双重校正），**否决**。修正后修复方向见对话。

## 13. D 方案判据选型（2026-07-06 22:00）—— V1→V3→V4 三轮验证

用户选定 **D 方案（物化层校正异常票）**：materialize 时识别异常票，`price_941 *= day_factor` 校正为后复权。核心问题是**判据**：如何精确识别「p941 不复权」的异常票，不误伤 95.8% 正常票。

### 三轮判据对比（train 段 47,317 样本）

| 判据 | 异常样本 | 票数 | 校后 outlier \|>0.5\| | label std | 误校正候选 |
|---|---|---|---|---|---|
| **V1** 绝对距离 \|p941-unadj\|<\|p941-close\| | 3349 (7.08%) | 974 | 4 | 0.067 | 3259（max\|Δ\|=0.50）|
| **V2** 机制型 \|p941×factor-close\|/close<0.10 & factor>1.10 | 206 (0.44%) | 79 | 5 | 0.068 | 117 |
| **V4** V1 + factor>1.5 门槛（终定）| **94 (0.20%)** | **9** | **4** | **0.068** | **4** |

label 原 std=0.2459，原 outlier \|>0.5\|=94。

### V1 过判（否决）

V1 异常 3349 中 **3223 个 factor<1.5**（99.9%）—— 小 factor 票 unadj≈close，绝对距离判定不稳，把日内波动的正常票误判异常。这 3223 样本仅含 **1 个原 label outlier** → V1 过判的根本不是 outlier 源，纯误判。

### V2 漏判（否决）

V2（原名 V3）判据 `p941×factor≈close_t` 被日内波动干扰：SH600421（2024-06-25，p941=5.69 不复权，factor=1.92）9:41→收盘日内涨 13%，使 `p941×factor=10.92` 偏离 close_t=12.37 达 11.7%>10% 阈值 → V2 漏判 → label 留在爆炸值 1.26。

### V4 终定（V1 距离判定 + factor>1.5 门槛）

```
unadj = close_t / factor
abnormal ⟺ |p941 - unadj| < |p941 - close_t|  AND  factor > 1.5
price_941_calibrated = price_941 * factor   # 仅 abnormal
```

**原理（第一性）**：异常票 p941 是不复权 raw 价 ≈ close_t/factor；正常票 p941 是后复权 ≈ close_t。factor>1.5 时 unadj 远离 close_t，p941 只能靠近其一 → V1 距离判定干净；factor<1.5 时 unadj≈close_t 判定不稳，但此类票 ×factor 校正影响小（<1.5 倍），即使漏判也不致 label 爆炸，门槛排除以杜绝误判。

**9 票清单**（V4 捕捉，94 train stock-day）：

| 票 | stock-day | factor | 类型 |
|---|---|---|---|
| SH600421 | 25 | 1.92 | 已知异常 |
| SH600599 | 5 | 1.85-1.88 | 已知异常 |
| SH600608 | 1 | 51.86 | 已知异常 |
| SH600636 | 10 | 7.41-7.72 | 新发现 |
| SH600696 | 23 | 2.67-2.71 | 已知异常 |
| SH605081 | 11 | 1.52-1.55 | 新发现 |
| SH688287 | 6 | 1.59-1.61 | 新发现 |
| SZ000638 | 12 | 4.42 | 已知异常 |
| SZ300106 | 1 | 1.58 | 新发现 |

已知 5 票全部覆盖，无漏判。

### V4 校正效果（决定性）

| 段 | 校正前 outlier \|>0.5\| | 校正后 | label std 前→后 |
|---|---|---|---|
| train 2024-2025 | 94 | **4**（残余全真实噪声）| 0.2459 → **0.0679** |
| valid 2026-Q1 | 5 | **0** | 0.0937 → 0.0641 |

train 残余 4 个 \|label\|>0.5 全是**真实噪声**（非口径）：SZ300147/SZ300339（高贝塔连涨 +24%~27%）、SZ301551/SZ301618（2024 国庆后创业板新股暴跌 −54%~64%，factor=1.0 无除权）。label std 从 0.246 压到 0.068（3.6×），l2 不再被 outlier 主导。

**误校正候选仅 4 个**（异常∩原label正常）：3 个 SH605081 经 p941/unadj∈[1.01,1.11] 核实为真异常（校正正确，原 label 0.43 跨口径 → 校正后 -0.07 真实）；1 个 SZ300106（p941/unadj=1.25）边界误判，factor=1.58 影响可控。

### 连带效果

- **change_941 自洽**：校正后 out_p941 后复权，`raw_p941 = out_p941/factor_d` 对异常票 = p941_raw×factor/factor = raw（不复权），change_941 = raw/昨收-1 正确。涨跌停买入拦截恢复。
- **deal_price 同口径**：`[$price_941, $close]` 校正后都后复权。
- **14 分钟因子不受影响**：全是 close/open/high/low/volume 比率，异常票自洽。
- **无需读 1min factor 字段**：判据完全基于 day close/factor + price_941。

### 实现要点（待用户确认）

`materialize_minute.py` 在 scatter out_p941 后、算 change_941 前，加 V4 校正（~5 行，close_d/factor_d 已在作用域 line 138）：

```python
unadj_d = close_d / factor_d
abnormal = (np.abs(out_p941 - unadj_d) < np.abs(out_p941 - close_d)) & (factor_d > 1.5)
out_p941[abnormal] = out_p941[abnormal] * factor_d[abnormal]
```

重物化 `price_941.day.bin` + `change_941.day.bin`（deal_price 引用这俩），重跑 qrun 全链路验证 best iter 回升。

## 14. v3：剔除慢因子（rolling windows [5,10]）+ 噪声签名坐实（2026-07-06 23:43）

> **背景**：§9 结论 + 三任务诊断（best iter=3 / 逐因子 IC / 慢因子审计）后，用户决策：
> - 任务1（best iter=3 诊断）：**真过拟合 + valid 零泛化**（train 200 轮 shaving 21.5%，valid 仅 shaving 0.03%，valid 全程 ≥0.988 不改善）。
> - 任务2（反向因子）：**不处理**（172 因子无一 ICIR<-0.3，最强反向 $vol_vs_yest 仅 -0.287；GBDT 原生处理 split 方向）。
> - 任务3（>10 日慢因子）：**改 Alpha158 默认 windows 参数**（从源头不生成）。

### 14.1 慢因子审计（任务3 输入）

逐因子窗口分类（172 因子，train 段 DK_R）：

| 类别 | 数量 | window 分布 |
|---|---|---|
| 日频慢（window>10，剔除候选） | **87** | w=20:29, w=30:29, w=60:29 |
| 日频快（window≤10） | 71 | w=1:9 kbar + w=2:4 price + w=5:29 + w=10:29 |
| 分钟（≤10min） | 14 | startup/accel/close_pos/vol_ratio/vol_vs_yest |

87 慢因子 IC 表现：ICIR mean=-0.008，**无一 |ICIR|>0.3**（最强 VMA60 仅 0.213）→ 剔除几乎不损失信号。

### 14.2 实施：highbeta_handler.py v3

`get_feature_config` 改走 `Alpha158DL.get_feature_config(conf)` 原生 config 扩展点，`conf["rolling"]={"windows":[5,10]}`（其余键与 `Alpha158` 同：kbar + price windows=[0] OPEN/HIGH/LOW/VWAP），从源头不生成 20/30/60。仍套 `Ref(...,1)` lag + 14 分钟因子。

验证（`HighBetaAlpha158.__new__` 绕过 `__init__` 数据加载）：**85 因子 = 71 日频快 + 14 分钟，0 慢**。窗口分布 w={1:9, 2:4, 5:29, 10:29}。前两字段 `Ref(($close-$open)/$open,1)`、`Ref(($high-$low)/$open,1)` lag 正确；分钟字段 `$startup_mom_*` 无 lag。

### 14.3 v2 vs v3 对比（同 test 窗 2026-04-01→07-02，同 benchmark SH000300 +44.11%）

recorder：v2=`61dfe273`（V4 校正后）/ v3=`2f7f72ae`（v3 handler）。

| 指标 | v2 (172) | v3 (85) | 解读 |
|---|---|---|---|
| best iter | 3 | 4 | 仍近退化 |
| train l2@best | 0.9816 | 0.9786 | — |
| **valid l2@best** | **0.9884** | **0.9887** | **≈持平 ≈ label 方差 ≈ 1.0，模型零泛化未改** |
| IC | -0.0127 | **+0.0221** | 反转正向 |
| Rank IC | +0.0126 | **+0.0260** | **+106%（截面排序信号翻倍）** |
| Rank ICIR | — | +0.2016 | — |
| 超额(不含成本)年化 | +30.81% | +4.76% | -26pp |
| **超额(含成本)年化** | **+20.01%** | **-6.98%** | **-27pp** |
| IR(含成本) | 0.761 | -0.313 | -1.07 |
| 日 std | 1.70% | **1.44%** | **持仓更稳** |
| maxDD(含成本) | -14.17% | **-9.65%** | **改善** |

### 14.4 噪声签名坐实（核心结论）

**IC 升、回测降、valid l2 不动** —— 三者解耦是 near-degenerate 模型的噪声签名：

1. **valid l2 ≈ 0.989 ≈ label 方差（CSZScoreNorm 后≈1.0）**：模型预测 ≈ 截面均值 + 微扰，best iter 仍 3-4，零泛化未改。
2. **IC/Rank IC 改善**（Rank IC +0.0126→+0.0260）：IC 是全截面平均指标，反映微弱真实信号，更稳定。v3 因子更干净，截面信号确实略好。
3. **回测大幅回退**（含成本 +20%→-7%）：top-20 是尾部选择，对微扰放大。near-degenerate 模型的 top-20 由噪声主导，因子集小改 → 不同 top-20 → 巨大回测方差。
4. **日 std 降 + maxDD 改善**：v3 持仓更分散/稳定，是结构性正向（非噪声）。

**→ v2 的 +20% 含成本超额不是真实 alpha，是 near-degenerate 模型在该 61 天 test 窗的幸运命中。** v3 把因子洗干净后（IC 更好、持仓更稳），回测反而暴露出噪声本质。**两个回测数都不可作为策略有效性判据。**

### 14.5 真正瓶颈（未解决）

**label SNR 过低**：172 因子 max |ICIR|=0.287、|IC| 均值 0.0215。label `Ref($close,-1)/$price_941-1`（T+1 收盘/T 日 9:41 价 -1）从可用特征几乎不可预测。**因子清洗（降维 172→85）是卫生 + 减过拟合面，不是 SNR 问题的解。** best iter=3-4 / valid l2≈方差 在 v2/v3 都成立，根因是 label-特征 SNR，与因子数无关。

### 14.6 决策与下一步（待用户）

- **v3 handler 保留**（因子更干净、IC 更好、持仓更稳是实打实的卫生收益；回测噪声不能作为「v3 比 v2 差」的理由，因 v2 的 +20% 同样是噪声）。
- **不在因子层继续打转**（已证 SNR 是瓶颈，再加/减因子不解决 valid l2≈方差）。
- 真正的杠杆在 **label 设计**：更长 horizon（如 T+5）、rank label、或换目标（如截面排名分位）以提 SNR —— 需用户定方向后再设计（brainstorming → spec）。

---

## §15 minute-only 实验（2026-07-07）：砍全部日频因子 → best iter 3→14，反转 §14「因子层无用」结论

> recorder: `3038d4c14db2460089b6bfe11869f9da`（experiment `minute_only`，handler `MinuteOnlyHandler`）。
> 用户指示：「尝试仅使用 1min 相关因子做训练回测」。

### 15.1 实验设计

新增 [MinuteOnlyHandler](../../qlib_ifind_beta/minute_only_handler.py)（`HighBetaAlpha158` 子类，仅 override `get_feature_config` 丢 71 日频、只返 14 `$minute_field`；L1 shared `DropnaProcessor(feature)` 护栏从父类继承）。配套 [workflow_minute_only.yaml](../../qrun/workflow_minute_only.yaml)（label / deal_price / 涨跌停 / 模型超参 / 切分全同 workflow.yaml）→ **唯一变量是 feature 85→14**。10/10 单测通过，无 ripple。

### 15.2 三版对比（同 test 窗 2026-04-01→07-01，~61 交易日，benchmark +44.11%）

| 指标 | v2 (158+14=172) | v3 (71+14=85) | **m14 (0+14=14)** |
|---|---|---|---|
| best iter | 3 | 4 | **14** |
| train l2@best | 0.9816 | 0.9786 | 0.9673 |
| valid l2@best | 0.9884 | 0.9887 | 0.9886 |
| IC | −0.0127 | +0.0221 | **+0.0341** |
| Rank IC | +0.0126 | +0.0260 | **+0.0329** |
| Rank ICIR | — | +0.2016 | **+0.2961** |
| 超额(不含成本)年化 | +30.81% | +4.76% | **+40.24%** |
| **超额(含成本)年化** | +20.01% | −6.98% | **+28.39%** |
| IR(含成本) | 0.761 | −0.313 | **1.251** |
| 日 std | 1.70% | 1.44% | 1.47% |
| maxDD(含成本) | −14.17% | −9.65% | **−7.15%** |

四个独立指标同向 monotone 改善：best iter(3→4→14)、IC、Rank IC、含成本 alpha（v2→v3 回测退是噪声暴露，见 §14.4；m14 重新上升且超 v2）。

### 15.3 反转证据：best iter=14 + 同向耦合

- **best iter=14 是最硬信号**：early_stopping_rounds=20，m14 前 13 轮 valid 持续改善才触顶。反证「因子越少越慢过拟合」——v3(85) 比 v2(172) 因子少半，best iter 仅 3→4；驱动 best iter 的是**因子-label 对齐度**而非因子数。14 因子单位信息量更高。
- **同向耦合 = 真信号（对比 §14.4 v2 噪声解耦）**：v2 = IC↑+回测↓+valid l2 平坦（解耦 = 噪声，+20% 是幸运命中）；m14 = IC↑+回测↑+best iter↑（同向耦合 = 信号）。m14 的 +28% 不是 v2 式噪声。

### 15.4 根因（第一性原理）：label 时间尺度对齐

label = `Ref($close,-1)/$price_941-1` 衡量「9:41 买入持有到 T+1 收盘」的 **~1.5 天短期收益**。9:30-9:40 分钟因子（startup_mom / accel / close_pos / vol_ratio / vol_vs_yest）与此短期 label **时间尺度天然对齐** → 同周期信号。Alpha158 日频因子（即便 v3 砍到 windows=[5,10] 的 5/10 日均线/动量/波动率）预测周-月中期收益，与 ~1.5 天 label **尺度错配 → 噪声而非信号**。monotone 改善链 v2→v3→m14 由此而来。

### 15.5 对 §14.5/§14.6 + technical-design §D6 的修正

- §14.5「真正瓶颈是 label SNR，因子清洗不是解」**部分修正**：SNR 瓶颈**部分来自日频因子的尺度错配**，非「标签完全不可预测」。m14 Rank ICIR=0.296 略破 172 因子的 0.287 上限，best iter 14 + 回测同向证明因子侧仍有空间。
- §14.6「不在因子层继续打转」**修正**：因子层仍有杠杆，但方向是**时间尺度对齐**（砍错配因子），不是窗口长度调整（v2→v3 砍慢因子只让 best iter 3→4）。
- **未修正部分**：valid l2 仍 0.989 ≈ 标签方差 → 绝对预测能力未改，提升的是排序方向；label 方差中仅 IC²≈0.1% 可被解释，**天花板仍受限于 label 本身可预测性**。

### 15.6 诚实 caveats

1. **test 仅 3 个月**（~61 交易日），+28% 年化为外推；test 期大牛市（benchmark +44%），alpha 需熊市/震荡市验证。
2. **valid l2 不变** → 绝对预测能力未提升，仅排序方向变好。
3. 14 因子是「9:41 高贝塔策略」原生时间尺度，**结论不可外推到中长期策略**。
4. 换手仍高（topk=20/n_drop=5），成本吃 12pp（不含成本 +40% → 含成本 +28%）。

### 15.7 决策（用户 2026-07-07 选定）

- **仅固化结果**（本 §15 + Get笔记），**不切主线 workflow.yaml**（test 太短，单窗口不足以下结论）。
- `MinuteOnlyHandler` + `workflow_minute_only.yaml` + `test_minute_only_handler.py` 保留供复查。
- 待用户定后续方向：扩展 test 验证稳健性 / 切主线 / label 设计。

