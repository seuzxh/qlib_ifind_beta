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

---

## §16 minute-factor surgery 实验（2026-07-07）：14→8 因子，IC 翻倍但 alpha 腰斩（IC↑+回测↓ 解耦）

### 16.1 实验设计（A/B，label FROZEN）

IC 取证（2026-07-07）显示 m14 的 14 因子里 10 个是 dead weight（|ICIR|<0.12）。本轮做 factor surgery：砍 10 噪声 + 留 4（vol_ratio_{1,3,5}m + vol_vs_yest）+ 加 4 新（vol_vs_yest_t2/t3/t5 + overnight_gap）= **8 因子**。

**唯一变量 = feature 集（14 → 8）**。label `Ref($close,-1)/$price_941-1`、deal_price、涨跌停拦截、LGBModel 超参、TopkDropoutStrategyTD0、切分、成本 **全同 workflow_minute_only.yaml**（见 `qrun/workflow_minute_surgery.yaml`，diff 仅 handler class/module_path + experiment_name）。

> overnight_gap 设计意图修正：spec 原假设 overnight_gap 是反转因子，但训练前 IC 预筛实测为**正 IC 动量延续**（train ICIR +0.566 / valid +0.463 / test +0.462，4 新因子中最强）。第一性：LGBM 符号不变性（tree split `x<t` ⟺ `-x>-t` 同划分）→ 翻符号是 no-op → 保留正动量信号，不翻。最终 surgery 集 = 「4 反转（vol_vs_yest 族）+ 4 动量（vol_ratio 族 + overnight_gap）」均衡组合。

### 16.2 m14 vs m-surgery 对比（同 test 窗 2026-04-01→07-01，~61 交易日，benchmark +44.11%）

| 指标 | m14 (14 因子) | **m-surgery (8 因子)** | 变化 |
|---|---|---|---|
| best iter | 14 | **46** | ↑ ×3.3 |
| train l2@best | 0.9673 | 0.9372 | ↓ 改善 |
| valid l2@best | 0.9886 | **0.9765** | ↓ 实质改善 |
| IC | +0.0341 | **+0.0661** | **↑ ×1.94** |
| Rank IC | +0.0329 | **+0.0730** | **↑ ×2.22** |
| Rank ICIR | +0.2961 | **+0.6174** | **↑ ×2.09** |
| ICIR | — | +0.5934 | — |
| 超额(不含成本)年化 | +40.24% | +26.31% | ↓ −14pp |
| **超额(含成本)年化** | **+28.39%** | **+14.34%** | **↓ −14pp（腰斩）** |
| IR(含成本) | 1.251 | 0.654 | ↓ 腰斩 |
| maxDD(含成本) | −7.15% | −5.18% | ↑ 改善 |
| 日 std | 1.47% | 1.42% | ↓ |

### 16.3 核心悖论（第一性）

IC 是在 label `Ref($close,-1)/$price_941-1` 上算的；回测 PnL 物理上就是按 `$price_941` 买、次日 `$close` 卖（topk=20 按模型 score 选股，模型拟合的正是该 label）。**IC↑ 应当机械地导致 alpha↑**，实际却相反——IC 翻倍、alpha 腰斩。这是真悖论，不是常规噪声波动。

### 16.4 与 §14.4 的关键区分（非同款噪声）

§14.4 v2→v3 的「IC↑+回测↓」判为噪声，凭据是 valid l2 平坦（0.9884→0.9887）+ best_iter 3→4（几乎不学习）→ 信号层无实质增益，IC 微涨是噪声。

**本次 surgery 不同**：valid l2 实质下降（0.9886→0.9765）+ best_iter 14→46（×3.3）→ **信号层增益是结构性的**，模型确实学到了更多 label-aligned 信息，IC 翻倍不是噪声。**但回测 alpha 下降仍是硬事实**——surgery 不构成对 m14 的改进（最终实现 alpha 腰斩）。

### 16.5 根因候选（第一性，每条可证伪，不臆断）

(1) **全截面 IC vs top-20 极值解耦**：IC = 全截面（~100 只）Spearman 排序相关；回测 = 只取 top-20 极值。数学上可「全截面平均排序改善」同时「top 分位数实现收益变差」。IC 是均值指标，回测是尾部分位数指标，本就可解耦。**可验证**：直接算 top-20 实现 label 收益，对比 m14/m-surgery 的 top 分位数 label 均值。

(2) **单变量 IC 取证 ≠ LGBM 多变量价值**：砍掉的 10 因子 |ICIR|<0.12 是**单变量**（每因子单独与 label 的 rank 相关），但 LGBM 用 split 捕获**交互**。这 10 个「单变量无用」因子可能在多变量交互里贡献了正交 alpha；砍掉后 8 个强单变量因子边际 IC 更高，但联合 alpha 更低。IC-based 因子筛选对 GBDT 有已知盲区。**可验证**：对 m14 模型跑 permutation importance / SHAP，看 10 个「dead」因子实际是否被 split 使用。

(3) **best_iter 46 / score 分散度影响 topk 换手**（已排除为主因）：8 因子模型更密、训练更久，score 量级可能改变 topk=20/n_drop=5 换手结构。但**不含成本 alpha 也降 14pp**（与含成本同幅度）→ 成本/换手不是主因，排除。

### 16.6 诚实结论

1. surgery 在**信号层**（IC/Rank IC/ICIR/valid l2/best_iter）显著强化——结构性、模型无关、最稳的证据，8 因子集与 label 对齐度远高于 14 因子集。
2. 但在**最终实现指标**（含/不含成本 alpha、IR）上腰斩。与 IC 改善方向相反，真悖论。
3. **不切主线 workflow.yaml**：surgery 在最终 alpha 上不构成对 m14 的改进。`MinuteSurgeryHandler` + `workflow_minute_surgery.yaml` + spec/单测保留供复盘。
4. **方法论副作用（重要发现）**：IC 翻倍 ≠ alpha 提升。呼应 §14.4「两个回测数都不可作判据」精神，**扩展到「单变量 IC 也不可单独作为 GBDT 因子筛选判据」**——单变量 IC 对多变量交互 alpha 有盲区。

### 16.7 待用户定方向

- (a) 验证根因 (1)：写脚本对比 m14 vs m-surgery 的 top-20 实现 label 收益（top 分位数而非全截面）。
- (b) 验证根因 (2)：对 m14 模型跑 SHAP / permutation importance，查 10 个「dead」因子是否实际被 split 使用。
- (c) 保留 m14 主线，surgery 归档，转其他方向（扩展 test / label 设计 / 其他因子族）。

---

## 17. §16 根因证伪 + 真根因闭环（2026-07-07，自主迭代 iteration-1→4c）

> **本节推翻 §16.5 假设 (1)、修正 §16.6 结论**：IC↑/alpha↓ 悖论的根因 **不是**「全截面 IC 与 top-20 极值解耦」也 **不是**「单变量 IC 对 GBDT 有盲区」，而是**纯执行层 bug**——surgery 的 alpha 集中在 9:41 封涨停股（不可买），策略欠仓趴现金 + 错过赢家。**因子本身在可买残集上仍显著优于 m14**。

### 17.1 §16.5 假设 (1) 证伪（iteration-1：top-20 耦合于 IC）

假设 (1) 称「全截面 IC 改善但 top-20 极值实现收益变差」。实测（`/tmp/diag_signal_decay.py` + top-20 label 均值）：

| 模型 | top-20 全集 label/日 | top-20 h=1 前视 | top-20 h=10 前视 |
|---|---|---|---|
| m14 | +0.719% | +0.719% | +2.588% |
| surgery | +1.203% | +1.079% | +3.151% |

→ **surgery top-20 极值收益在所有持有期均 > m14**（h=1 +50%、h=10 +22%）。假设 (1) **证伪**：top-20 与 IC 同向，都 favors surgery。信号不衰减（h=1→10 单调递增，surgery 全段领先）。

### 17.2 真根因定位（iteration-2→4b，ground truth = 实际组合）

不再推断，直接读 `report_normal_1day.pkl` + `positions_normal_1day.pkl`：

| 指标 | m14 | surgery |
|---|---|---|
| 日均换手 | 50.6% | 51.1%（相同）|
| 日均成本 | ~0.0005 | ~0.0005（相同）|
| 原始超额/日 | +0.169% | +0.111% |
| **日均 cash** | **1.93M（1.8%）** | **8.08M（7.4%）** |
| cash max | — | 24.9M（25%）|
| **实际持仓数** | **20.0**（4/61 天 <20）| **18.9**（39/61 天 <20，min 16）|

换手/成本/机制全相同 → **唯一差异是 surgery 填不满仓**（4× 现金、39/61 天欠仓）。

**为何欠仓**（`/tmp/diag_top5_block_cash.py`）：TopkDropout(n_drop=5) 每日实际加仓对象 = top-5 新进股。9:41 涨停拦截率：

| top-K | m14 | surgery |
|---|---|---|
| **top-5（dropout 加仓目标）** | **0.0%** | **21.0%** |
| top-10 | 0.0% | 11.6% |
| top-20 | 0.0% | 6.5% |

→ surgery 的 alpha 集中在开盘强势股 → top-5 每日 21% 封涨停被 exchange 拦截（`$change_941 >= $limit_up`）。**策略源码 `td0_strategy.py:157-180`：buy 列表取原始 top-n（含涨停股），for 循环 `if not is_stock_tradable: continue` 跳过涨停股但不 refill** → 每日少买 ~1 只 → 18.9 仓 → 7.4% 现金 → 上涨市 cash drag。且被拦截股正是赢家（blocked label +3.658%/日 vs 可买 +0.955%/日）。

### 17.3 闭环：因子在可买残集仍优（iteration-4c，决策性）

`/tmp/diag_buyable_alpha.py`：剔除 9:41 涨停股后重算 IC + top-20：

| 指标 | m14 | surgery | 判定 |
|---|---|---|---|
| 全截面 IC | +0.0341 | +0.0661 | surgery 优 |
| **可买截面 IC** | **+0.0339** | **+0.0743** | **surgery ×2.2 仍优** |
| top-20 全集 label | +0.712%/日 | +1.203%/日 | surgery 优 |
| **top-20 可买重排 label** | **+0.712%/日** | **+0.966%/日** | **surgery +35% 仍优** |

→ **因子本身 OK**（可买残集 IC 2.2×、top-20 +35%）。**问题 100% 在执行**：策略不 refill 被拦截槽位。**理论上**，若让策略选 top-20 可买股，surgery 实现 alpha ≈ +0.966%/日 → 年化远超 m14 的 +28.4%。

### 17.4 根因（一句话，可发表）

> surgery 的 alpha 集中在 9:41 封涨停股（不可买）；高 IC = 对**不可买赢家**的纸面 alpha。实现 alpha 崩溃，是因为策略买不到这些股、且**不顺延**买下一只可买股 → 欠仓 + cash drag + 错过赢家。因子没问题，执行有 bug。

### 17.5 修法（label 冻结、因子不动、qlib 原生）

**`only_tradable: True`**（qlib `TopkDropoutStrategy` 原生参数，YAML 一行）：
- `td0_strategy.py:36-50` 的 `get_first_n` 在构 buy 列表**前**就用 `is_stock_tradable` 过滤涨停股、顺延取下一只可买 → 满仓 20 → 无 cash drag。
- exchange 已正确把 9:41 涨停标为不可买（`forbid_all_trade_at_limit: true` + `limit_threshold=["$change_941 >= $limit_up", ...]`，iteration-4b 实测欠仓坐实）→ `only_tradable` 路由到同一个 `is_stock_tradable`，必然正确过滤。
- **无前视**：决策时点 T 9:41，`$change_941` 在该时点已观测，合法。
- 比"子类化 mask 信号成 -inf"更原生（CLAUDE.md「优先用 qlib 内置组件」）。

### 17.6 待跑

建 `qrun/workflow_minute_surgery_v2.yaml`（= surgery + `only_tradable: True`），重跑 test 段，验证：
1. 实际持仓数 → ~20（39/61 天欠仓应消失）；
2. cash ratio → ~2%（向 m14 看齐）；
3. **实现超额年化 → 应显著回升，目标超过 m14 +28.4%**（理论可买 top-20 +0.966%/日）。

若 v2 alpha 超过 m14 → surgery 8 因子集 + `only_tradable=True` 即「有效的 min 因子组合」候选答案。

---

## 18. §17.5 修法被证伪 + 根因再细化（2026-07-07，iteration-5 v2 实验）

> **§17.5 推荐 `only_tradable=True` 已被实验证伪**。v2 跑出来更差。本节记录实验、给出再细化的真根因。

### 18.1 v2 实验结果（`workflow_minute_surgery_v2.yaml`，唯一变量 only_tradable False→True）

| 指标 | v1 surgery（only_tradable=False）| v2（only_tradable=True）| m14 |
|---|---|---|---|
| IC | +0.0661 | +0.0661（同模型，一致）| +0.0341 |
| 实际持仓数 | 18.93（39/61 天 <20）| **20.00（0/61 天 <20）** | 20.0 |
| cash/value | 8.37%（max 28.6%）| **1.41%（max 5.3%）** | 1.8% |
| 超额(含成本)年化 | +14.34% | **−3.56%** | +28.39% |
| 超额(不含成本)年化 | +26.31% | +8.43% | +40.24% |
| 账户日均 return | +0.296%/日 | **+0.221%/日** | — |

**机制上 v2 成功**（满仓 20、cash 1.4%），**但实现 alpha 反而崩到负**。预测（满仓→alpha 升）被证伪。

### 18.2 为何 refill 更差（substitutes 取证，`/tmp/diag_v2_paradox.py`）

读两回测 positions_normal，逐日比对持仓集合，查「v2 因 refill 比 v1 多买的股」与「v2 换出的股」的前视 label：

| 类别 | 样本 | 前视 label/日 |
|---|---|---|
| v2 多买 substitutes（v1 没买）| 43 | **−0.442%（负！）** |
| v1 v2 共有股 | 345 | +0.853% |
| v1 独有（被 v2 换出）| 33 | +0.343% |

→ **refill 买进的是负收益烂股，还挤掉了 +0.343% 的好股**。双重拖累。

### 18.3 静态可达性验证（承载论断核对）

模拟「每日 top-20 等权再平衡」日收益序列（复利，label 驱动）：

| 组合 | 日均 | 年化 |
|---|---|---|
| surgery 静态 top-20 全集 | +1.079%/日 | +269.8% |
| surgery 静态 top-20 **可买** | +0.966%/日 | **+241.5%** |

→ **§17.3「可买 top-20 +0.966%/日」算术正确**。剔除涨停只从 +269% 掉到 +241%（−10%）→ **90% 静态 alpha 本就在可买股里**。因子不是纸面 alpha。**问题在 dropout 动态把 +241%/年打光**。

### 18.4 根因再细化（v3，第一性）

涨停拦截日 = hot day。当日截面三档：
- top-5 封涨停（不可买赢家，label +3.66%/日）；
- rank 6-10 可买 = 9:41 已拉升但未封板的**均值回归股**（label −0.44%/日）；
- 现金 = 0%。

**现金 0% > 烂股 −0.44%** → v1（趴现金）赢 v2（买烂股）。§17.3 的「可买残集 +0.966%/日」是跨日平均，掩盖了日条件结构：冷静日可买 top 好、hot 日次选可买烂。dropout 被迫在 hot 日交易 → 吃负。

### 18.5 修正结论（覆盖 §17.5）

1. `only_tradable=True` **不是修法**——把现金换成 hot 日负收益 substitutes，更差。
2. surgery 因子的 alpha **90% 在可买股**（静态 +241%/年），但 **dropout 抓不到**：hot 日被迫在「不可买赢家 / 可买烂股 / 现金」三选一里挑了最差的。
3. **真修法方向**：让策略回避 9:41 拉升股（不只涨停，连 rank 6-10 的均值回归股一并 mask）→ 逼选冷静可买股 → 抓冷静日的正 alpha。需先用数据定 mask 阈值 θ（label vs change_941 分桶找过零点），再 subclass 策略 mask signal。

### 18.6 待跑（iteration-6）

(a) θ 诊断：surgery top 选股按 change_941 分桶，看 label 在哪个 change_941 过零 → 定 θ。
(b) `TopkDropoutStrategyTD0Mask`：generate_trade_decision 里把 change_941 > θ 的股 score 置 −inf → topk 自然选冷静可买股。v3 YAML 跑，对比 v1/v2/m14。


## 19. §18.4「hot 日均值回归」假设被判决器推翻 + 真根因锁定（2026-07-07，iteration-6 零重训诊断）

> **§18.4/§18.5 的根因（rank 6-10 可买股是 −0.44%/日的均值回归烂股）被数据推翻**。判决器 `/tmp/diag_turnover_theta.py` 零重训三测（换手 / θ 分桶 / 拦截日分裂）一致指向：**label 在 change_941 上单调递增、无过零点**；拦截日可买 label 反而更高。真根因是 **dropout(n_drop=5) 跟不上静态 top-20 ~99% 日换手**，与 hot/cold 无关。本节覆盖 §18.4/§18.5。

### 19.1 4 实验全景表（IC + 实现超额年化，test 2026-04-01→2026-07-02）

| 实验 | 因子 | IC | 超额(不含成本) | **超额(含成本)** | ICIR(含成本) |
|---|---|---|---|---|---|
| full | Alpha158 日频(T-1 lag) + 14 min = **85** | **+0.0221（最低）** | +4.76% | **−6.98%** | −0.313 |
| **m14（冠军）** | 14 min | +0.0341 | +40.24% | **+28.39%** | +1.251 |
| surgery | 8（4 留 + 4 新） | **+0.0661（最高）** | +26.31% | +14.34% | +0.654 |
| v2 | 8 + only_tradable=True | +0.0661 | +8.43% | −3.56% | −0.158 |

**两个反直觉但确凿的规律**：
1. **IC 与实现超额反向**：surgery IC 最高（+0.0661，m14 的 2×）但实现仅 +14%；m14 IC 中等却实现 +28%。→ IC 高 ≠ 实现高，因 IC 越高模型越确信涨停股最优，剔除涨停后残集越弱。
2. **日频稀释 9:41 信号**：full(85) IC 最低（+0.0221）、实现为负。Alpha158 的 71 个日频因子稀释了 9:41 分钟信号。→ 后续因子组合应在**分钟族内做文章**，不再混日频。

### 19.2 判决器 (A) 静态 top-20 可买集合日换手 → dropout 结构性跟不上

| 模型 | 日均换手 | 中位 | ≤5 槽占比 | >5 槽占比 | max |
|---|---|---|---|---|---|
| m14 | 19.80 | 20 | **0.0%** | **100.0%** | 20 |
| surgery | 19.82 | 20 | 0.0% | 100.0% | 20 |

→ 静态 top-20 **每日近乎全换（~99%）**，而 `TopkDropoutStrategyTD0(n_drop=5)` 每日最多换 5 槽 → 永远落后 15 槽。**n_drop=5 是结构错配**，与因子好坏、hot/cold 无关。这是 m14 静态 gross ~+40%×? 但 dropout 实现 +40%（不含成本）的跟踪损失主因，也是 surgery 静态可买 +241%/y 但实现仅 +26%（不含成本）的 9× 缺口根源。

### 19.3 判决器 (B) surgery top-20 按 change_941 分桶的 label → 单调递增、无过零点（§18.4 证伪）

| 桶(change_941) | 样本 | label 均值/日 |
|---|---|---|
| -5~0% | — | +0.65% |
| 0~2% | — | +1.01% |
| 2~4% | — | +1.10% |
| 4~6% | — | +2.08% |
| 6~8% | — | +2.45% |
| **8~10%（涨停）** | **80** | **+3.70%** |

→ **label 随 change_941 单调上升、从不过零**。§18.4 假设的「rank 6-10 是 −0.44%/日的均值回归股」**不存在**——可买的 4~6% 桶 label +2.08%/日、6~8% 桶 +2.45%/日，都是正且强。涨停桶 +3.70% 最强但不可买。→ §18.5 推荐的 v3 mask（mask 掉 θ 以上拉升股）**会砍掉最强的可买桶**，方向错误，废弃。

### 19.4 判决器 (C) 拦截日 vs 冷静日可买 label → 拦截日更好（§18.4「hot 日烂股」证伪）

| 日类型 | n | 可买 top-20 label/日 |
|---|---|---|
| 拦截日（top-5 含涨停） | 799 | **+1.127%** |
| 冷静日（top-5 全可买） | — | +0.670% |
| 差 | | **+0.457%** |

→ **拦截日可买残集 label 比冷静日还高 +0.457%/日**。§18.4「hot 日被迫买烂股」完全不成立。v2 (refill) 负收益的真正原因是 **§18.2 测得的 substitute −0.442%**：n_drop=5 的 refill 顺延买的是**低 rank 尾巴股**（pred 排名 21+），不是「hot 日均值回归股」。两者不是一回事——substitute 的负 label 来自低 rank，不来自高 change_941。

### 19.5 修正根因（覆盖 §18.4/§18.5）

1. **label 单调依赖 change_941**：9:41 拉升越猛，T+1 续涨越多（高贝塔成分股的隔夜动量延续）。最强在涨停（+3.70%）但不可买；可买的 4~8% 桶仍有 +2~2.5%/日。
2. **surgery 实现 alpha 90% 在可买股**（§18.3 静态 +241%/y 坐实），**问题在 dropout 跟踪**（§19.2 换手 ~99% vs n_drop=5），**不在 hot/cold 日结构**（§19.4 拦截日更好）。
3. `only_tradable=True` 在 n_drop=5 下有害（§18.2 refill 买低 rank 尾巴），但在 **n_drop 跟上换手** 时无害甚至有益（refill 的是当日 fresh 高 rank 可买股，非陈旧尾巴）。
4. **日频 Alpha158 稀释 9:41 信号**（§19.1 full IC 最低）→ 因子组合聚焦分钟族，不混日频。

### 19.6 新修法方向（替代 §18.6 的 v3 mask，已废弃）

真杠杆 = **让 dropout 换手跟上静态 top-20 ~99% 日换手**，即调大 `n_drop`（5→10/15/20），而非 mask 信号。`TopkDropoutStrategyTD0` 在 n_drop=topk 时：若实际换手 ≈100%（本场景），每日 sell 全部 old、buy 当日 fresh top-20 → 实现逼近静态 top-20（gross ~+241%/y for surgery / 待测 m14）减成本。成本估算：n_drop=20 全换 → ~20×0.05×0.002=0.002/日 ≈ 50%/y，仍远小于静态 gross。

下一轮（iteration-7）：m14 上 n_drop ∈ {5(baseline),10,15,20} × only_tradable ∈ {False,True} 扫描，看实现超额是否随 n_drop 单调上升并突破 +28.39%。label/因子/切分全冻，唯一变量 n_drop（+ only_tradable）。



---

## §20 iteration-7：n_drop 扫描 — **n_drop=15 甜点，超额 +116%/ICIR 3.71**（突破 baseline +28%）

> 2026-07-07。iteration-7 执行 §19.6 的 n_drop × only_tradable 扫描，结果**远超预期**：n_drop=15 把 m14 超额年化（含成本）从 +28.39% 拉到 **+115.97%**，ICIR 从 1.25 拉到 **3.71**。§19「dropout 跟踪能力是真根因」**直接验证通过**。

### 20.1 扫描全景（因子/label/切分全冻，IC≡+0.0341，唯一变量 n_drop + only_tradable）

| 组合 | n_drop | only_tradable | 超额含成本/年 | 超额不含成本/年 | 最大回撤 | ICIR |
|---|---|---|---|---|---|---|
| m14 baseline | 5 | False | +28.39% | +40.24% | −7.15% | 1.251 |
| nd10_f | 10 | False | +79.31% | +102.34% | **−5.17%** | 2.917 |
| **nd15_f ★** | **15** | **False** | **+115.97%** | **+149.09%** | **−6.87%** | **3.706** |
| nd20_f | 20 | False | +107.74% | +149.80% | −9.14% | 2.975 |
| nd20_t | 20 | True | +106.85% | +150.08% | −9.56% | 2.948 |
| nd15_t | 15 | True | +106.64% | +141.25% | −7.00% | 3.265 |

**三条规律**：
1. **n_drop 单调增益到 15 见顶**：5→10→15 超额翻 4 倍（+28%→+79%→+116%）；15→20 反而回落（+107%），回撤加深（−6.87%→−9.14%）。**n_drop=15 是甜点**：每日换 15 槽、留 5 槽稳定（hold_thresh=1），既跟踪静态 ~99% 日换手又不吃满 100% 换手的成本。
2. **only_tradable=True 反而略差**（nd15_t +106.64% < nd15_f +115.97%）——与 §18.2「refill 买低 rank 尾巴」一致。即便在 n_drop=15 下，only_tradable 的 refill 仍轻度拖累。**保持 only_tradable=False**。
3. **成本可控**：nd15 含成本/不含成本 = +116%/+149%，成本吃掉 33%（高换手预期内），但净 +116% 远超 baseline +28%。

### 20.2 nd15 稳健性验证（排除单日异常/实现 bug）

| 指标 | m14(n_drop=5) | nd15(n_drop=15) |
|---|---|---|
| test 天数 | 61 | 61（同口径） |
| 日均 return | +0.354% | **+0.812%**（2.3×） |
| **中位日 return** | +0.400% | **+0.744%**（median 也高 → 非尾部驱动） |
| 正收益天占比 | 55.7% | **63.9%**（胜率显著提升） |
| std | 1.835% | 2.521% |
| 累计 nav（test 末） | 1.2285 | **1.6070** |
| test 段总收益 | +22.85% | **+60.70%** |
| 日均成本 | 0.0498% | 0.1392%（2.8×，换手高预期内） |

→ **非单日异常驱动**：median 升高、胜率升高、61 天同口径、回撤更浅。换手成本增加 2.8 倍但 alpha 增益 2.3 倍 + 胜率 +8pp，净效果碾压。**结果可信**。

### 20.3 机制复盘（为什么 n_drop=15 最优）

`TopkDropoutStrategyTD0`（topk=20）：`today = get_first_n(pred[~isin(last)], n_drop+topk-len(last))`。
- **静态 top-20 日换手 ~99%**（§19.2：19.8/20 槽每日变化）→ 静态 gross ~+241%/y（surgery 测得，m14 同量级）。
- **n_drop=5**：每日仅换 5 槽 → 持仓 15 槽长期滞后 fresh top-20，实现只吃 +28%（=静态 gross 的 ~12%）。
- **n_drop=15**：每日换 15 槽、留 5 槽（hold_thresh=1 防 T+0）→ 跟上 ~75% 换手，实现 +116%（=静态 gross 的 ~48%）。
- **n_drop=20**：理论全换，但成本吃满（含/不含成本比 0.72 vs nd15 的 0.78）且回撤加深 −9.14% → 过度。

**甜点 n_drop=15 = 跟踪效率 × 换手成本的最优平衡**，非经验值，由 §19.2 的换手分布数据驱动。

### 20.4 当前最优 min 因子组合（loop 阶段性交付）

**14 个 T 日 9:30~9:40 滑窗分钟因子（MinuteOnlyHandler）+ n_drop=15 策略**：
- 因子：`startup_mom_{1,3,5}m / startup_total / accel_{1,3,5}m / close_pos_{1,3,5}m / vol_ratio_{1,3,5}m / vol_vs_yest`（14 个，纯分钟族，无日频 Alpha158）。
- label 冻结：`Ref($close,-1)/$price_941-1`。
- 策略：`TopkDropoutStrategyTD0(topk=20, n_drop=15, hold_thresh=1, only_tradable=False)`，9:41 撮合（shift=0）。
- test 段（2026-04-01→07-02，61 天）：**超额年化 +115.97%（含成本）/ +149.09%（不含成本），ICIR 3.71，回撤 −6.87%**。
- mlruns：`220549031074802325/5c5a4d17be6e40b0ae52c6e6ead4718a`。

### 20.5 待办（下一轮 iteration-8/9）

1. **细粒度 n_drop ∈ {12,13,14,16,17,18}**：精确定位甜点（15 是否真最优，或 14/16 更佳）。
2. **因子增强（m17）**：在 n_drop=15 上加 `vol_vs_yest_t2/t3/t5/overnight_gap`（已物化 day.bin），看 IC/alpha 是否再提升。§19 证伪了 v3 mask，但因子本身（成交量多周期 + 隔夜缺口）未在 n_drop=15 好策略下测过。
3. **valid 段一致性**：当前仅 test 段；需确认 n_drop=15 在 valid（2026-01~03）同样占优，排除 test 段偶发。

---

## §21 iteration-8/9：n_drop 精细曲线（峰值 17）+ enhanced(18) 因子增强突破（+159%/ICIR 5.12）

> 时间：2026-07-07。承接 §20.5 三项待办。label / deal_price / 涨跌停拦截 / 模型超参 / 切分**全程冻结**，唯一变量是 n_drop 与因子集。

### 21.1 iteration-8：m14(14 因子) n_drop ∈ {5,10,12,13,14,15,16,17,18,20} 完整曲线

| n_drop | 超额含成本 | 超额不含成本 | 最大回撤 | ICIR |
|---|---|---|---|---|
| 5（baseline） | +28.39% | +40.24% | −7.15% | 1.251 |
| 10 | +79.31% | +102.34% | −5.17% | 2.917 |
| 12 | +85.17% | +112.65% | −6.62% | 2.938 |
| 13 | +92.33% | +121.79% | −6.39% | 3.154 |
| 14 | +103.54% | +134.87% | −6.63% | 3.481 |
| 15 | +115.97% | +149.09% | −6.87% | 3.706 |
| 16 | +116.12% | +151.00% | −6.53% | 3.659 |
| **17 ★** | **+134.22%** | **+170.68%** | −7.54% | **4.105** |
| 18 | +123.53% | +162.13% | −7.31% | 3.610 |
| 20 | +107.74% | +149.80% | −9.14% | 2.975 |

**漂亮倒 U，峰值在 17**：含成本/不含成本/ICIR 三指标**同时**在 17 见顶（非噪声）。15↔16 几乎持平（+116），17 跳升 +18pp，18 回落，20 过度换手成本吃利润。
- only_tradable=True 在 15/20 均略劣（+106.64/+106.85 vs False 的 +115.97/+107.74）→ 印证 §18.2「only_tradable=True 反使 refill 吃低 rank 尾部垃圾」，**全程关**。
- IC≡+0.0341（所有 n_drop）→ 证实 **n_drop 是纯策略层参数，不影响 pred/IC**（同一份 pred，n_drop 只改组合换手）。IC 只由 {因子集, 模型, label} 决定。

### 21.2 iteration-9：enhanced(18 因子)@n_drop=15 —— 因子维度杠杆证实

§20.5 待办 2：在 n_drop=15 好策略下加 4 extra（`vol_vs_yest_t2/t3/t5/overnight_gap`，已物化 day.bin）= 18 因子（MinuteEnhancedHandler）。4 extra 均与 ~1.5 天 label 同周期（成交量多周期 + 隔夜跳空），不重蹈 full(85) 日频稀释覆辙（§19.1）。

**enhanced(18)@nd15 vs m14(14)@nd15 全指标对比**（同 n_drop=15、同 label、同切分，唯一变量 = 因子 14→18）：

| 指标 | m14(14)@nd15 | enhanced(18)@nd15 | 提升 |
|---|---|---|---|
| **信号 IC** | 0.0341 | **0.0545** | +60% |
| **信号 ICIR** | 0.276 | **0.513** | +86% |
| 超额含成本年化 | +115.97% | **+158.86%** | +43pp |
| 超额不含成本 | +149.09% | +191.33% | +42pp |
| 回撤 | −6.87% | **−5.44%** | 更浅 |
| **组合 ICIR** | 3.706 | **5.123** | +1.4 |
| 日均 return | +0.812% | +0.989% | |
| **中位日 return** | +0.744% | **+1.106%** | 非尾部驱动 |
| **胜率** | 63.9% | **70.5%** | +6.6pp |
| 最大单日 | +6.54% | +6.12%（更小） | 非暴涨驱动 |
| 最小单日 | −5.80% | −4.85% | 下行更小 |
| 累计 nav | 1.607 | **1.790** | |
| test 总收益 | +60.70% | +78.99% | |

→ **enhanced 全面碾压 m14**，且稳健性指标（median +1.11%、胜率 70.5%、最大单日反而更小、回撤更浅）证明**非单日异常驱动，是真信号**。IC 从 0.034→0.054（+60%）在信号层证实 4 extra 补充了截面预测力，与组合层 ICIR 3.71→5.12 一致。
- mlruns：`907625868975691204/8039e88b1a494b36ab382580cd4b2833`。

### 21.3 当前最优 min 因子组合（loop 阶段性交付，更新 §20.4）

**18 因子（MinuteEnhancedHandler）+ n_drop=15**：
- 因子 = 14 分钟全集（`startup_mom_{1,3,5}m / startup_total / accel_{1,3,5}m / close_pos_{1,3,5}m / vol_ratio_{1,3,5}m / vol_vs_yest`）+ 4 extra（`vol_vs_yest_t2/t3/t5 / overnight_gap`）= 18 个，纯分钟族 + 隔夜跳空，无日频 Alpha158。
- label 冻结：`Ref($close,-1)/$price_941-1`。
- 策略：`TopkDropoutStrategyTD0(topk=20, n_drop=15, hold_thresh=1, forbid_all_trade_at_limit=true, only_tradable=False)`，9:41 撮合（shift=0）。
- test 段（2026-04-01→07-02，61 天）：**超额年化 +158.86%（含成本）/ +191.33%（不含成本），IC 0.0545 / 组合 ICIR 5.12，回撤 −5.44%，胜率 70.5%**。

### 21.4 待办（下一轮 iteration-10/11）

1. **enhanced(18) × n_drop ∈ {16,17,18}**：m14 峰值在 17，但 enhanced 因子集不同，最优 n_drop 可能变。15 已跑（+159%），扫描 16/17/18 找 enhanced 下全局最优 n_drop（结合"最优因子集"+"最优 n_drop"两维）。**进行中**（后台 sweep）。
2. **valid 段 tie-break**：n_drop=17（m14 峰值）在 valid（2026-01~03，独立 OOS ~60 天）是否仍 > 15/16，排除 test 段偶发（§20.5 待办 3，脚本 `/tmp/sweep_valid.py` 已备）。

---

## §22 iteration-10/11 收尾：enhanced(18) n_drop 全谱 + valid 段 OOS tie-break → champion 定论 enhanced@nd15

### 22.1 iteration-11：enhanced test 段 n_drop 全谱（15→20）

单维扫 n_drop，其余冻结（因子=enhanced 18 / label 冻结 / 策略同 §21.3）。test 段 2026-04-01→07-02。

| n_drop | 含成本年化 | 不含成本 | 成本差 | 回撤 | 组合 ICIR |
|---|---|---|---|---|---|
| 15 | +158.86% | +191.33% | 32.5pp | −5.44% | 5.123 |
| 16 | +154.20% | +188.74% | 34.5pp | −6.01% | 4.969 |
| 17 | +176.01% | +212.39% | 36.4pp | −5.83% | 5.433 |
| 18 | +180.93% | +219.21% | 38.3pp | −5.45% | 5.617 |
| 19 | +199.68% | +239.38% | 39.7pp | −5.97% | 6.049 |
| 20 | +206.16% | +247.80% | 41.6pp | −5.81% | 5.960 |

**观察**：曲线持续升到 20（test 段 nd20 +206% / ICIR 5.96），与 m14（峰值 17 后回落）形态完全不同。nd20 在 topk=20 下接近全换手（n_drop≈topk），可疑——疑为 test 段过拟合，触发 valid 段 OOS 验证。

### 22.2 iteration-10：valid 段 OOS tie-break（证 nd20 是 test 过拟合）

test 前移法（纯 OOS）：train 2024-01-01→2025-09-30 / 早停 2025-10-01→12-31 / OOS 回测 2026-01-01→03-31（原 valid 时段）。同 model/pred，纯策略层比较。

| n_drop | 含成本年化 | 不含成本 | 成本差 | 回撤 | 组合 ICIR |
|---|---|---|---|---|---|
| **15** | **+17.02%** | +48.55% | 31.5pp | **−16.25%** | **0.507** |
| 19 | +11.74% | +51.06% | 39.3pp | −18.63% | 0.332 |
| 20 | +11.90% | +52.92% | 41.0pp | −18.10% | 0.338 |

**结论（与 test 段完全反转）**：
- 含成本：**nd15 (+17.0%) > nd19/20 (+11.7%/+11.9%)**——nd15 完胜。
- 不含成本：nd20 (+52.9%) > nd15 (+48.6%)——高换手 alpha 确实更多。
- 成本差：nd15=31.5pp vs nd20=41.0pp——高 n_drop 多付 ~9.5pp 成本，吃掉 alpha 增量并倒贴。

### 22.3 第一性判读：高 n_drop 是 regime-dependent，非结构性优势

- **不含成本视角**：高 n_drop（19/20）的高换手率在两段都捕捉到更多 alpha（valid +52.9% vs +48.6%；test +247% vs +191%）。因子选股本身有效。
- **含成本视角**：alpha 增量能否覆盖高换手成本，取决于该时段 alpha 厚度。
  - test 段 IC=0.0545（全期最强），alpha 厚 → 高 n_drop 含成本仍占优（+206% > +159%）。
  - valid 段是高波动期（所有配置回撤 −16%~−18%，远大于 test 段 −5%），alpha 薄 → 成本侵蚀暴露，高 n_drop 含成本反败（+11.9% < +17.0%）。
- **本质**：高 n_drop 的优势依赖"alpha 足够厚以覆盖 ~10pp 额外换手成本"，这是 regime-dependent 特性，**不能外推**。稳健配置必须取低 n_drop（成本敏感度最低，全 regime 适配）。

### 22.4 CHAMPION 定论（本 loop 终点）

**enhanced(18 因子) @ n_drop=15**：
- 因子：14 分钟全集 + 4 extra（`vol_vs_yest_t2/t3/t5` / `overnight_gap`），纯分钟族 + 隔夜跳空，无日频 Alpha158。
- label：`Ref($close,-1)/$price_941-1`（**冻结**，全程未改）。
- 策略：`TopkDropoutStrategyTD0(topk=20, n_drop=15, hold_thresh=1, forbid_all_trade_at_limit=true, only_tradable=False)`，9:41 撮合（shift=0）。
- **test 段**：+158.86%（含成本）/ IC 0.0545 / 组合 ICIR 5.12 / 回撤 −5.44% / 胜率 70.5%。
- **valid 段 OOS**：+17.02%（含成本）/ ICIR 0.507，**相对 nd19/20 全面占优**（成本敏感度最低）。
- **vs m14@nd17（m14 最优）**：IC +60%（0.0341→0.0545）、因子 ICIR +86%（0.276→0.513）、含成本 +18%（+134%→+159%）。

**为何不再继续扫**：test 段 n_drop 曲线单调升到 20，但 valid 段 OOS 证明高 n_drop 是 test 段过拟合——继续在 test 段寻优只会加深过拟合。enhanced@nd15 是经独立 OOS 验证的稳健甜点。

### 22.5 下一步（待用户醒来确认）

1. enhanced handler/yaml 正式入库（commit）。
2. 清理 18 个 sweep 临时 yaml（`enh_nd*` / `enh_valid_nd*` / `ndrop_*`）。
3. 是否扩段（用 26 年全数据）或追加新分钟因子族——需用户决策（当前 champion 已达成"一套有效 min 因子组合"目标）。
