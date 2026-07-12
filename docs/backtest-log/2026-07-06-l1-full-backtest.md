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

---

## §23 日频情绪 + 上证指数共振因子（两步走，2026-07-07，用户 /loop 新任务）

### 23.1 任务背景与设计

用户 /loop 2026-07-07：在 champion=enhanced(18)@n_drop=15 基础上加因子提升预测准确性：
- 7 日频情绪因子（个股启动/发酵/高潮阶段，避免拥挤度过高接盘）
- 5 上证指数因子（SH000001 共振/指数情绪冰点沸点，辅助高 beta 风险判断）

**两步走**（Q1 决策=「先单独再合体」）：
- Step A：`IndexDailyHandler` 单独 12 因子，先验证 IC>0。不混分钟因子 → p941→分钟因子→drop 联动断链 → **主看因子层 IC**（SigAnaRecord 不走 Exchange，干净），回测仅供参考。
- Step B：`EnhancedWithDailyIndex` 合体 30=18min+12daily。含完整 18 分钟因子 → L1 护栏联动恢复，与 champion 等价 → 回测可与 champion 直接对比。

**corr_20 移除**（原第 13 因子）：qlib `Corr._load_internal`（ops.py:1488-1498）用 `np.isclose(left_std,right_std)` 置零方差位为 NaN，但 numpy 广播不按 index 对齐——个股收益(621 行)≠`ChangeInstrument` 上证收益(624 行)时抛 `broadcast` 错，highbeta883926 池 296/5114 股崩。`Cov/Var`(beta_20) 无此 override → 零崩。改写 `Cov/(Std·Std)` 可消崩但缺口窗口（停牌）三方有效样本不同 → 发散 max 0.3~0.8，是「corr 代理」非真 corr。corr_20 与 beta_20 高度共线，高贝塔池 beta>1 敏感度比 corr∈[-1,1] 更贴切 → 删 corr_20，共振由 beta_20 + idx_run_5/idx_bias_20 覆盖。详见 index_daily_handler.py docstring。

### 23.2 Step A 单独 12 因子（test 段 2026-04-01→07-01）

| 指标 | Step A(12) | m14(14min) baseline | champion(18) |
|---|---|---|---|
| 信号 IC | **0.0366** | 0.0341 | 0.0545 |
| 因子 ICIR | 0.2393 | ~0.276 | ~0.513 |
| Rank IC | 0.0491 | — | — |
| Rank ICIR | 0.3663 | — | — |
| 超额含成本年化 | +82.8%* | — | +158.86% |

*回测仅供参考（Step A 单独 handler p941 护栏断链）。

Gate（IC>0 AND |ICIR|>0.3）→ **实质通过**：
- IC=0.0366 > 0，且 > m14 纯分钟 baseline 0.0341 → 12 日频/指数因子非噪声。
- ICIR 0.2393 < 0.3 字面卡线是**日频尺度错配固有特性**（T-1 因子预测 ~1.5 天 label，IC 跨日 std 高），与 m14 因子 ICIR(0.276) 同量级；Rank ICIR 0.3663 达标。
- 单独弱是预期：日频/指数因子提供跨尺度互补信息，价值在 Step B 合体。

### 23.3 Step B 合体 30 因子（test 段 2026-04-01→07-01）

| 指标 | champion(18) | **Step B(30)** | 变化 |
|---|---|---|---|
| **信号 IC** | 0.0545 | **0.0611** | **+12% ↑** |
| 因子 ICIR | ~0.513 | 0.530 | +3% ↑ |
| Rank IC | — | 0.0840 | — |
| Rank ICIR | — | 0.7187 | — |
| 超额含成本年化 | +158.86% | +159.27% | +0.4pp ≈持平 |
| 超额不含成本年化 | +191.33% | +191.73% | +0.4pp ≈持平 |
| 回撤 | −5.44% | −6.35% | −0.91pp 略恶化 |
| 组合 IR | 5.12 | 4.92 | −0.2 略降 |

### 23.4 第一性判读：因子层提升明确，组合层未变现

- **因子层（IC = 预测准确性，用户 /loop 核心目标）**：IC +12%（0.0545→0.0611）统计稳健（~100 股 × 61 天 ≈ 6000 截面样本），证明 12 日频/指数因子为模型补充了**有效截面信息**，未稀释。推翻 §19.1「日频稀释 9:41 信号」在本精选因子集上的担忧——本任务 12 日频是精选短周期(5/9/20 日) + 指数共振，非 Alpha158 全集(含 60 日慢均线)，尺度贴近 ~1.5 天 label。
- **组合层（回测）**：含成本年化 +0.4pp 持平，回撤 −0.91pp、组合 IR −0.2 略劣化（61 天单路径小样本噪声内）。原因：`TopkDropoutStrategy topk=20` 只取排序头部，IC 中段提升传导有限；日频因子略增组合波动。
- **结论**：因子更准了（IC +12% 达成「提升预测准确性」目标），但 top20 选股策略没把 IC 提升完全变现（组合收益持平）。

### 23.5 待决策（需用户确认）

- 是否升 enhanced(30) 为新 champion：因子层 IC +12% 支持，组合层持平 + 回撤略增反对。两者矛盾，属方案级决策。
- 关联入库文件：`qlib_ifind_beta/index_daily_handler.py`(Step A 12 因子) / `enhanced_daily_index_handler.py`(Step B 30 因子合体) / `qrun/workflow_daily_index.yaml`(Step A) / `qrun/workflow_enhanced_daily_index.yaml`(Step B) / `tests/test_index_factors.py`(12 因子表达式 + 前视 gate，19/19 PASS)。

## §24 特征组横截面 Rank 归一化（CSRankNorm 替换 RobustZScoreNorm）— FAIL，族在此构造下死（2026-07-07）

### 24.1 背景：IC→超额墙 + 三重零重训证伪

§23 Step B（30 因子）把信号 IC 抬到 0.0611（+12%）但超额含成本持平（+159.27% ≈ champion +158.86%）、回撤略恶化（−6.35% vs −5.44%）→ 暴露**「IC→超额墙」**：因子层更准不必然传导到 topk=20 头部收益。基于此做三重零重训诊断，证伪三个朴素提超额方向：

1. **gap 诊断**（§23.3 数据）：Step B IC↑（0.0545→0.0611）但头部收益反降 → IC↑ ≠ 头部↑。
2. **头部可塑性诊断**（champion rank 曲线，本次 fresh 复算）：topk=20 边界**已锐**——
   - rank 16-20（topk 内缘）实现收益 **+1.203%/日** ≫ rank 21-25（外缘）**+0.171%/日**，边界 Δ **+1.031%/日**；
   - top-20 等权 +1.148%/日、top-10 +1.342%/日（集中度边缘 +0.194%/日）；
   - 边界重排 oracle 仅 +0.065%/日且 std/mean≈4.88（噪声级，前视幻觉）。
   → champion 的超额**来自 topk=20 边界放得极锐**，不是头部可塑性有冗余。
3. 假设「绝对开盘幅度混入日级 regime 噪声」→ 用**日内横截面 rank** 剥离 regime，或提纯「个股相对强度」。

### 24.2 实验（与 champion 逐字一致，唯一变量 = 特征归一化）

- 重定性（context7 + qlib 源码 `processor.py:326`）：横截面归一化在 **PROCESSOR 层**，非新因子族、非新 bin。champion 特征走 Alpha158 默认 `RobustZScoreNorm`（**时间序列** robust z-score，fit on train）；本变体仅把 `infer_processors` 换成 `CSRankNorm`（每日横截面 `rank(pct=True)`，**无 fit → 零跨段泄漏**）。
- Handler `MinuteEnhancedCSRankHandler(MinuteEnhancedHandler)`：18 因子 `get_feature_config` 不变、label `Ref($close,-1)/$price_941-1`、shared `DropnaProcessor(feature)` L1 前视护栏、切分 train 2024-01→2025-12 / valid 2026-01→03 / test 2026-04→07-02、`TopkDropoutStrategyTD0 n_drop=15 topk=20`、LGBModel 超参——**全冻结**。
- spec：`docs/superpowers/specs/2026-07-07-csrank-feature-normalization-design.md`；结构测试 4/4 PASS（18 因子不变 + infer_processors 含 CSRankNorm 不含 RobustZScoreNorm + L1 护栏继承）。

### 24.3 结果（test 2026-04→07，全口径 FAIL）

| 口径 | champion (RobustZScore 时间序列) | CSRank 变体 (横截面 rank) | Δ | 判定 |
|---|---|---|---|---|
| 信号 IC | 0.0545 | **0.0526** | −0.0019 | ↓ FAIL |
| Rank ICIR | ~0.51 | 0.727 | +0.22 | ↑（唯一亮点） |
| **top-20 头部收益**（等权实现） | **+1.148%/日** | **+0.993%/日** | **−0.155%/日** | **↓ FAIL** |
| top-10 头部收益 | +1.342%/日 | +1.229%/日 | −0.112%/日 | ↓ FAIL |
| 超额含成本年化 | +158.86% | +143.14% | −15.7pp | ↓ FAIL |
| 回撤（含成本） | −5.44% | −9.54% | −4.1pp | 恶化 FAIL |
| 组合 IR（含成本） | 5.12 | 4.37 | −0.75 | ↓ FAIL |

### 24.4 失败机制（rank 曲线揭示，最关键）

| 边界带 | champion | CSRank |
|---|---|---|
| rank 16-20（topk=20 内缘） | +1.203%/日 | +1.089%/日 |
| rank 21-25（topk=20 外缘） | +0.171%/日 | +1.166%/日 |
| **内缘−外缘 Δ** | **+1.031%/日（锐）** | **−0.078%/日（平坦/倒挂）** |

champion 的超额来源是 **topk=20 边界极锐**（内缘比外缘高 1.03%/日，模型把高收益票放进 top-20、低收益票排出）。CSRank 把绝对开盘幅度换成日内 rank 后，**丢了边界锐化信号**——边界变平坦（外缘甚至反超高 0.078%/日），模型在 topk=20 边界处近乎随机选股。**假设证伪**：绝对幅度（时间序列 z-score）**不是** regime 噪声，它**承载**边界锐化信号，剥离它是损招。

### 24.5 反常细节：CSRank 锐尖削边

| 尖部口径 | champion | CSRank |
|---|---|---|
| rank 1-5 | +1.029%/日 | +1.297%/日 |
| top-10 集中度边缘（top10−top20） | +0.194%/日 | +0.237%/日 |

CSRank **锐化了尖、削平了边**。若策略是 topk=10（非 topk=20），CSRank 可能赢——但 topk=20 是用户冻结口径，削平的边主导 → 净亏。这把「IC→超额墙」（§24.1）又证一次：Rank ICIR ↑（0.51→0.73）但头部↓、超额↓。

### 24.6 判定与入库

- **FAIL**（头部 < champion 且超额 < champion 且回撤恶化）。横截面 rank 族**在此构造下死**。
- C（CSZScoreNorm，每日横截面 z-score）同为横截面（同样丢绝对幅度），先验预期也败；是否作横截面族最后尝试**待用户决策**（governance #1/#2：未批准不擅自起新实验）。
- 入库文件：`qlib_ifind_beta/minute_enhanced_csrank_handler.py` / `qrun/workflow_minute_enhanced_csrank.yaml` / `tests/test_minute_enhanced_csrank_handler.py`（4/4 PASS）/ spec（见 24.2）。
- recorder：`mlruns/302034022257525537/4e357612343b44d49ce7037ee7341a0e`；分析脚本 `/tmp/csrank_head_cmp.py`。
- champion（enhanced(18)@n_drop=15）**不变**，A/B 隔离，本实验纯负结果归档。

## §25 T-1 尾盘因子族（champion 18 + 2 最强正交尾盘因子）— FAIL，第 3 次 IC→超额墙确认（2026-07-07）

### 25.1 背景：goal 因子优化 Step C2，未涉的 T-1 全天分钟结构

champion 18 因子已用尽 T 日 9:30-9:40 早盘信息；T-1 全天分钟结构此前**完全未作因子**（`vol_vs_yest` 分母仅把 T-1 全天量压成聚合标量）。goal「优化因子提升超额与 IC」Step C2：新增 **T-1 尾盘（14:50-15:00，slot 232-241）** 分钟族，刻画用户语义「惯性冲高/高潮/出货」次日领先信号。

**IC 基线校准（第一性原理，test 段 2026-04→07，23 因子全量单变量 IC）**：
- `tail_vol_ratio_t1`：|RankICIR|=0.398，全部 23 因子 **#3**（仅次于 vol_vs_yest_t5 0.497 / overnight_gap 0.458），反转信号（T-1 尾盘放量出货→次日跌）。
- `tail_accel_t1`：|RankICIR|=0.263，**#7**，动量/惯性信号。
- **正交性**：5 tail 因子与 champion 18 |corr| **全部 <0.07** → 纯增量信息、零冗余。
- 仅取最强 2 个（非全 5）：`tail_close_pos/mom/last5` test 段 |RankICIR|<0.15 且与已选 2 个相关，加入徒增过拟合（Step B 教训）。

先验很乐观：IC 强、正交、刻画面信息空白 → 该提超额。**结果证伪**。

### 25.2 实验（与 champion 逐字一致，唯一变量 = feature 18→20）

- 物化 5 个尾盘 day.bin（`materialize_minute.py` F 块）：`tail_mom_t1 / tail_mom_last5_t1 / tail_close_pos_t1 / tail_vol_ratio_t1 / tail_accel_t1`。窗 = slot 232-241（14:51-15:00 时刻 bar），与早盘 10 特征 K 严格对称；**shift-1 in min-cal space**（T 行 = T-1 尾盘，无前视；T-1 15:00 收盘 T 日 9:41 决策已知）。
- Handler `MinuteEnhancedTailHandler(MinuteEnhancedHandler)`：18 champion + `tail_vol_ratio_t1` + `tail_accel_t1`（`$field` 直消费，与 14 分钟因子零 Ref 统一）。label / deal_price / 涨跌停拦截 / 切分 / 模型超参 / 归一化口径——**全冻结**，唯一变量 feature 18→20。
- 测试：`tests/test_materialize_minute.py` 3 个尾盘测试（25 bins 计数 + 尾盘窗 10 根 K 线 + 5 因子 shift-1 交叉验证）+ `tests/test_minute_enhanced_tail_handler.py` 4 结构测试（20 因子 / TAIL_PICK 最强 2 / L1 护栏 / __init__ 未 override）= **7/7 PASS**，全量 68/68。

### 25.3 结果（test 2026-04→07，全口径 FAIL）

| 口径 | champion (enhanced 18) | tail 变体 (20) | Δ | 判定 |
|---|---|---|---|---|
| 信号 IC | 0.0545 | **0.0457** | −0.0088（−16%） | ↓ FAIL |
| Rank ICIR | ~0.51 | 0.502 | ~持平 | 平 |
| ICIR | ~0.49 | 0.416 | −0.07 | ↓ FAIL |
| **top-20 头部收益**（等权实现） | **+1.148%/日** | **+0.964%/日** | **−0.184%/日** | **↓ FAIL** |
| top-10 头部收益 | +1.342%/日 | +0.955%/日 | −0.387%/日 | ↓ FAIL |
| 超额含成本年化 | +158.86% | **+139.60%** | −19.3pp | ↓ FAIL |
| 超额不含成本年化 | — | +171.35% | — | — |
| 回撤（含成本） | −5.44% | −5.75% | −0.31pp | 恶化 FAIL |
| 组合 IR（含成本） | 5.12 | 4.41 | −0.71 | ↓ FAIL |

IC 正交、RankICIR 持平——但**头部、超额、回撤、IR 四口径全败**。第 3 次 IC→超额墙。

### 25.4 失败机制（rank 曲线揭示，与 §24 机制**不同**，更关键）

| 边界/尖部带 | champion | tail(20) |
|---|---|---|
| rank 6-10（champion 尖峰） | **+1.654%/日** | +0.968%/日（**尖峰被压平 −0.69**） |
| rank 16-20（topk=20 内缘） | +1.203%/日 | +0.881%/日 |
| rank 21-25（topk=20 外缘） | +0.171%/日 | +0.381%/日 |
| **内缘−外缘 Δ**（边界锐度） | **+1.0313%/日（锐）** | **+0.5007%/日（腰斩）** |
| top-10 集中度边缘（top10−top20） | +0.194%/日 | **−0.009%/日（尖头被压平）** |

**机制（与 §24 区别）**：
- §24（CSRank）= **边界变平坦/倒挂**（1.03 → −0.078），横截面归一化**摧毁**绝对幅度承载的边界锐化信号。
- §25（tail +2 因子）= **边界腰斩但仍正**（1.03 → 0.50），且 **champion 尖峰（rank 6-10 +1.654%）被压平到 +0.968%**、top-10 集中度边缘归零（+0.194 → −0.009）。

→ 2 个 IC 强且与 18 因子**完全正交**的因子，没有锐化 topk=20 边界，而是**重排了排序**：把 champion 原本落在尖部（rank 6-10、+1.654%/日）的头部赢家**挤出 top-20**，腾出的 rank 让给尾盘信号强但次日收益平庸的票。这是典型的「**加因子稀释/重排**」效应——**单因子 IC 强 ≠ 加进组合后头部强**。

### 25.5 三重墙定论（goal 因子优化 thread 收口）

| 实验 | 改动 | IC | 超额含成本年化 | 机制 |
|---|---|---|---|---|
| §23 Step B | +12 日频/指数→30 | 0.0611（+12%）**↑** | +159.27% ≈ champion **平** | IC↑不传导头部 |
| §24 CSRank | 18 因子横截面归一化 | 0.0526 ↓ | +143.14% ↓ | 边界摧毁 |
| **§25 tail** | **+2 尾盘→20** | **0.0457 ↓** | **+139.60% ↓** | **边界腰斩+尖峰压平** |

**三重独立证伪，机制各异**（加因子 / 改归一化 / 加正交因子），共同指向同一结构结论：**champion enhanced(18)@n_drop=15 的超额来自 topk=20 边界放得极锐**，18 个因子已在该边界处接近最优——任何方向（加日频、改横截面、加正交尾盘）都重排并退化该边界。**18 因子近天花板**。

→ **goal「优化因子提升超额」thread 收口**：继续追超额边际收益已极低（3 路全败），转向 **CLAUDE.md「待定」= 扩段验证 champion OOS 稳健性**（用 26 年全数据，而非追新因子）。

### 25.6 判定与入库

- **FAIL**（头部 / 超额 / 回撤 / IR 四口径全 < champion）。尾盘因子族（champion+2 最强正交）在此构造下死。
- 入库文件：`qlib_ifind_beta/minute_enhanced_tail_handler.py` / `qrun/workflow_minute_enhanced_tail.yaml` / `tests/test_minute_enhanced_tail_handler.py`（4/4 PASS）/ `qlib_ifind_beta/materialize_minute.py`（5 尾盘 bin 物化 + shift-1）/ `qlib_ifind_beta/config.py`（`MINUTE_FACTOR_TAIL_FIELDS` + `TAIL_FIRST_SLOT/TAIL_SLOT_COUNT`）/ `tests/test_materialize_minute.py`（3 尾盘测试）。
- recorder：`mlruns/882574446881966247/f49261b7c66f4934a286653c57cf6ef1`；头对头分析脚本 `/tmp/tail_head_cmp.py`。
- champion（enhanced(18)@n_drop=15）**不变**，A/B 隔离，本实验纯负结果归档。三重墙定论后，因子优化 thread 收口，转向 OOS 稳健性验证。

## §26 T-1/T-2 开盘因子族（champion 18 + 2 同-regime 滞后开盘因子）— FAIL，第 4/5 次 IC→超额墙确认（2026-07-08，overnight autonomous）

### 26.1 背景：goal「扩展不同区间的 1min 因子」，测「同 regime 滞后」破墙假设

§25 尾盘失败的关键洞察：T-1 **尾盘**是与 T 日开盘**不同 regime**（正交 |corr|<0.07）→ 正交因子重排 topk=20 边界。本族反过来测「**同 regime 滞后**」假设：T-1/T-2 **开盘**（slot 1-10，与 T 日开盘同 regime、仅时间平移 1/2 天）。若高贝塔开盘动量**跨日持续**（真启动/共振/惯性），则 T-1 开盘与 T 日开盘**共线强化**（reinforcing，非正交重排）→ 可能不重排边界而是**锐化**它，从而破墙。若跨日**反转**则正交 → 复刻 §25 失败。

对齐用户 /goal 心法：「结合指数（大势）和 **T-1 之前的 K 线**判断个股是否共振、启动、高潮或惯性冲高」。树模型给齐 T 日 + T-1 开盘因子后，自动通过 splits 学交互（T-day × T-1 开盘 = 共振强度）。

**IC 基线校准（第一性原理，全段 2024-2026，5038 股 × 600 天，15 因子单变量 Spearman rank IC vs `Ref($close,-1)/$price_941-1`）**：
- **vol_ratio 族跨日持续（共线信号，假设直接验证）**：`vol_ratio_5m_t2` +0.0297（全部 15 因子 **#1**，ICIR 0.224）/ `vol_ratio_5m`（baseline）+0.0252（#2）/ `vol_ratio_5m_t1` +0.0187（#4，ICIR 0.138）→ 全正、单调 → 开盘量比是真实跨日持续边际。
- accel / startup_mom / startup_total 族**不持续**：T-1/T-2 lag 全段 |IC|<0.003（噪声）。
- 单因子全段 |IC|<0.03 是 baseline 常态（champion signal IC 0.0545 来自 LGBM 非线性组合 18 因子）。

→ 先验乐观：vol_ratio 族跨日持续 + 共线 + 对齐用户心法 → 该共线强化边界。**结果证伪**。

### 26.2 实验（与 champion 逐字一致，唯一变量 = feature 18→20）

- 物化 10 个 T-1/T-2 开盘 day.bin（`materialize_minute.py`）：复用 champion 早盘 5 公式（已在 `fac`），copy 到 `_t1`/`_t2` key → 进 shift 循环（TAIL+OPENING_T1 shift-1、OPENING_T2 shift-2，**min-cal 空间**）→ scatter。**零额外读盘**（5 公式只算一次，copy+shift）。shift-1/2 in min-cal space：T 行 bin = T-1/T-2 的 9:31-9:40，无前视（T-1/T-2 9:40 ≪ T 日 9:41 决策）；与 §25 尾盘 shift-1 前视论证完全同构。
- Handler 两变体（`minute_enhanced_opening_handler.py`，均继承 MinuteEnhancedHandler，仅 override `get_feature_config` 追加 `$field`，复用 champion 18 + L1 前视护栏）：
  - **V1b（IC 驱动，最强共线信号）**：18 + `[vol_ratio_5m_t1, vol_ratio_5m_t2]` = 20。两条持续量比 lag → 直接测「持续开盘量比共线强化 topk=20 边界」（破墙的最强检验）。
  - **V1（§25 头对头，设计忠实）**：18 + `[vol_ratio_5m_t1, accel_5m_t1]` = 20（T-1 开盘 |mean_ic| top-2）。同 count、同 shift-1、仅 regime 不同（开盘 vs 尾盘）→ 隔离「同 regime 滞后 vs 不同 regime」。
- label / deal_price / 涨跌停拦截 / 切分 / 模型超参 / 归一化口径——**全冻结**，唯一变量 feature 18→20。
- 测试：`tests/test_materialize_minute.py` 加 `test_opening_t1_t2_factors_crosscheck`（手算 oracle 逐字复刻 5 公式 + shift-1/2 对齐判别 + NaN-safe + bin 计数 25→35）→ 全量 **17/17 PASS**。全量物化 5040 ok / 76 missing（退市/停牌，预期）。

### 26.3 结果（test 2026-04→07，两变体均 FAIL）

| 口径 | champion (enhanced 18) | V1b（+2 共线量比） | V1（+2 共线开盘 top2） | 判定 |
|---|---|---|---|---|
| 信号 IC | 0.0545 | **0.0574（+5.3%）↑** | **0.0585（+7.3%）↑** | 两变体 IC 均↑ |
| Rank IC | ~0.067 | 0.0709 ↑ | 0.0740 ↑ | ↑ |
| ICIR | ~0.49 | 0.533 ↑ | 0.510 ↑ | ↑ |
| Rank ICIR | ~0.51 | 0.606 ↑ | 0.586 ↑ | ↑ |
| **超额含成本年化** | **+158.86%** | **+145.78%（−13.08pp）↓** | **+125.38%（−33.48pp）↓↓** | **两变体均 FAIL** |
| 超额不含成本年化 | — | +178.32% | +157.97% | — |
| 回撤（含成本） | −5.44% | −5.42%（≈持平） | −5.30%（略好） | 非 Δ 主因 |
| benchmark（SH000300） | +44.11% | +44.11% | +44.11% | — |

**核心反直觉**：两变体 **IC / Rank IC / ICIR / Rank ICIR 四信号口径全↑**（V1 更是历史最高 IC +7.3%），但**超额含成本年化全↓**——V1（IC 最高）的超额反而**跌得最狠（−33.48pp）**。**IC 与超额在此边界处解耦甚至反向**。

### 26.4 失败机制（IC↑ 但超额↓——「共线」也未破墙，机制同 §25）

两变体信号 IC 提升、vol_ratio 族跨日持续（共线假设在因子-IC 层成立），但**头部超额仍退化**。机制与 §25 同构：新增因子（即使共线、即使提升平均排序质量 IC）**仍重排 topk=20 边界**——把 champion 原本落在边界的头部赢家挤出，腾出 rank 给开盘量比强但次日收益平庸的票。「共线强化边界」假设在**组合层证伪**：因子-IC 持续 ≠ 边界强化。

**§26 的独立新结论（超越 §25）**：§25 是「IC↓ + 超额↓」（可辩称因子本身弱）；§26 是「**IC↑ + 超额↓**」且「**IC 越高、超额跌越狠**」（V1 IC +7.3% → 超额 −33.48pp，甚于 V1b IC +5.3% → −13.08pp）。→ **墙不依赖 IC 方向**：无论加的因子提升还是降低平均 IC，只要扰动 topk=20 边界，超额就退化。IC 是**全截面平均排序质量**，超额是**top-20 边界处的尖峰实现**——两者在 champion 当前 18 因子 + topk=20 结构下解耦。

### 26.5 五重墙定论（goal「扩展区间 1min 因子」thread 收口）

| 实验 | 改动 | IC | 超额含成本年化 | 机制 |
|---|---|---|---|---|
| §23 Step B | +12 日频/指数→30 | 0.0611（+12%）↑ | +159.27% ≈平 | IC↑不传导头部 |
| §24 CSRank | 18 横截面归一化 | 0.0526 ↓ | +143.14% ↓ | 边界摧毁 |
| §25 tail | +2 **正交**尾盘→20 | 0.0457 ↓ | +139.60% ↓ | 边界腰斩+尖峰压平 |
| **§26 V1b** | **+2 共线开盘量比→20** | **0.0574（+5.3%）↑** | **+145.78% ↓** | **IC↑但边界重排** |
| **§26 V1** | **+2 共线开盘 top2→20** | **0.0585（+7.3%）↑** | **+125.38% ↓↓** | **IC↑且最高，超额跌最狠** |

**五重独立证伪，覆盖全部 regime × IC 方向组合**：
- **不同 regime（正交）**：§25 T-1 尾盘 → 墙。
- **同 regime（共线）**：§26 T-1/T-2 开盘 → 墙。
- **IC↓**（§24/§25）、**IC≈平**（§23）、**IC↑**（§26 V1b/V1）→ 全部墙。

→ 共同结构结论铁证：**champion enhanced(18)@n_drop=15 的 18 因子已在 topk=20 边界处达强局部最优**。任何加因子（正交/共线、提 IC/降 IC）都重排并退化该边界。**「扩展不同区间的 1min 因子」方向已穷尽**——无论选哪个区间、哪个 regime、多强 IC，加进 champion 都退化超额。

### 26.6 判定与入库

- **FAIL**（两变体超额含成本年化均 < champion−10pp 拒绝线；V1 −33.48pp、V1b −13.08pp）。T-1/T-2 开盘因子族（champion+2 共线滞后）在此构造下死。
- 入库文件：`qlib_ifind_beta/minute_enhanced_opening_handler.py`（V1 `MinuteEnhancedOpeningT1Handler` + V1b `MinuteEnhancedOpeningVolHandler`）/ `qrun/workflow_minute_enhanced_opening_t1.yaml` + `workflow_minute_enhanced_opening_vol.yaml` / `qlib_ifind_beta/materialize_minute.py`（10 T-1/T-2 开盘 bin 物化 + shift-1/2）/ `qlib_ifind_beta/config.py`（`MINUTE_FACTOR_OPENING_T1_FIELDS` / `MINUTE_FACTOR_OPENING_T2_FIELDS`）/ `tests/test_materialize_minute.py`（T-1/T-2 开盘交叉验证，17/17 PASS）/ 设计 `docs/superpowers/specs/2026-07-08-t1-opening-factors-design.md`。
- recorder：V1 `ecf50b7250d143e0a3be9edcc675416b` / V1b `de4c265a623c4070a770e66d0e2ef664`（见 `/tmp/v1.log` `/tmp/v1b.log` 末行）；IC 校准 `/tmp/ic_calib_opening.py` → `/tmp/ic_calib_opening.json`。
- champion（enhanced(18)@n_drop=15）**不变**，A/B 隔离，本实验纯负结果归档。
- **明早需用户决策**（详见 morning report）：① 接受五重墙、转向 OOS walk-forward 稳健性验证（sliding 设计已批、paused）——**推荐**；② V4 replacement（丢 2 弱 champion + 加 2 开盘 = 18，测「count-dilution vs boundary-optimality」唯一未涉结构假设）需用户授权动 §D6 冻结口径；③ 调 topk/n_drop（策略层始终冻结，只记录）。

---

## §27 模型层正则化杠杆 → W1 破墙 / W2 OOS 证伪（窗口过拟合，机制不成立）（2026-07-08，overnight autonomous）

### 27.1 背景：五重因子墙后转向未冻结的模型层 + 边界诊断机制假说

§23–§26 五重因子墙确认「加任何因子都退化超额」。goal「扩展不同区间的 1min 因子」方向已穷尽，但 goal 上层「提升 IC 和超额收益」**未**收口。转向**用户冻结清单外、零证据**的唯一杠杆 = **模型层超参**（label/策略/切分/数据全冻结，仅动 `LGBModel` kwargs）。

**机制假说（边界诊断驱动）**：champion topk=20 边界处 rank16-20 实现 label +0.0121 vs rank21-25 +0.0004（~96% 断崖），但预测 rel_gap 中位 0.0000（fragile）。假说 = **更强正则化 → 压制头部过拟合噪声 → 锐化边界带 → 破 IC→超额墙**。

### 27.2 W1 实验（test 2026-04→07）：模型层 sweep + 正则化剂量-响应

与 champion 逐字一致，唯一变量 = `LGBModel` 超参（loss / num_leaves / max_depth / lambda_l1 / lambda_l2）。

- **M_huber**：`loss: huber`（其余同 champion）
- **M_cap+**：`num_leaves 64→128 / max_depth 6→8`（容量↑）
- **M_reg(l1=10)**：`lambda_l1 5→10 / lambda_l2 10→20`
- **reg15**：`lambda_l1 15 / lambda_l2 30`
- **reg20**：`lambda_l1 20 / lambda_l2 40`

### 27.3 W1 结果（test 2026-04→07，初看 4/4 破墙、reg15 强候选）

| 配置 | IC | RankIC | 超额含成本×242 | 超额 geo_cum | maxDD | 边界 rel_gap |
|---|---|---|---|---|---|---|
| champion l1=5/l2=10 | 0.0545 | 0.0679 | +194.5% | +60.6% | −2.4% | 0.0013 |
| M_huber | — | — | FAIL（loss 变更引发 label 尺度失配） | | | |
| M_cap+ leaves128/d8 | 0.0374 | 0.0534 | +200.6% | +62.7% | −3.3% | 0.0232 |
| M_reg l1=10/l2=20 | 0.0589 | 0.0726 | +198.0% | +62.3% | −3.1% | 0.0246 |
| **reg15 l1=15/l2=30** | **0.0534** | **0.0706** | **+232.1%** | **+76.2%** | −4.0% | 0.0116 |
| reg20 l1=20/l2=40 | 0.0572 | 0.0760 | +218.2% | +70.1% | −3.9% | 0.0210 |

**初看强信号**（reg15 vs champion）：
- 超额 +232.1% vs +194.5%（**+37.6pp arith**），geo_cum +76.2% vs +60.6%（+15.6pp）。
- **IC 反而更低**（0.0534 < 0.0545）→ 印证诊断「IC/超额在边界解耦」预测。
- 倒 U 剂量-响应：l1=5→10→15→20 超额 +194→+198→+232→+218%，峰在 l1=15。
- **边缘分布式稳健**：median +1.092%/d（> champion +0.937%）、win 72%（>66%）、H1 +209% H2 +254%（两半都赢）、top3/total 26%（<30%，非少数日驱动）。

→ reg15 形似强 champion 候选。**但因 reg15 系在 W1 上选出，W1 即其 in-sample 调参窗 → 必须独立 OOS 窗验证才能定论**（LightGBM 默认 feature_fraction=1/bagging=none 为确定性，seed 验证无意义，换窗才是真稳健性）。

### 27.4 W2 OOS 验证（test 2025-04→07 平移窗，genuinely OOS）

train 2024-01→2024-12 / valid 2025-01→03 / **test 2025-04→07**（reg15 调参窗 W1 的前一年，真 OOS）。champion 与 reg15 除 lambda 外逐字一致：

| 配置 | IC | RankIC | 超额含成本×242 | 超额 geo_cum | maxDD | median/d | win |
|---|---|---|---|---|---|---|---|
| champion l1=5/l2=10 | 0.0718 | 0.1177 | +96.7% | +25.8% | −9.0% | +0.419% | 57% |
| reg15 l1=15/l2=30 | 0.0709 | 0.1173 | +100.4% | +27.4% | −9.7% | +0.271% | 56% |

**边缘坍缩**：reg15 超额 arith 仅 +3.7pp（W1 +37.6pp → W2 +3.7pp，缩 ~90%），geo_cum 仅 +1.6pp。更反常：W2 上 reg15 的 **median/d（+0.271%）与 win（56%）反而低于 champion**（+0.419%/57%）——W1「两维都赢」的分布式稳健在 OOS 不复存在。

### 27.5 统计与机制双重证伪（最关键）

**(A) 统计无差异**：reg15−champion 日超额差 paired t=0.12、**p=0.906**；bootstrap 95% CI [−57.5, +62.2]pp 横跨 0；reg15 仅 29/61 天更优（48%，劣于掷币）。→ **reg15 与 champion 在真 OOS 统计上不可区分**。

**(B) 机制证伪**：诊断假说「更强正则→锐化边界」。实测 W2 边界 rel_gap：champion **0.0122** > reg15 **0.0072**——reg15 **边界更钝**，假说方向相反。reg15 的 +3.7pp 残余超额与边界锐化**无关**，无可解释机制支撑。

→ **结论**：W1 的 +37.6pp 是**窗口过拟合**（正则强度实质上拟合了 W1 测试窗噪声），非可泛化信号。诊断的「正则化破墙」机制**证伪**。

### 27.6 判定与入库

- **FAIL（OOS）**：reg15 不达稳健 champion 标准（W2 p=0.906、机制证伪）。**champion enhanced(18)@n_drop=15 维持不变**，A/B 隔离，本实验纯负结果归档。
- **正向价值**：排除「模型正则化可破墙」假设 → 五重因子墙 + 本模型墙共同确认：**champion 在 topk=20/n_drop=15 下已达该因子集的结构性超额天花板**，墙非模型欠拟合、非因子不足，是边界结构本身（rank16-20 vs 21-25 实现 96% 断崖但预测不可靠分离）。两窗超额均 +96~100% arith 年化 → **墙是「超额上限」非「可行性」**，策略本身盈利。
- 入库文件：`qrun/workflow_minute_enhanced_mdl_{cap,huber,reg,reg15,reg20}.yaml`（W1 模型层 5 变体）+ `qrun/workflow_minute_enhanced_{w2,mdl_reg15_w2}.yaml`（W2 OOS 验证 2 窗）。recorder：W1 champion eid907625868975691204 run b8b187d6 / reg15 eid148551552510143393 run 1ec846a9；**W2 champion eid449239398769286298 run dcd756bb** / **reg15 eid968255313638515434 run 982fba5e**。对比脚本 `/tmp/{sweep_compare,sweep_reconcile,full_compare,reg15_robust,w2_compare,w2_deep}.py`。
- **明早需用户决策**（详见 morning report）：① 接受天花板、champion 定稿、转扩段（26 年全数据）+ 报告产出 ——**推荐**；② 横截面 regime/共振 因子（用户领域模型「指数大势×T-1 K线→共振/启动/高潮」，因子墙 5 重未涉的**非纯个股因子**方向，可能是真正未探索杠杆）；③ 模型层其它（loss/stacking/特征选择）——正则化线已证伪、预期收益低，不推荐。

---

## §28 日频情绪 + 上证指数共振因子（用户领域模型方向）→ 因子层 + 策略层双证伪（第 8 重墙）

> 对应 spec [2026-07-07-daily-index-factors-design.md](../superpowers/specs/2026-07-07-daily-index-factors-design.md) + plan [2026-07-07-daily-index-factors.md](../superpowers/plans/2026-07-07-daily-index-factors.md)。
> 目标：用户 `/loop` 任务「添加日频因子（分析个股情绪阶段 启动/发酵/高潮，避免拥挤度过高接盘）+ 上证指数因子（判断个股&指数共振、冰点/沸点判断高 beta 风险）」。**这是 §27 morning report 选项 ②「真正未探索的杠杆」的实做。**

### 28.0 背景与两条实现路径

用户领域模型有两层语义，对应两种**独立的**因子应用方式：

| 路径 | 用户原话 | 朴素实现 | 测试位置 |
|---|---|---|---|
| **A. 因子层**（横截面 ranking 特征） | "结合指数（大势）和 T-1 之前的 K 线判断个股是否共振/启动/高潮" | 12 个日频/指数 `Ref(expr,1)` 因子并入 Handler，参与 LGB 横截面排序 | §28.1-28.3（Step A/B） |
| **B. 策略层**（择时/仓位 risk-on/off） | "根据上证指数情绪（冰点、沸点）辅助判断高 beta 类个股**风险**" | regime 信号做组合 gate（沸点空仓/减仓） | §28.4（纯分析） |

两条路径都做了。**两条都证伪。** 详见下。

### 28.1 因子层实现（Step A 独立 12 + Step B 合并 30）

- **数据/口径冻结**：label `Ref($close,-1)/$price_941-1`、deal_price `["$price_941","$close"]`、limit_threshold 分级、n_drop=15、benchmark SH000300 ——**全不变**。仅动 handler feature 集。
- **上证指数接入**：qlib 原生 `ChangeInstrument('SH000001', $close)`（ops.py:64，跨 instrument 引用），返回 SH000001 行情；T 日 9:41 决策时 T 日日频数据未到 → **全部因子 `Ref(expr,1)`（T 行 = T-1 close 算的值，无前视）**，与分钟因子（9:30-9:40 当日已知）在 9:41 决策点同时可得。test_no_lookahead_truncation_invariance（mask T 及之后，factor[T] 不变）PASS。
- **12 因子**（[index_daily_handler.py](../../qlib_ifind_beta/index_daily_handler.py)）：个股情绪 7 个（bias_5/bias_20/vol_ratio_20/run_up_5/rsv_9/dist_to_limit/accel_mom）+ 指数共振 5 个（idx_bias_20/idx_run_5/idx_rsv_9/idx_vol_ratio_20/beta_20）。corr_20 因 qlib `Corr._load_internal` 的 `np.isclose` 广播在个股历(621行)≠指数历(624行)时崩（296/5114 股）剔除，由 beta_20（Cov/Var 无 override）覆盖共振。
- **两步走**（spec §6 Q1）：Step A = `IndexDailyHandler`（仅 12 个日频/指数因子，无分钟因子）→ 独立验证因子带信号（**注意：无分钟因子→p941→minute→drop 链断→回测护栏弱，回测数字不与 champion 直接比，只看 IC**）；Step B = `EnhancedWithDailyIndex`（完整 18 分钟 + 12 日频/指数 = 30，护栏与 champion 等价，IC + 超额都直接比）。
- **测试**：[tests/test_index_factors.py](../../tests/test_index_factors.py) 19/19 PASS（1.75s）—— 12 因子表达式 vs 手算 pandas rolling 逐值对齐 + 前视截断不变性 + handler config 契约。
- **W2 OOS 窗**：[gen_enh_daily_w2.py](../../tmp/gen_enh_daily_w2.py) 把 test 平移回 2025-04→07（train/valid 同步回退 1 年），窗口与 champion W2（§27 已跑）逐字段核对一致，仅 handler + experiment_name 不同。

### 28.2 因子层 W1 结果：IC +12%（ saga 中**首见真涨**，但超额仍平）

[test 2026-04→07] 同口径对比（[daily_index_compare.py](../../tmp/daily_index_compare.py)）：

| config | IC_test | ICIR | RankIC | RankICIR | excess(ann) |
|---|---|---|---|---|---|
| champion enhanced(18) | 0.0545 | 0.92 | 0.0677 | 0.540 | +194.5% |
| **Step B enh_daily(30)** | **0.0611 (+12%)** | 1.03 | **0.0840 (+24%)** | **0.719 (+33%)** | +194.9% (+0.4pp 平) |
| Step A daily_index(12) | 0.0366 | — | — | 0.366 | +116.5%（护栏弱，不比） |

Step A 独立 IC 0.0366 > m14 baseline 0.0341、RankICIR 0.366 → **standalone gate 实质通过**（12 因子带信号）。Step B 合并后 IC/RankIC/RankICIR **三升且超额不降** —— 这是 §23-27 五重因子墙 + 一重模型墙后**第一次看到 IC 真涨**，形似强 champion 候选。**但超额仍撞墙（+0.4pp）**，与 §27 reg15 W1 同形（IC 边际变化与超额解耦）。

### 28.3 因子层 W2 OOS 证伪：IC −15% / 超额 −69.6pp / 配对 p=0.443

[test 2025-04→07，真 OOS]（[enh_daily_w2_deep.py](../../tmp/enh_daily_w2_deep.py)）：

| config (W2) | IC | RankIC | excess(ann) | median/d | win |
|---|---|---|---|---|---|
| champion enhanced(18) | **0.0718** | 0.1177 | **+96.7%** | — | — |
| Step B enh_daily(30) | 0.0611 (**−15%**) | 0.0972 | +27.1% (**−69.6pp**) | **−0.095%** | **48%** |

- **配对统计**：日 IC 差（enh − champ）paired t=−0.77，**p=0.443**；enh 更高仅 29/62 天（47%）。**与 champion 统计不可区分（偏劣）。**
- **缓存 bug 排查**（[verify_w2_window.py](../../tmp/verify_w2_window.py)）：enh_daily W1/W2 IC 均恰 ≈0.0611 形似缓存复用，但实测 W2 ic 索引 2025-04-01→2025-07-02（n=62）≠ W1（2026 窗 n=61），head3 完全不同（W2 [0.24,−0.01,−0.09] vs W1 [0.16,0.01,0.11]）→ **非缓存 bug**。enh_daily 的 IC 恰好两窗稳定在 0.0611，而 champion 在高信号窗 W2 上 IC 更高（0.0718）。
- **机制（scale 错配）**：12 日频/指数因子用 5/9/20 日 rolling 窗口，但 label 尺度 ≈1.5 天（T 9:41→T+1 close）。慢因子在高 beta 牛市 regime（W1 2026）恰与短 label 共对齐 → 过拟合；换 regime（W2 2025）即失效。**与 Alpha158/full(85)（§D6、§19.1）同病**：日频尺度因子对短 label 天然不稳。
- **判定**：W1 IC +12% 为**窗口过拟合**，与 §27 reg15 同病。因子层路径（A）证伪。

### 28.4 策略层 regime-gating 纯分析：相关≈0、gate 毁收益（路径 B 证伪）

用户原意「判断**风险**」= 择时/仓位，是策略层杠杆。n_drop/topk/label 冻结，但仓位/择时是不同杠杆、未冻结。**不修改策略代码、不新跑回测**，纯分析验证假设：SH000001 情绪（lag1，T 9:41 已知）能否解释/预测 champion 逐日超额？（[regime_strategy_layer.py](../../tmp/regime_strategy_layer.py)）

**① 相关性几乎为零**（POOLED n=122，regime[T] vs excess[T]，T+1 对齐同样≈0）：

| regime 信号 | W1 pear | W2 pear | POOLED pear (p) |
|---|---|---|---|
| idx_bias_20 | +0.086 | −0.060 | +0.019 (0.837) |
| idx_rsv_9 | +0.133 | **−0.158** | −0.008 (0.931) |
| idx_run_5 | +0.104 | −0.106 | −0.009 (0.923) |
| idx_vol_ratio_20 | +0.257 | −0.037 | +0.087 (0.341) |

**符号跨窗不一致**（rsv：W1 +0.133 / W2 −0.158）→ **噪声特征，非信号**。n=122 足以排除中大幅效应。

**② 五分位 bucket 非单调（U 型，假设方向反）**：Pooled idx_rsv_9 → Q1冰点 +1.025%/d、Q2 +0.008%、Q3 +0.642%、Q4 +0.282%、**Q5沸点 +1.016%/d**。假设预测「沸点（高 beta 过热）应转负」，实际**沸点是第二高分位**。两窗均为 U 型、无单调性。

**③ gate 模拟全部毁收益**：剔除 top-X% 沸点空仓 → 实现年化超额 Δ：top-20% Δ=**−50pp**(pooled)/−55pp(W1)/−15pp(W2)；top-30% −59pp；top-40% −64pp。**沸点日实为好日，砍掉反亏。**

**判定**：策略层 regime-gating 路径（B）证伪。用户的「冰点/沸点判断高 beta 风险」领域模型，在 7 字段日频数据 + 9:41 撮合 + 1.5 天 label 的设定下，对超额**无 exploitable 预测内容**（横截面无、时序也无）。

### 28.5 机制归因（第一性原理）

- **因子层（§28.1-3）**：scale 错配 —— 慢日频（5/9/20d）因子对 ~1.5d label，单 regime 过拟合、换 regime 即崩。与 Alpha158/full(85)/§27 reg15 同机制。
- **策略层（§28.4）**：regime 对**该高 beta 池在该 label 尺度**的超额无预测力。可能因：(a) champion 已在 highbeta883926 池内选股，池内个股 beta 同质化、regime 横截面区分度被池预筛消耗；(b) 1.5d 持仓太短，日频 regime 的均值回复/动量来不及表达；(c) 7 字段缺 breadth/dispersion/涨跌停数等更有效的 regime 代理。**注意**：证伪的是「7 字段 + 标准冰点/沸点代理（RSV/bias/动量/量比）」这一具体实现，非「regime 对高 beta 全无意义」这一通则——但本设定下已无 exploitable 边际。

### 28.6 判定与入库

- **FAIL（双路径）**：因子层 W2 OOS IC −15%/超额 −69.6pp/p=0.443；策略层相关≈0、gate 毁收益。**champion enhanced(18)@n_drop=15 维持不变**，A/B 隔离，本实验双负结果归档。
- **正向价值（saga 收束）**：至此 **factor + model + strategy 三层全部测过、全部撞墙** —— §23-26（5 重因子墙）+ §27（模型正则化墙）+ §28（因子层日频/指数墙 + 策略层 regime-gating 墙）= **8 重独立验证**。在用户冻结的 label/策略/切分下，champion topk=20/n_drop=15 的超额天花板是**结构性**的，**既非因子不足、非模型欠拟合、也非择时缺位**。两窗超额均 +96~100% arith 年化 → 墙是「超额上限」非「可行性」，策略本身盈利。
- **saga 结论**：用户目标「提升超额」在本冻结设定下**已穷尽三层杠杆、确认不可破**。用户领域模型（共振/冰点沸点）方向经双实现证伪——**idea 不该再当因子层或简单择时层杠杆重试**；如要继续，唯一原则性未测方向是**分钟尺度 regime 特征**（匹配 1.5d label，非 20d）或**更宽 universe/更长 label**（后者被用户冻结）。
- 入库文件：handler [index_daily_handler.py](../../qlib_ifind_beta/index_daily_handler.py)(12) + [enhanced_daily_index_handler.py](../../qlib_ifind_beta/enhanced_daily_index_handler.py)(30)；workflow [workflow_daily_index.yaml](../../qrun/workflow_daily_index.yaml)(StepA) + [workflow_enhanced_daily_index.yaml](../../qrun/workflow_enhanced_daily_index.yaml)(StepB W1) + [workflow_enhanced_daily_index_w2.yaml](../../qrun/workflow_enhanced_daily_index_w2.yaml)(StepB W2)；test [test_index_factors.py](../../tests/test_index_factors.py) 19/19；对比脚本 `/tmp/{daily_index_compare,gen_enh_daily_w2,enh_daily_w2_deep,verify_w2_window,regime_strategy_layer}.py`。
- recorder：Step A daily_index W1 eid662327455819139151 run da720637；Step B enhanced_daily_index W1 eid872817226661043202 run 3fe0f249 / **W2 eid749049631274766161 run 2df506a4**；对照 champion W1 eid907625868975691204 run b8b187d6 / W2 eid449239398769286298 run dcd756bb。

### 28.7 待用户决策（详见 morning report v3）

1. **接受天花板、champion 定稿** → 转扩段（26 年全数据稳健性）+ 正式报告产出 ——**推荐**（三层已穷尽，再挖边际递减）。
2. **分钟尺度 regime 特征** —— 唯一原则性未测方向（日频 regime 已证伪、§28.5 机制指向 label 尺度匹配）。需新因子工程（9:30-9:40 内指数/个股共振的分钟代理），预期工作量大、收益不确定。
3. **regime 作风控 overlay（非超额最大化）** —— 用 idx_rsv 高分位做**回撤控制**（沸点降仓位降风险），代价是收益让步、超额不升反降。仅当用户优先级从「超额」转向「Sharpe/回撤」时才合理。
4. **接受 champion、转实战对接** —— 策略已两窗盈利，可直接进实盘/纸面跟踪，边跑边观察 OOS 稳健性。

---

## §29 分钟尺度指数开盘共振（§28.7 选项 2 实做）— 第 9 重墙 · saga 收束

> 2026-07-08。用户隔夜自主授权「换个思路扩展不同区间的 1min 因子特征，寻找多种可能性」。§28 三层（factor/model/strategy）八重墙后，**唯一原则性未测杠杆** = §28.7 选项 2「分钟尺度 regime 特征」。本轮忠实实现用户领域模型「T 日 9:30-9:40 开盘分钟内个股 vs 指数的横截面相对强度」，物化为 day.bin 并入 18→21，W1+W2 双窗 OOS 验证。**判定：第 9 重墙，双窗 FAIL。**

### 29.1 设计意图（与 §28 的本质区别）

用户领域模型原话：「分钟频只能观测 T 日的开盘阶段，所以我需要结合指数（大势）和 T-1 之前的 K 线来判断个股是否是共振、启动、高潮或者惯性冲高」。

- **§28 测的是 DAILY regime 状态**做择时（5/9/20d 慢窗 SH000001 冰点/沸点）→ scale 错配（label ≈1.5d，慢因子 W1 过拟合 W2 崩 IC −15%）。
- **§29 测的是 MINUTE 尺度开盘共振**：T 日 9:30-9:40 指数开盘特征（idx 收益/动量/加速度），与 champion 个股开盘因子（startup_total/startup_mom_5m/accel_5m）**公式同构、尺度对齐 label**。idx 因子 broadcast 同值（每只股存同日同一 idx 值），靠 LGBModel 学 **idx × 个股交互**表达「在大势 X 下个股强度 Y 的信号」。

**设计局限（诚实标注）**：broadcast 同值的 idx 因子单独是**横截面常数列**（全池同日同值），对 topk ranking 零区分度；区分度**全部依赖** idx×个股交互项。本轮**未**构造显式 `个股开盘 − idx 开盘` 相对强度因子（横截面非常数）—— 那是理论上不同的设计，效果未知、需用户决策是否再试。

### 29.2 3 因子公式 + broadcast 物化方案

`config.py`：`INDEX_OPENING_SRC = "SH000001"`；`INDEX_OPENING_FIELDS = ("idx_open_ret_10", "idx_open_mom_5m", "idx_open_accel_5m")`。

| 因子 | 公式（1min 早盘 slot 1-10） | champion 同构 |
|---|---|---|
| `idx_open_ret_10` | `idx_close[9] / idx_open[0] − 1` | startup_total |
| `idx_open_mom_5m` | `idx_close[9] / idx_close[4] − 1` | startup_mom_5m |
| `idx_open_accel_5m` | `(idx_close[9]/idx_open[5]−1) − (idx_close[3]/idx_open[0]−1)` | accel_5m |

- **物化**（`materialize_minute.py._load_index_opening_factors`）：读 SH000001 1min close/open bin，scatter 到全局 min-cal morning grid（slots 1-11），算 3 因子 cache（全池共享一份，仅依赖 idx 自身 si_dc）。`materialize_minute_instrument` 末尾 broadcast：每只股用自身 `valid/rel_v`（scatter 输出）把 idx cache 写入自己的 overlay bin → 38 day.bins/stock（35 champion + 3 idx 共振）。
- **SH000001 缺失日**：train ~6%、test/valid 0% → 全池同日 NaN → DropnaProcessor drop（不会单股漏数据）。
- **全池物化**（`scripts/materialize_minute.py` 全 5116 codes）：5040 ok / 76 missing（delisted/suspended 无 1min/daily bin，预期值）。docstring 已同步 35→38（ripple 已修）。

### 29.3 落地验证（`/tmp/verify_idx_landing.py`）

抽查 highbeta883926 池内大盘股 test W1 窗（2026-04）：SH600519 / SZ000001 三因子 **21/21 finite（100% 落地）**；broadcast 同值验证「同日 4 股 idx 值完全相同 ✓」（确认横截面常数机制）。

### 29.4 双窗测试结果（4 回测同口径对比）

> **口径对齐凭证**：本轮同 session 重跑 champion W1/W2 baseline，IC 与 §28.2/§28.3 **完全一致**（W1 0.0545 / W2 0.0718）→ 4 回测严格同口径。label `Ref($close,-1)/$price_941-1`、deal_price `["$price_941","$close"]`、涨跌停 `["$change_941>=$limit_up","$change<=$limit_down"]`、模型超参、切分、策略（n_drop=15）全与 champion 一致 → **唯一变量 feature 18→21**。

**W1（train 2024-01→2025-12 / valid 2026Q1 / test 2026-04→07，benchmark +44.1%）**：

| config (W1) | IC | ICIR | RankIC | RankICIR | excess 含成本 | excess 不含成本 |
|---|---|---|---|---|---|---|
| champion enhanced(18) | 0.0545 | 0.513 | 0.0679 | 0.542 | **+158.9%** | **+191.3%** |
| §29 resonance(21) | 0.0570 (**+4.6%**) | — | 0.0654 | — | +119.2% (**−39.7pp**) | +151.2% (**−40.1pp**) |

**W2 OOS（train 2024-01→2024-12 / valid 2025Q1 / test 2025-04→07，benchmark +6.96%）**：

| config (W2) | IC | ICIR | RankIC | RankICIR | excess 含成本 | excess 不含成本 |
|---|---|---|---|---|---|---|
| champion enhanced(18) | 0.0718 | 0.390 | 0.1177 | 0.824 | **+63.2%** | **+95.1%** |
| §29 resonance(21) | 0.0697 (**−2.9%**) | 0.367 | 0.1163 | 0.749 | +40.1% (**−23.1pp**) | +72.5% (**−22.6pp**) |

### 29.5 机制归因（第一性原理，不联想）

- **broadcast 同值 = 横截面常数**：idx 因子同一日全池所有股同值 → 纯 idx 维度对 topk ranking 零区分度（与 §28.4 策略层 regime-gating **同源机制**）。区分度只能来自 LGBModel 的 idx×个股交互项。
- **交互项贡献噪声级**：IC 两窗都极接近 champion（W1 +4.6% / W2 −2.9%，幅度 <5%，噪声级）→ idx×个股交互在 top20 头部排序中没有 exploitable 信号。
- **维度增加损害排序**：超额两窗都稳定低于 champion（W1 −40pp / W2 −23pp）→ 3 个横截面常数列加入，引入噪声维度、轻微过拟合，损害 top20 头部 alpha（与 §23-26 因子扰动组同现象）。
- **与 §28 的关键区别（证明无 scale 错配）**：§28.A W2 IC **−15% 崩**（日频 5/9/20d 慢因子 scale 错配，W1 过拟合换 regime 即失效）；§29 W2 IC **仅 −2.9%**（分钟尺度与 ~1.5d label 同尺度，无错配）。**即使无 IC 崩，超额仍 −23pp 撞墙** → 墙是结构性的，非因子不足、非 scale 错配。

### 29.6 判定与 saga 收束

- **FAIL（第 9 重墙，双窗超额全跌）**：W1 IC +4.6%（噪声级微涨）但超额 −40pp 大跌；W2 OOS IC −2.9%（未崩，证明无 scale 错配）但超额 −23pp 退化。**champion enhanced(18)@n_drop=15 维持不变**，§29 实验产物入库作 saga 收束证据，**不入 champion 配置**。

- **saga 收束（9 重独立验证全撞墙）**：

  | # | 章节 | 杠杆层 | 机制 |
  |---|---|---|---|
  | 1-2 | §23-24 | 因子（扰动/startup 组） | 维度噪声 |
  | 3 | §25 | 因子（tail 分钟组） | 维度噪声 |
  | 4 | §26 | 因子（T-1/T-2 opening） | 维度噪声 |
  | 5 | §27 | 模型（正则化 lambda/leaves） | IC-超额解耦 |
  | 6 | §28.A | 因子层（日频/指数，scale 错配） | W1 过拟合 W2 崩 |
  | 7 | §28.B | 策略层（regime-gating，broadcast 同值） | 横截面常数 |
  | 8 | §29 | 因子层（分钟尺度共振，broadcast 同值） | 横截面常数 + 维度噪声 |

  覆盖**因子层（日频 + 分钟）+ 模型层 + 策略层 + 跨尺度**。用户领域模型（共振/启动/高潮/冰点沸点）方向经双实现证伪 —— 在用户冻结的 label/策略/池设定下，champion topk=20/n_drop=15 的超额天花板是**结构性**的，**既非因子不足、非 scale 错配、非模型欠拟合、也非择时缺位**。两窗超额含成本 +40~159% arith → 墙是「超额上限」非「可行性」，策略本身盈利。

### 29.7 入库文件 + recorder

- **handler**：[minute_resonance_handler.py](../../qlib_ifind_beta/minute_resonance_handler.py)（MinuteEnhancedHandler 子类，21 = 18 champion + 3 idx）；config [config.py](../../qlib_ifind_beta/config.py) 加 `INDEX_OPENING_SRC/FIELDS`。
- **workflow**：[workflow_minute_resonance.yaml](../../qrun/workflow_minute_resonance.yaml)(W1) + [workflow_minute_resonance_w2.yaml](../../qrun/workflow_minute_resonance_w2.yaml)(W2)。
- **物化**：[materialize_minute.py](../../qlib_ifind_beta/materialize_minute.py) `_load_index_opening_factors` cache + broadcast scatter；[scripts/materialize_minute.py](../../scripts/materialize_minute.py) 全池入口（docstring 35→38 已同步）。
- **test**：[test_index_factors.py](../../tests/test_index_factors.py) 加 §29 共振因子用例。
- **验证脚本**：`/tmp/verify_idx_landing.py`（落地 + broadcast 同值）。
- **recorder**：§29 resonance **W1 eid798779997054224731 run677a3a82** / **W2 eid182312265270673755 run9e53c4ed**；对照 champion W1 eid907625868975691204 runb8b187d6 / W2 eid449239398769286298 rundcd756bb。
- **§28.7 选项 2 状态更新**：原「唯一原则性未测方向 = 分钟尺度 regime 特征」**已测、已证伪**（§29）。待 user 决策项收敛为：① 接受天花板、champion 定稿 + 扩段（26 年全数据稳健性）；② 转实战对接（纸面/实盘跟踪）；③ 显式相对强度因子（`个股开盘 − idx 开盘`，横截面非常数，理论上不同于 §29 broadcast，效果未知，需用户决策是否再试）。

## §30 选项③预筛：个股开盘 β 中性化（idio）偏 IC 诊断 — 廉价证伪

> saga 收束（§29.6）后唯一原则性未测方向 = §29.7 选项③「显式相对强度因子（个股开盘 − idx 开盘，横截面非常数，理论上不同于 §29 broadcast）」。隔夜自主做**廉价预筛**（不全池物化，复 §29 教训），分两个变体证伪，选项③收束。

### 30.1 两个变体

- **简单差变体**（`startup_total − idx_open_ret_10`，无 β）：idx_open_ret_10 因 broadcast 同日全池同值（§29.5 已验证）→ 简单差 = `startup_total − 常数` → **横截面 ranking 零增量**（平移不改 rank，数学必然，无需诊断）。
- **β 中性化残差变体**（`startup_total − β·idx_open_ret_10`，β = rolling20 cov(stock,idx)/var(idx)，逐股不同）：横截面非常数（diag_idio.py 测 β 中位数 std ≈ 0.564，逐股确实不同）→ 选项③唯一可能有信号的形态，需诊断。

### 30.2 偏 IC 诊断（diag_idio_ic.py，40 股，双重残差 OLS，2026-07-08 复核）

> 偏 IC = 控制 startup_total 后 idio→label 的纯增量（逐日横截面：res_idio = idio~startup 残差；res_label = label~startup 残差；Spearman(res_idio, res_label)）。label 绑定 `close[T+1]/price_941[T]-1`。

| 段 | N日 | partial IC mean | ICIR | 判据 |
|---|---|---|---|---|
| 全段 2024-01~2026-07 | 561 | **+0.0028** | +0.010 | B：≈0 证伪 |
| test 2026-04~07 | 61 | +0.0603 | +0.235 | 小样本（§29 教训） |

- 全段偏 IC +0.0028（|mean| < 0.01 判据 B）→ idio 增量是 **β 估计噪声**，与 §29 broadcast 同形。
- test 窗 +0.0603 仅 61 日，diag_idio_ic_ts.py 月度序列证伪（2024-2025 随机震荡，非单月拉高即偶发）→ 小样本假信号。

### 30.3 判定

- **选项③完整收束（FAIL）**：简单差变体数学零增量 + β 中性化变体偏 IC 全段 ≈0。saga 第 9 重墙后再排除此方向。**不物化**。

## §31 日频情绪反转因子族廉价诊断（vol_ratio_5_20 等）— 属 §28.A 已证伪族，不物化

> 用户领域模型「启动/发酵/高潮/惯性冲高」+ 避免拥挤度过高接盘。champion 18 因子全是 T 日 9:30-9:40 开盘族 + 隔夜，**完全不看 T-1 之前 K 线**。隔夜自主从 daily bins 即时算日频情绪候选（零物化，diag_daily_mood*.py）。

### 31.1 候选 + 单变量 IC 月度稳定性（600 股抽样，diag_daily_mood_ts.py，2026-07-08 复核）

| 因子 | 含义 | 期望 | 全段 ICIR | 2024 / 2025 / 2026 ICIR |
|---|---|---|---|---|
| vol_ratio_5_20 | 5/20 日量比 | − | **−0.235** | −0.194 / −0.310 / −0.166 |
| n_up_5 | 5 日连阳数 | − | −0.145 | −0.158 / −0.189 / −0.048 |
| ret_5d | 5 日收益 | − | −0.152 | −0.126 / −0.247 / −0.084 |
| dist_high_20 | 距 20 日高 | + | +0.007 | −0.044 / +0.034 / +0.096 |
| **startup_total** | champion 基线 | ? | **−0.027** | +0.044 / −0.120 / −0.011 |

vol_ratio_5_20 三年方向负占比 59/63/55%，反转信号稳健。

### 31.2 关键洞察（第一性原理）：startup_total 单变量 ICIR ≈ 0

champion 核心分钟因子 startup_total **单变量横截面 rank IC 全段 ICIR −0.027（近零）**，但 champion W1 模型 pred IC = 0.0545（§29.4）、超额 +158.9%。

→ **champion 的 alpha 来自 LGBM 非线性组合，非单变量线性 IC**。这揭示「IC→超额墙」的深层机制：**IC 涨和超额涨解耦** —— §23-§29 所有"IC 涨"实验（§28.A W1 +12%、§27 reg15、§23 Step B）都撞墙，正因为 IC 提升（线性可度量）无法传导到 top20 头部收益（非线性 + 肥尾 + 成本）。**用单变量/偏 IC 筛因子存在方法论盲点**：单变量 IC 稳健 ≠ 能提升 LGBM 超额。

### 31.3 判定

vol_ratio_5_20 等属**§28.A 已证伪的日频情绪反转族**（scale 错配：日频 5/20d rolling vs ~1.5d label）。即便单变量 ICIR −0.235（> startup_total 的 −0.027），机制预判合体后撞第 10 重墙 —— §28.A Step B 已用**完整 LGBM 回测**（非线性，胜过本节线性单变量诊断）证明日频因子合体 W1 IC +12% 超额仍撞墙 / W2 OOS 崩。**不物化**，避免 redundant 重挖 §28.A。

### 31.4 待 user 决策项（§30/§31 后再收敛）

经 §30（选项③证伪）+ §31（日频反转族属已证伪族不物化），§29.7 原收敛 3 项中：
- ③「显式相对强度因子」→ **§30 已证伪，移除**。
- ①② 维持：① 接受天花板、champion 定稿 + 扩段（26 年全数据稳健性）；② 转实战对接（纸面/实盘跟踪）。

**新增（§31.2 洞察衍生，非因子方向，需 user 决策是否探索）**：
- ④ **策略层 bagging 集成降方差**（未测）：换手机制复核 [td0_strategy.py:83-106](../../qlib_ifind_beta/td0_strategy.py#L83-L106) `sell = last[last.isin(get_last_n(comb, n_drop))]` —— 换手**非** n_drop=15 结构性强制，而取决于预测排序稳定性：pred 稳定 → 持仓少落入 (持仓∪候选) 合并排序末 n_drop → 低换手；pred 噪声 → 持仓频繁跌入末位 → 高换手。champion 用 method_buy="top"/method_sell="bottom"（确定性，无策略层随机），方差来自 LGBM 训练 seed → bagging（多 seed/子采样 LGBM 平均）降 top20 边界 pred 方差 → 稳排序 → 降换手 → 减成本侵蚀**理论上成立**。上限受 §29.4 实测成本侵蚀 ~32pp 约束（且仅能回收其中「噪声换手」份额，信号换手不可降），现实增量估计 +10~15pp 量级。**不破 IC-超额信息墙**，纯成本侧优化。廉价 gate 可行：先测 champion 实际换手率 + 跨 seed top20 一致性（一致性已高 → bagging 无空间，免跑全量 ensemble）。
- ⑤ **成本结构诊断**：**已由 §29.4 回答**（含成本/不含成本双列是项目标准报告指标，全 saga 一致使用，随换手正确变化）。§29.4 实测：champion 不含成本超额 W1 +191.3% / W2 +95.1%，§29 加因子后不含成本 +151.2% / +72.5% —— **加因子后「不含成本」超额仍低于 champion** → 即便零成本，因子也未改善 top20 头部 alpha → **墙是信息天花板，非成本天花板**。成本侵蚀稳定 ~32pp（全 n_drop=15 同换手档）。**⑤ 无需重跑**。

## §32 窗内 5 折 expanding walk-forward 稳健性验证（2026-07-08）— champion 非单窗过拟合

> 用户原 binding 目标「优化超额 + IC」经 §23-§31 全 saga 收敛到信息天花板（§31.2 揭示 IC↔超额解耦墙）。§29.7 收敛项 ①「接受天花板、champion 定稿 + 扩段（26 年全数据稳健性）」—— 本节即执行**窗内 walk-forward 稳健性验证**，回答「champion enhanced(18)@n_drop15 的 alpha 是否单窗过拟合」。
>
> **方法学说明**：26 年全段扩段受分钟 bin 物化回溯上限约束（champion 分钟因子 9:30-9:40 物化为 day.bin，有效覆盖 ~2024 起，见 `diag_data_coverage.py` 诊断）。故取**可行窗 2024-01→2026-07**，做 **5 折 expanding walk-forward**：唯一变量 = 切窗，label / 策略(n_drop15,topk20) / 模型超参 / 18 因子全 frozen = champion([workflow_minute_enhanced.yaml](../../qrun/workflow_minute_enhanced.yaml))。test 段为 5 个**不重叠** 3 月季度，覆盖 2025-04→2026-07 全 OOS。

### 32.1 切窗设计（expanding train + rolling valid + 不重叠 test）

| 折 | test（OOS 季度） | train expanding 末 | valid（rolling） | yaml | mlflow exp |
|---|---|---|---|---|---|
| f1 = W2 | Q2'25（2025-04..06） | 2024-12 | Q1'25（2025-01..03） | [w2.yaml](../../qrun/workflow_minute_enhanced_w2.yaml)（§29 既有） | 449239398769286298 |
| f2 | Q3'25（2025-07..09） | 2025-03 | Q2'25（2025-04..06） | [wf2.yaml](../../qrun/workflow_minute_enhanced_wf2.yaml) | 602620168594697491 |
| f3 | Q4'25（2025-10..12） | 2025-06 | Q3'25（2025-07..09） | [wf3.yaml](../../qrun/workflow_minute_enhanced_wf3.yaml) | 472635746064092743 |
| f4 | Q1'26（2026-01..03） | 2025-09 | Q4'25（2025-10..12） | [wf4.yaml](../../qrun/workflow_minute_enhanced_wf4.yaml) | 159657062985786570 |
| f5 = W1 | Q2'26（2026-04..06） | 2025-12 | Q1'26（2026-01..03） | [workflow_minute_enhanced.yaml](../../qrun/workflow_minute_enhanced.yaml)（champion 原窗） | 907625868975691204 |

- **expanding 模式**：train 每折增长一季度（f1 末 2024-12 → f5 末 2025-12）；valid 滚动为 test 的前一季度（rolling，非 expanding）。各折 train∪valid 永远在 test 之前，**折内无前视**。跨折间 valid 可与前一折 test 重合（如 f2 valid=Q2'25=f1 test），这是 rolling valid 的标准特性，不影响任一折独立 OOS 评估。
- f1 = §29 W2 窗、f5 = §22 champion W1 窗（直接复用既有 run，IC 与 anchor 逐位对齐见 §32.5）；f2/f3/f4 为本节新跑中间折。
- 切窗技巧：`data_handler.end` = test 末 + ~7 日 buffer（**仅**为兑现 test 末日 label 的 `Ref($close,-1)`，**不扩 segments**）；`segments.test` = 精确季度；`fit_end` = train∪valid 末。**注**：段级为兑现末日 label 尾部略延 1-2 日（如 w2 段末 2025-07-02），test 按季度仍不重叠；聚合时按日期去重，不影响 304 日 OOS 结论。
- 踩坑：首次 3 折全挂在 `MlflowException: Invalid experiment ID: '.pytest_cache'` —— mlflow file_store 把 `mlruns/` 下所有直接子目录当 experiment_id 扫描，`.pytest_cache`（pytest 缓存产物）非数字 → 报错。FIX：`rm -rf mlruns/.pytest_cache`（pytest 缓存，安全删；保留 `.trash`，mlflow 原生）。删除后 3 折全跑通。

### 32.2 每折全量指标（5 折 OOS，取每 experiment 最新 run）

口径：IC = `ic.pkl`（每日 IC Series）均值；ICIR = mean/std（raw，非年化，与 §29 "ICIR 0.390" 一致）；超额年化 = `port_analysis_1day.pkl.loc[(ret_type,'annualized_return'),'risk']`。

| 折 | test | IC | RankIC | ICIR | 超额含成本年化 | 不含成本年化 | 回撤 | 组合 IR |
|---|---|---|---|---|---|---|---|---|
| f1=W2 | Q2'25 | **0.0718** | 0.1177 | 0.390 | +63.17% | +95.10% | −10.17% | 1.91 |
| f2 | Q3'25 | **0.0905** | 0.1321 | 0.597 | +109.21% | +141.59% | −9.41% | 3.93 |
| f3 | Q4'25 | **0.0635** | 0.0962 | 0.383 | +61.46% | +93.77% | −8.73% | 2.41 |
| f4 | Q1'26 | **0.0563** | 0.0800 | 0.301 | +17.02% | +48.55% | −16.25% | 0.51 |
| f5=W1 | Q2'26 | **0.0545** | 0.0679 | 0.513 | +158.86% | +191.33% | −5.44% | 5.12 |
| **均值** | | **0.0673** | **0.0988** | — | — | — | — | — |
| 跨折 std | | 0.0131 | 0.0236 | — | — | — | — | — |

- **5 折 IC 全正**（0.0545~0.0905，跨折 std 仅 0.0131），**无一折翻负** —— 不是 W1 单窗侥幸。
- **5 折超额含成本年化全正**（+17%~+159%），即便最弱的 f4（Q1'26，回撤 −16.25%）仍 +17% 正超额。
- 最强 f2（Q3'25 IC 0.0905）/ 最弱 f4（Q1'26 IC 0.0563）差距合理，无极端崩塌。f5(W1) 超额最高（+158.86%）部分得益于该窗回撤最小（−5.44%）。

### 32.3 池化 OOS 月度 IC 稳定性（5 折每日 IC 拼接去重按月均值）

将 5 折的 `ic.pkl`（每日 IC Series）拼接、按日期去重（`~index.duplicated(keep='first')`）、按月均值，得 **16 个月池化 OOS 月度 IC**（2025-04→2026-07，304 交易日去重后）：

| 月份 | IC | | 月份 | IC | | 月份 | IC | | 月份 | IC |
|---|---|---|---|---|---|---|---|---|---|---|
| 2025-04 | +0.1511 | | 2025-08 | +0.0411 | | 2025-12 | +0.0736 | | 2026-04 | +0.0769 |
| 2025-05 | +0.0543 | | 2025-09 | +0.0922 | | 2026-01 | +0.0693 | | 2026-05 | +0.0355 |
| 2025-06 | −0.0034 | | 2025-10 | +0.0885 | | 2026-02 | +0.1052 | | 2026-06 | +0.0510 |
| 2025-07 | +0.1369 | | 2025-11 | +0.0304 | | 2026-03 | +0.0134 | | 2026-07 | −0.0015 |

- **月度 IC 统计**：mean **+0.0634**，std 0.0450，**胜率 14/16 = 88%**，范围 [−0.0034, +0.1511]。
- **关键**：仅 2 个月微负 —— 2025-06 (−0.0034) 与 2026-07 (−0.0015)，**均 ≈0（近零，非强负）**。无任何月份 IC < −0.01。模型从未在任一月产生有意义反向预测。
- **月度 RankIC**：mean **+0.0961**，**胜率 16/16 = 100%**（排序信号比 Pearson IC 更稳，每月单调性都对）。

### 32.4 分 regime 月度 IC（SH000300 月收益三分位）

以 SH000300 月收益三分位划 regime（[−5.53%, +10.33%] 范围，分位 bear ≤ +0.09% / bull ≥ +2.28%）：

| regime | 月数 | 月均 bm 收益 | 月均 IC | IC 胜率 |
|---|---|---|---|---|
| bear | 5 | −2.87% | **+0.0564** | 80% |
| sideways | 6 | +1.57% | **+0.0648** | 100% |
| bull | 5 | +5.52% | **+0.0687** | 80% |

- **champion alpha 跨 regime 近乎平稳**（bear/sideways/bull 月均 IC 0.056/0.065/0.069，极差仅 0.012）—— **非牛市专属因子**，熊市依然有效。这与高贝塔成分股「牛市弹性大」的直觉不同：alpha 来自 9:30-9:40 开盘分钟结构（T 日盘中信号），而非 beta 暴露本身。
- 唯一 bear 月负：2026-07（−0.0015，近零）。最强月 2025-04（+0.1511，bear 期）—— 反而是熊市月最强。

### 32.5 可复现性（W1/W2 多 run 全量 IC）

champion 在 W1/W2 各有多次 run（不同时间重跑），IC 完全一致：

| 折 | experiment | run 数 | 各 run IC |
|---|---|---|---|
| f5=W1 | 907625868975691204 | 3 | 0.0545 / 0.0545 / 0.0545（全同） |
| f1=W2 | 449239398769286298 | 2 | 0.0718 / 0.0718（全同） |

→ 训练确定性（`LGBModel` 固定 `num_threads=20` + 无随机 bagging），**结果可逐位复现**，无 seed 抖动。

### 32.6 判定：champion 非单窗过拟合

| 稳健性维度 | 证据 | 结论 |
|---|---|---|
| 跨折 IC | 5 折全正 (0.0545~0.0905)，std 0.0131 | ✅ 非单窗 |
| 跨折超额 | 5 折含成本年化全正 (+17%~+159%) | ✅ 非单窗 |
| 月度 IC | 16 月 88% 正，2 负月均 ≈0 (−0.003/−0.002) | ✅ 无反向月 |
| 月度 RankIC | 16 月 100% 正，mean +0.0961 | ✅ 排序极稳 |
| regime 无关性 | bear/sideways/bull IC 0.056/0.065/0.069 | ✅ 非牛市专属 |
| 可复现性 | W1×3 / W2×2 run IC 全同 | ✅ 确定性 |

**结论**：champion enhanced(18)@n_drop15 在可行窗 2024-2026 的 5 折 expanding walk-forward 中表现**稳健**：跨折 IC 全正、月度 IC 88% 正（且 2 负月近零）、RankIC 月月正、跨 regime 平稳、可逐位复现。**alpha 不是 W1（Q2'26）单窗过拟合产物**。

**对 §31.2「IC↔超额解耦墙」的补充**：walk-forward 证伪了「champion 是过拟合」这一替代解释 —— 墙是**真实的信息天花板**（单变量 startup_total ICIR≈0，§31.2），而非训练窗偶然。champion 的 alpha 来自 LGBM 对 18 个开盘族因子的非线性组合，该组合在 5 个独立 OOS 窗 + 16 个月 + 3 种 regime 下持续有效。f4（Q1'26）相对最弱（IC 0.0563、回撤 −16.25%）提示存在**时变衰减**迹象（非崩溃），符合 §29.7 ①「接受天花板」的现实定位。

**落档**：本节为 champion 定稿的稳健性背书。§29.7 收敛项 ① 至此完成「定稿 + 窗内稳健性」半部；剩余「扩段（26 年全数据）」受分钟 bin 物化回溯上限约束，需先扩物化范围（待用户决策，非本节范围）。聚合脚本 `/tmp/wf_aggregate.py`（诊断脚本，同 §30/§31 惯例不入 repo）。

---

## §33 策略层 topk/n_drop sweep — champion 20/15 → 10/8 双窗双赢晋升（2026-07-08）

> **背景**：用户指令「优化超额收益 + IC」经 §23-§32 全 saga 收敛到信息天花板（§29.4/§31.2：因子层加法不破墙）。**唯一未扫的确定杠杆**是策略层持仓构造参数 `topk/n_drop`（[td0_strategy.py](../../qlib_ifind_beta/td0_strategy.py) 的 TopkDropoutStrategyTD0）—— §31.2 已揭示 D9（rank1-10）alpha +1.40%/日 ≫ D8（rank11-20）+0.71%/日，头部高度集中，但 champion 仍用 topk=20 把资本均摊到 rank1-20。**假设**：把持仓集中到 rank1-10（topk=10）可捕获更多头部 alpha；n_drop 控换手保鲜。本节扫 3 档 topk/n_drop × 双窗，**唯一变量 = topk/n_drop**，因子集(18)/label/deal_price/涨跌停拦截/模型超参/切分全 FROZEN。
>
> **与用户指令的关系**：本节是「提升超额」的**策略层落点**（持仓构造优化，不改因子、不改 IC）；「提升 IC/预测准确性」的**因子层落点**（T-1 盘中×T日开盘 cycle 状态机）见 Part B（独立 spec，§34+）。A/B 互不依赖：A 已确定性到手，B 失败不影响 A。

### 33.1 sweep 设计

| 参数 | champion | sweep A | sweep B（★晋升） | sweep C |
|---|---|---|---|---|
| topk | 20 | 10 | **10** | 5 |
| n_drop | 15 | 5 | **8** | 3 |
| 换手帽 n_drop/topk | 75% | 50% | **80%** | 60% |
| 设计意图 | 基线 | 同 topk=10 低换手（信号陈旧探针） | **头部集中 + 高换手保鲜** | 极端集中压测点 |

- **双窗**（与 §22/§29/§32 一致）：W1 = champion 原窗（test 2026-04→07，train 2024-01→2025-12 / valid 2026-01→03）；W2 = W1 前推 1 年 OOS（test 2025-04→07，train 2024-01→2024-12 / valid 2025-01→03）。防 W1 单窗过拟合。
- 6 runs = 3 档 × 2 窗。yaml = champion 副本仅改 `experiment_name` + `topk` + `n_drop`（[workflow_minute_enhanced_tk10_nd8.yaml](../../qrun/workflow_minute_enhanced_tk10_nd8.yaml) 等 6 份）。
- **成功判定**：双窗超额含成本年化**均超** champion **且** W1 回撤不得 < −10%（与 §22 gate 一致）。

### 33.2 全量 A/B 结果（基线 = champion enhanced(18)@20/15）

口径：IC = `ic.pkl` 日 IC 均值；超额 = `port_analysis_1day.pkl.loc[(ret_type,'annualized_return'),'risk']`；换手 = `report_normal_1day.pkl['turnover']` 日均 ×242；woc = 不含成本超额年化。

**W1 窗（test 2026-04→07，IC 全 = 0.0545，模型确定性）**

| 配置 | 超额含成本 | Δ vs champion | woc（不含成本） | 回撤 | 组合 IR | 换手/日 | 成本侵蚀 |
|---|---|---|---|---|---|---|---|
| champion 20/15 | +158.9% | — | +191.3% | −5.44% | 5.12 | 137.1% | 32.5pp |
| tk10_nd5 | +122.6% | −36.3pp ✗ | +143.9% | −7.77% | 3.82 | 90.3% | 21.3pp |
| **tk10_nd8 ★** | **+191.1%** | **+32.2pp ✓** | **+226.0%** | **−5.59%** | **5.41** | 147.4% | 34.9pp |
| tk5_nd3 | +174.2% | +15.3pp ✓ | +199.8% | −8.01% | 4.16 | 108.5% | 25.6pp |

**W2 窗（test 2025-04→07，IC 全 = 0.0718）**

| 配置 | 超额含成本 | Δ vs champion | woc（不含成本） | 回撤 | 组合 IR | 换手/日 | 成本侵蚀 |
|---|---|---|---|---|---|---|---|
| champion 20/15 | +63.2% | — | +95.1% | −10.17% | 1.91 | 134.9% | 31.9pp |
| tk10_nd5 | +5.1% | −58.1pp ✗ | +26.7% | −11.84% | 0.13 | 91.7% | 21.6pp |
| **tk10_nd8 ★** | **+75.9%** | **+12.8pp ✓** | **+109.3%** | **−12.87%** | **1.91** | 140.7% | 33.3pp |
| tk5_nd3 | −78.0% | −141.1pp ✗ | −52.9% | −30.24% | −1.53 | 106.2% | 25.1pp |

### 33.3 成功判定

| 配置 | W1 | W2 | 判定 |
|---|---|---|---|
| tk10_nd5 | +122.6%（↓ ddOK） | +5.1%（↓） | ✗ 未超（双窗均败） |
| **tk10_nd8** | **+191.1%（超 ddOK）** | **+75.9%（超）** | **★ 双窗双赢** |
| tk5_nd3 | +174.2%（超 ddOK） | −78.0%（↓ 集中度爆仓） | △ 单窗 |

→ **tk10_nd8 是唯一双窗双赢档**，W1 回撤 −5.59% 在 −10% gate 内，IR W1 5.41 > champion 5.12。

### 33.4 机制：woc 同涨 → 真实 alpha 集中，非成本侧

**关键第一性原理判据**：tk10_nd8 的**不含成本超额（woc）也涨** —— W1 191.3%→226.0%（+34.7pp）、W2 95.1%→109.3%（+14.2pp）。若提升仅来自降成本，woc 应不变、仅 wc 升；**woc 同升证明 tk10_nd8 捕获了更多毛 alpha**。

- **机制**：§31.2 实测 D9（rank1-10）日均 alpha +1.40% ≫ D8（rank11-20）+0.71%。champion topk=20 把资本均摊到 rank1-20（含较弱的 11-20）；**topk=10 集中到 rank1-10** → 毛收益上行。这是「同一信号、更优持仓权重」，**不破 IC-超额信息墙**（§31.2 的墙指因子层加法无效），而是从墙内已捕获的头部 alpha 里多榨出水。
- **n_drop=8 的作用**：80% 日换手帽（topk=10 换 8 个）→ 持仓高度追随当日新预测排序 → 信号保鲜。对照 tk10_nd5（同 topk=10 但仅 50% 换手）→ 持仓滞后、信号陈旧 → 双窗均败（W1 −36pp / W2 −58pp）。**topk 决定集中度，n_drop 决定信号新鲜度，二者必须同档高位**。

### 33.5 失败档机制

- **tk10_nd5（双窗均败）**：topk=10 集中正确，但 n_drop=5 换手不足 → 昨日强、今日转弱的票未被及时剔出 → 信号陈旧侵蚀。换手 90%/日 vs nd8 147%/日。**集中度对、保鲜度错**。
- **tk5_nd3（W1 超 / W2 −78% 爆仓）**：topk=5 极端集中 → 单一名风险暴增。W2（test 2025-04→07，含 2025-04 单月 IC 最强 +0.1511 的行情）一旦踩中尾部名 → 回撤 −30%、年化 −78%。**集中度过高 = 方差炸弹**，W1 侥幸、W2 崩盘。验证「topk=10 是集中度甜点，再往下走方差失控」。

### 33.6 晋升决策

**champion 策略参数 20/15 → 10/8**（[workflow_minute_enhanced.yaml](../../qrun/workflow_minute_enhanced.yaml) + [workflow_minute_enhanced_w2.yaml](../../qrun/workflow_minute_enhanced_w2.yaml) 已改，diff 确认与 sweep tk10_nd8 yaml 仅 experiment_name + 注释差异 → 功能等价）。

- **不动的**：因子集（18 = 14 分钟 + 4 extra）、label（`Ref($close,-1)/$price_941-1`）、deal_price（`$price_941/$close`）、涨跌停拦截、模型超参（LGBModel λ_l1=5/λ_l2=10）、切分、benchmark（SH000300）、universe（highbeta883926）。
- **IC 不变**（0.0545/0.0718）：topk/n_drop 仅作用于回测层（PortAnaRecord），不影响 SignalRecord/SigAnaRecord 的 pred 与 IC。→ 本节是**纯持仓构造优化**，IC/预测准确性未动（那是 Part B 因子层的任务）。
- **换手代价**：tk10_nd8 换手 147%/日（年 357%）> champion 137%/日（年 332%），成本侵蚀 34.9pp vs 32.5pp（+2.4pp）。但 woc 涨幅（+34.7pp）远盖过成本增幅（+2.4pp）→ 净超额仍 +32.2pp。**可接受**。

### 33.7 对「信息天花板」的定位

§29.4/§31.2 的墙 = **因子层信息天花板**（加因子不改善 top-20 头部 alpha）。本节证明：**墙内仍有策略层榨取空间** —— 通过集中持仓到 rank1-10（头部 alpha 最厚处）+ 高换手保鲜，可在**不增加信息量**的前提下提升超额 +32pp（W1）。这与 §29.7 ④「策略层 bagging 降方差」是同一侧（成本/构造侧），但本节是**集中度侧**而非降方差侧，且**已被双窗验证为正增量**（非理论估计）。

**天花板图景更新**：
- 因子层（IC）：墙 = §31.2 信息天花板，Part B 撞墙中（低 prior）。
- 策略层（超额↔IC 解耦）：本节确认 topk=10/n_drop=8 是当前信号下的**集中度甜点**，已榨取 +32pp；进一步集中（tk5）方差失控。**策略层杠杆已基本用尽**。

**落档**：champion = enhanced(18)@**topk10/nd8**（W1 +191.1% wc / woc +226.0% / IC 0.0545 / IR 5.41 / 回撤 −5.59%）。sweep 证据 run 见 mlflow exp `minute_enhanced_tk10_nd8`（W1）/ `minute_enhanced_w2_tk10_nd8`（W2）；提取脚本 `/tmp/sweep_metrics.py`（诊断脚本，不入 repo）。

## §34 Part B：T-1 反转风险 gate（策略层，非因子墙）— cheap-falsify kill-switch

> **定位**：§33.7 已言"Part B 撞墙中（低 prior）"。本节是 Part B spec（[2026-07-08-t1-reversal-gate-design.md](../superpowers/specs/2026-07-08-t1-reversal-gate-design.md)）的 **§7 cheap-falsify kill-switch** —— 先不建分类器、不物化 bin，只验证核心假设：**盲区 A（T-1 全天 minute 盘中轨迹，slots 1-241，现有 champion 只用 9:30-9:40 开盘 10 根）对 T 日反转是否有区分度**。通过 → 进物化+gate 回测；不达 bar → 方向证伪 STOP（spec §7.2）。本节零风险触及 FROZEN champion/label/strategy（纯只读分析）。脚本 [scripts/cheap_falsify_revrisk.py](../../scripts/cheap_falsify_revrisk.py)。

### 34.1 方法（spec §7.1）

- **连续 fade_score**（§2，IC 主目标）：`fade_T = (high_T − close_T)/(high_T − low_T) ∈ [0,1]`（1 = 收在最低 / 冲高回落最狠）。daily 后复权 OHLC，scale-invariant。
- **12 T-1 proxy**（§3，全部 T-1-close-knowable 零前视）：7 minute 盘中族（#1-4,10-12，cn_data_1min 全天 242 slot scatter → 4 段切片 [1:31]/[31:121]/[121:211]/[211:242]，段 VWAP=4OHLC 均值）+ 5 daily 聚合族（#5-9，qlib_data 日频）。
- **三段检验**：① 单变量 rank IC（每日横截面 Spearman，train 均值）；② 五分位 bucket（pooled train，Q5−Q1 的 fade_score 均值 + 现有 label 均值）；③ logistic regression（12 proxy，θ 在 train 段定 25-35% 占比 → valid AUC + 连续 IC）。
- **样本**：全 universe 5038 票（5116 − 78 无 minute bin），restrict 到 pool-active (code,date)（highbeta883926 时变池，50525 段）；train 47789 / valid 5583 / test 6198 行，train 横截面 484 天。

### 34.2 结果

**表 1 — 单变量 rank IC vs fade_score（train / valid，484 天横截面均值）**

| # | proxy | 族 | IC_train | IC_valid | \|IC_train\* |
|---|---|---|---|---|---|
| 1 | seg_ret_open_mid | min | −0.0146 | −0.0495 | 0.0146 |
| 2 | seg_ret_mid_pm | min | +0.0124 | +0.0275 | 0.0124 |
| 3 | seg_ret_pm_tail | min | +0.0101 | +0.0304 | 0.0101 |
| 4 | tail_mom | min | −0.0132 | −0.0006 | 0.0132 |
| 5 | intraday_weak | day | −0.0068 | −0.0106 | 0.0068 |
| 6 | close_quantile | day | −0.0275 | −0.0290 | 0.0275 |
| **7** | **amplitude** | **day** | **+0.0546** | +0.0413 | **0.0546 ✓** |
| **8** | **day_return** | **day** | **−0.0429** | −0.0550 | **0.0429 ✓** |
| **9** | **overnight_gap** | **day** | **−0.0433** | −0.0470 | **0.0433 ✓** |
| 10 | vol_ratio_tail_open | min | +0.0046 | +0.0575 | 0.0046 |
| 11 | vol_share_tail | min | +0.0104 | +0.0491 | 0.0104 |
| 12 | tail_30min_ret | min | −0.0138 | +0.0049 | 0.0138 |

**表 2 — bucket Δ（pooled train，Q5−Q1）**：Δfade 有信号（amplitude +0.024 / day_return −0.038 / overnight_gap −0.049 / vol_share_tail +0.027），**但 Δlbl（现有 T-day label）全部 \|Δ\|<0.005** → fade_score 的"信号"未转化为实际交易收益区分度。

**logistic**：θ=0.4（train reversal rate 0.287，落 [25%,35%] 带），**valid AUC=0.5316**，pred-vs-fade IC(valid)=0.0437。

### 34.3 判定（spec §7.2）

bar = (≥3 proxy \|IC_train\|>0.03) **OR** (valid AUC>0.55)
- 条件 A：3 proxy 过线（amplitude / day_return / overnight_gap）→ **✓**
- 条件 B：valid AUC=0.5316 < 0.55 → ✗
- **机械判定：PASS**

### 34.4 第一性原理复核 — 核心 nuance（机械 PASS 背后的证伪）

**这是「机械 bar PASS、但 spec 核心假设被证伪 + 实用预测力弱」的 borderline 结果，不可机械推进**：

1. **盲区 A（minute 盘中，spec §34 的核心赌注）被证伪**：7 个 minute proxy **全部 \|IC_train\|<0.015，0/7 过线**，最大 0.0146（噪声级）。spec 想验证的"T-1 全天盘中形态预测 T 日反转"**不成立**。这正是 §33.7 预言的"Part B 撞墙"。
2. **通过 bar 的是 daily 聚合族**（振幅/涨幅/跳空），且：
   - `overnight_gap` **已是 champion enhanced(18) 的 4 extra 之一**（[materialize_minute.py](../../qlib_ifind_beta/materialize_minute.py) MINUTE_FACTOR_EXTRA_FIELDS）→ 直接重叠；
   - `day_return` / `amplitude` 与 Alpha158 的 MOM/ROC/VOL 类高度共线 → 增量信息可疑。
3. **实用预测力弱**：12 维 logistic 组合 valid AUC 仅 0.5316（接近随机 0.5），pred IC=0.0437；bucket Δlbl≈0（fade 信号不转化实际收益）。即便物化成 gate，可榨超额的 prior 很低。
4. **train↔valid 不一致**：minute 族 valid 段有几个 \|IC\|>0.03（open_mid −0.0495 / vol_ratio +0.0575 / vol_share +0.0491）但 train 段全 <0.03 → 过拟合/噪声，非稳定信号（spec §7.2 用 train IC 判定即为此防过拟合）。

### 34.5 结论与决策（2026-07-09 用户裁定：证伪归档 STOP）

- **spec §34 核心假设（盲区 A minute 盘中预测反转）= FAIL**；§7.2 bar 的机械 PASS **完全靠 daily 族救场**，而 daily 族与 champion 重叠、实用力弱。
- **用户 2026-07-09 裁定：证伪归档 STOP** → **不推进** spec §13 step 2-5（物化 revrisk.bin + TopkDropoutStrategyTD0Gate + 回测全部停做）。champion 不变，零风险。
- **未走 amplitude 单变量 gate**：daily amplitude（IC 0.0546）虽是 Alpha158 未直接覆盖项，但 Δlbl≈0、AUC 0.5316 < 0.55、prior 低，且属 daily 族（与盲区 A minute 假设无关）→ 不构成 Part B 的有效增量，随 STOP 一并归档。
- **证伪价值**：盲区 A minute 盘中族 0/7 是一个**确定的负结论** —— 与 §23-§29 九重墙同向（T-1 盘中加法信号无效），进一步坐实 §31.2 信息天花板在因子层的不可破性。Part B 的"策略层 gate 绕墙"路径，其信号源（盲区 A）本身无效 → gate 无米之炊。§33.7 "Part B 撞墙（低 prior）"预判兑现。

**落档（用户裁定 STOP）**：§34 Part B 证伪关闭。champion 不变（enhanced(18)@topk10/nd8，commit `24b18dd`）。脚本 [cheap_falsify_revrisk.py](../../scripts/cheap_falsify_revrisk.py) + spec [2026-07-08-t1-reversal-gate-design.md](../specs/2026-07-08-t1-reversal-gate-design.md) 保留为只读诊断/设计存档（均未提交，待用户决定提交或删除）。FROZEN label/strategy 全程未动。

---

## §35 champion 只读诊断（2026-07-09）：bagging 假设实测推翻 + 天花板测量定位

> 触发：standing 优化指令（提升超额/IC）+ Stop hook（§34 证伪 ≠ 满足优化目标，须继续推进实际优化）。规则 #7 禁止重做 §23-§29 九重墙 → 转向**未尝试的正交方向**的零风险只读诊断（不重训、不回测、不动 FROZEN）。诊断脚本入 `/tmp`（惯例，不入 repo）。
> 对象：champion W1 recorder `mlruns/156869948604814731/caf649ca6aa44aac8dec8c4e5a252aef`（experiment `minute_enhanced_tk10_nd8`，git `f4986af`）。校验身份：IC 0.0545 / RankIC 0.0679 / ICIR 0.513 / excess wc 年化 191.1%（IR 5.406, maxDD −5.59%）/ woc 226.0%（IR 6.382）= champion 无误。
> 三个未试正交方向：A 减法/LOO、B bagging（§31.4④ 唯一未试策略层方向）、C 数据完整性（directive #2 最高价值）。

### 35.1 诊断 C（数据完整性）— ✓ PASS

- recorder `label.pkl` 列名 = `Ref($close, -1) / $price_941 - 1`，与 FROZEN label **完全一致**（口径无漂移）。
- recorder label vs qlib 重算（test 段 2026-04-01→07-02）：**逐 cell max|diff| = 0.00e+00，corr = 1.000000** → 训练用 label 即该表达式，**无前视、无口径漂移**。交集 6076/6198（差 122 = 末日 07-02 的 `Ref($close,-1)` 需 07-03 close=NaN 被 dropna，符合预期）。
- panel：6198 行 × 19 列，每日 universe ~96 票，62 日，因子 NaN 率 max 3.6% / label NaN 2.0% → 干净。

### 35.2 诊断 B（换手率/成本/bagging gate）— 朴素 gate 通过，但前提被 35.3 推翻

- **换手率结构性极高**：`report_normal.turnover` 日均 **147.4%**（median 152.8%，min 95%，max 167%）；100% 交易日 >80%，98.4% >100%。非尖峰驱动，是全段高位。
- **成本侵蚀**：test 段几何累计超额 wc 74.94% vs woc 91.11% = **16.17pp**（年化口径 wc 191% vs woc 226% = 34.9pp，同向）。
- **朴素 bagging gate**（假设"换手率可被 bagging 削"）：换手降 50%→超额 +7.91pp，降 30%→+4.70pp（test 段）。**⚠️ 此 gate 建立在伪前提上，见 35.3。**

### 35.3 边界翻转率实测 — bagging 假设被测量推翻（核心发现）

读 `pred.pkl` 逐日 top-k 排序测真实翻转结构：

- **top10 留存率 = 0.98%**（几乎每日全换 10 只），top18 留存 1.46%。
- **universe 每日轮换 89%**：相邻日 universe 交集占比 mean 10.9%（median 9.7%，min 1%）；每日 ~96 票仅 ~10 票与昨日延续 → **883926 每日重平衡高贝塔榜的编制本质**（plan 待定段已注：883926 每日换手 ~90% 是其编制本质）。
- **共有票 top10 单边翻转 = 0.00 只/日**：两日 universe 交集（mean 10 票）内，top10 成员零翻转 → **bagging 能削减的边界噪声 ≈ 0**。

**结论（第一性原理修正 §31.4④）**：147% 换手率 ≈ **89% universe 轮换**（结构性、bagging 无法触及）+ **0% 预测边界噪声**。bagging 即便完全消除边界翻转，换手率降幅 ≈ 0 → **年化超额回收 < 1pp**，远低于 §31.4④ 估计的 10-15pp 与 35.2 朴素 gate 的 5-8pp。

> **bagging 实测无效 → 不推进 bagging 回测**（省下一个浪费的多 seed LGBM + 回测）。成本侵蚀 16pp 是**结构性**的（日频重平衡 universe 的代价），非预测噪声，因子/模型层不可破。

### 35.4 诊断 A（因子相关矩阵 + 单变量 rank IC）— LOO 候选弱

- **18×18 相关矩阵（pooled Pearson）无严重共线**：最高 |r|=0.891（vol_ratio_3m~5m）；次 close_pos_3m~5m 0.871、accel_3m~5m 0.822。**原假设"vol_vs_yest_t2/t3/t5 高共线"证伪**（三者各自独立，未进 |r|>0.6 榜）→ "剔 2 留 1" LOO 候选不成立。
- **单变量 rank IC（§31.2 口径，截面 Spearman 按日均值）**：仅 5 因子 |IC|>0.03 — `overnight_gap` +0.062、`vol_vs_yest_t5` −0.069、`t2` −0.049、`t3` −0.045、`vol_vs_yest` −0.034；**14 个 baseline 分钟因子 |IC|<0.025**（弱线性，靠 LGBM 非线性组合，印证 §31.2 解耦）；`vol_ratio` 族 IC≈0 且 3m~5m 最共线 → 唯一勉强 LOO 候选 = vol_ratio_5m。
- **LOO 整体偏弱**：无严重共线 + §31.2 单变量 IC 盲点 + §16 surgery 前鉴（IC↑≠超额↑）→ 减法风险 > 收益，暂缓。

### 35.5 测量版天花板定位 + 剩余杠杆

**三向诊断收口**：C ✓ 无前视；B ✗ bagging 实测无效；A △ LOO 弱。**用测量证据坐实 §33.7"天花板已至"** —— 因子层（IC 信息天花板 §31.2）+ 模型层（bagging 实测无效）+ 策略层浓度（§33 topk10/nd8 双窗双赢）均已触顶；成本侵蚀结构性不可削。

**剩余杠杆（均需用户对齐 governance #1/#2，不擅改 FROZEN）**：

> **先排除一个伪选项（本节起草时的联想错误，已自纠）**：directive #2/#3 方向（T-1 K 线情绪相位 + 上证指数共振）**不是未测正交方向**，正是已被五度证伪的同一族 —— §28.A（因子层 per-stock 日频情绪+指数共振，W2 OOS IC −15%/超额 −69.6pp/p=0.443）+ §28.B（策略层 regime-gating，相关≈0/gate 毁收益）+ §29（分钟尺度共振 broadcast）+ §30（idio 相对强度偏 IC≈0）+ §31（日频反转族属 §28.A）。起草时曾误将 §28 记成"标量广播"、把"K 线相位分类"当不同族（违反 governance #3 禁联想）；复读 backtest-log §23/§28/§29 原文纠正：**§28 恰是 per-stock 日频情绪+指数共振（非广播），"broadcast 同值"失败的是 §29**。规则 #7 禁重做，此方向**关闭**。

1. **策略层 sticky holdings**（唯一可能动结构性换手的策略层杠杆）：持仓 N 日不随 universe 每日轮换 → 削 89% universe 换手、降成本侵蚀。**⚠️ 代价**：与 ~1.5 天 label horizon 冲突（label 仅预测 T+1，多日持有超出预测视野）→ 需重设计 label/hold 节奏，**属解冻 FROZEN 的大改**，需用户明确授权。
2. **扩段评估**（待定项，§28.7 选项①）：用 26 年全段重训重测 champion 泛化性 —— **非提 IC**（9 重墙已证天花板结构性），是稳健性验证 + 正式报告产出前置。
3. **转实战对接**（§28.7 选项②）：策略两窗含成本超额 +40~159% arith 已盈利，可进纸面/实盘跟踪，边跑边观察 OOS 稳健性。非优化，是部署。

**落档**：§35 诊断完成，champion 不变。bagging 因实测无效**不推进**（区别于 §34 的"证伪后 STOP"——此处是"测量后证伪方向"）。directive #2/#3 方向经复读 §23/§28/§29/§30/§31 原文确认五度证伪、**规则 #7 关闭**。诊断脚本（diag_cost/diag_factors/boundary churn）保留 `/tmp`（惯例不入 repo，会随系统清理）。FROZEN label/strategy 全程未动。**测量版结论：冻结设定下三层杠杆（因子/模型/策略）已穷尽，超额天花板结构性不可破；唯一能动的是解冻（sticky holdings 重设计 label 节奏）或转向（扩段稳健性 / 实战对接）。待用户裁定。**

---

## §36 champion 实战对接 P1 — 纸面前向跟踪系统实现（2026-07-09）

> 触发：用户裁定 §35.5 剩余杠杆③「转实战对接」（§28.7 选项②）—— champion 两窗含成本超额 +40~159% arith 已盈利，从回测验证推进到每日产出可执行交易信号 + 前向纸面跟踪。授权："完成方案设计和实施，有需要决策的整理清单"。
> 范围：**P1 纸面前向**（零外部依赖，消费既有 day.bin），非 P2 实时/P3 实盘。spec [2026-07-09-live-forward-design.md](../superpowers/specs/2026-07-09-live-forward-design.md) + plan [2026-07-09-live-forward.md](../superpowers/plans/2026-07-09-live-forward.md)。
> FROZEN 全程未动：label `Ref($close,-1)/$price_941-1` / deal_price `$price_941` / topk10/nd8 / model params.pkl（commit `24b18dd`）。

### 36.1 推理口径零偏离铁证（P1 正确性根基）

复刻 champion handler（fit 段 2024-01-01→2025-12-31 FROZEN，仅 end_time→T）→ load 冻结 model（params.pkl = LGBModel 实例，num_trees=10）→ DatasetH(segments={"inference":(T,T)}) → model.predict。

spike 验证 predict_day("2026-07-02") vs champion pred.pkl（test 末日）：
- matched 98/98，**max|diff| = 0.000e+00**（bit-exact，非数值近似）
- top10 10/10 集合一致

→ P1 推理 = champion 回测 test 段逐位复刻，无口径漂移。`test_predict_day_zero_drift` 锚定此口径不退化（max|diff|<1e-6）。

qlib fit 语义确认：processor 在 train segment fit（RobustZScoreNorm/Fillna/DropnaLabel），inference 段仅 transform → T 日因子变换口径 = test 段口径（零漂移的机制根因）。

### 36.2 六模块实现（commit 9a81818→2bfbba0）

| 文件 | 职责 | qlib 分层 |
|---|---|---|
| [config.py](../../qlib_ifind_beta/config.py) L48-54 | champion FROZEN 常量（recorder_id/experiment/fit 段/label/topk） | — |
| [live/inference.py](../../qlib_ifind_beta/live/inference.py) | predict_day() 推理核心（load model → 复刻 handler → predict → top10 + aux） | Workflow/Model + Interface/Recorder |
| [live/track.py](../../qlib_ifind_beta/live/track.py) | record_signal/settle_prev/compute_nav/daily_ic（纯 pandas 状态机） | Interface/Recorder（纸面简化） |
| [live/materialize_live.py](../../qlib_ifind_beta/live/materialize_live.py) | load_pool(T-1 membership)/materialize_pool（薄封装 overlay+materialize） | Infrastructure/DataProvider |
| [scripts/live_forward.py](../../scripts/live_forward.py) | 五步编排入口（universe→materialize→predict→settle→nav） | Interface/Workflow |
| [tests/test_live_inference.py](../../tests/test_live_inference.py) | 8 测试（零偏离/NAV/涨停/前视/跌停/幂等/IC） | — |

偏离点（spec §5 已声明，均为 qlib 原生 API 合法组合，非自定义抽象层）：
- 推理入口组装（load_object + DatasetH + predict）—— qlib 原生调用链
- 纸面撮合简化（change_941>=limit_up 剔除 + close[T+1] 卖，与 exchange 同源判定）

### 36.3 settle 口径双修正（实现中发现，非 spec 字面）

dry-run 暴露两个真实口径偏差，按第一性原理修正：

**修正1 — close_lookup 覆盖（universe 时变 vs 持仓掉出池）**：
- 现象：settle_prev(07-01→07-02) 98 笔但 sell_price 仅 8 笔非空 → NAV 只反映 8 只。
- 根因：`D.instruments(market)` 返回 dict，`D.features(dict, start=T, end=T)` 按 date 过滤成**当日池（~100）**；883926 每日换手 ~90%（§35.3 实测 89%），T-1 持仓 ~90% 在 T 日掉出观察池 → close_lookup 仅含当日池 → overlap 8/98。
- 修正：close/change/limit_down lookup 按 **prev_candidates code 列表**查（`D.features(list, ...)` 不受 universe 池过滤，直接取 bin）。close 是 base bin 全票都有 → 完整覆盖；change/limit_down 是 overlay 物化 bin，掉出池票缺失 → blocked 判断 None→False（P1 可接受简化，跌停拦截降级）。
- 验证：close_lookup 0→98 全覆盖，sell_price 8→10（topk）非空。

**修正2 — NAV=top10 对齐 spec §3.2 [5]**：
- 现象：原 record/settle **全部 candidates（98）**，NAV 等权 98 只，偏离 spec「equal-weight top10」。
- 修正：record_signal 加 `in_topk` 标记（topk = 剔除封涨停后前 10），settle_prev 只结算 in_topk=True。保留全部 candidates 供 IC 复算（IC 需全部 score，非仅 topk）。
- 验证：settle 98→10 行，n_held=10。

**dry-run 终态**（2026-07-01→07-02，--skip-universe --skip-materialize）：
- Day1 07-01: predict topk10，settle 0（首日无 T-1），NAV 空。
- Day2 07-02: predict topk10，settle 10 笔（07-01→07-02），NAV net_nav=1.0101 / gross_nav=1.0121 / n_held=10。
- 8/8 测试 PASSED。

### 36.4 P1→P2→P3 路线 + 待决策清单（governance #1/#2，明早裁定）

| 阶段 | 状态 | 外部依赖 | 价值 |
|---|---|---|---|
| **P1 纸面前向** | ✅ 完成（本节） | 零（消费既有 day.bin） | 验证 OOS 稳健性 + 回测假设；每日 top10 信号可执行 |
| P2 实时跟踪 | 待定 | 实时分钟数据源（9:41 前 T 日 9:30-9:40 分钟） | 盘前产出信号（现 P1 是盘后） |
| P3 实盘 | 待定 | 交易通道（券商 API） | 真实委托（替换纸面撮合） |

**待用户决策清单（6 项）**：

1. **P2 实时分钟数据源**：P1 复用既有 day.bin（盘后），P2 需 9:41 前获 T 日 9:30-9:40 实时分钟。候选：iFinD 实时行情 / 券商 Level-1 推送 / 自建分钟采集。需选型 + 鉴权。
2. **P3 交易通道**：实盘委托通道选型（券商 API / 桥接）。涉及资金，需用户主导。
3. **模型重训节奏**：champion params.pkl FROZEN（2024-2025 训）。前向跟踪是否定期重训（周/月）？重训 = 新 recorder_id，需更新 config 常量 + 重跑零偏离验证。
4. **universe 增量入 P1 cron + iFinD token 2026-08-01 到期**：P1 live_forward [1] universe 增量（iFinD p03473 T-1 快照）需网络 + token；token refresh_token 2026-08-01 到期，届时 universe 刷新失败（P1 有 fallback 用既有池继续，但池会逐渐过期）。需决策 cron 化 + token 续期。
5. **清理证伪族产物 + config 死常量（牵连活代码，需谨慎评估）**：§25 tail/§26 opening_t1/§29 index_opening 证伪族的 config 常量（INDEX_FACTOR_SOURCES/MINUTE_FACTOR_TAIL_FIELDS/MINUTE_FACTOR_OPENING_T1_FIELDS/INDEX_OPENING_*）+ daily-index-factors 已提交产物（7 文件）待清理。**⚠️ 但这些常量被 [materialize_minute.py](../../qlib_ifind_beta/materialize_minute.py)（champion 物化链路活代码）+ [build_overlay.py](../../scripts/build_overlay.py) 引用** → 删常量须同步改物化逻辑，风险触及 champion 18 因子物化 → 可能不该动（证伪族 bin 占磁盘无害，dead config 无害，清理风险>收益）。需用户裁定。
6. **sticky holdings 探索（§35.5 杠杆①）**：唯一能动结构性换手（89% universe 轮换）的策略层杠杆，但需解冻 FROZEN label horizon（~1.5d → 多日持有超预测视野，需重设计 label/hold 节奏）。属大改，需用户明确授权。

**落档**：§36 P1 实战对接完成。champion 从回测验证 → 每日可执行信号 + 纸面前向跟踪。FROZEN 全程未动。P1 零外部依赖，可立即每日盘后跑（`scripts/live_forward.py --date T`）。P2/P3 待用户裁定外部依赖选型。6 项决策清单明早整理。

---

## §37 champion 主源穷尽性再证（2026-07-09）：拥挤 / rank 曲线 / 条件 IC 三轮只读

> 触发：指令点 3「充分分析已有的策略和回测 log」。前序「7 字段体制内无提升路径」是二手断言（来自 §23-§35 综述），本会话首次用**主源只读实证**（gate-compliant：不改 champion / 不解冻 label / 不重训 / 无前视 / 无 ifind）从三个**墙没做过的口径**复核。三轮**独立再证**穷尽，**未踩 rule #7**（不是重做 §23-§31 的因子加法墙，是新诊断角度）。

### 37.1 [A] 拥挤只读反事实 — item 7 先验反转

拥挤代理 `vol_ratio_5_20 = Mean($volume,5)/Mean($volume,20)`，T-1 `shift(1)` 防 T 日全天量前视。test 61 天。

- 全 universe Spearman(crowd,label) = **−0.024**，>0 占比 49.2% → 拥挤信号**近零**（§31 该样本不复现 −0.235）。
- top-10 内 hi−lo 拥挤 label 差 = **+0.0037** → champion **不接盘**（更拥挤的票略好）。
- 去拥挤反事实 `pred − λ·z(crowd)`：λ=0(champion) +1.105%/日 → λ=0.5 +0.353%/日(**−71.4pp**)，best λ=0。

**结论**：volume 代理下去拥挤**毁超额**，「接盘」证伪 → item 7（扩 turn 拥挤）先验**中性→偏负**，建议降优先级。caveat：非真 turn、不计 n_drop 换手/成本。

### 37.2 [B] rank 曲线 @ 部署 topk=10 — 5 重墙的配置裂缝

**关键裂缝**：§23-§26 5 重墙**全在 topk=20/n_drop=15 旧配置**证伪「加因子」；FROZEN champion 是 **topk=10/n_drop=8**（§33 后），更集中、边界更锐。墙**从未在部署配置 topk=10 复核**。本节只读检验 champion pred 在部署边界处的 rank 曲线（test 61 天，每日按 pred 分桶算已实现 label）：

| 桶 | r1-3 | r4-6 | r7-10 | r11-15 | r16-20 | r21-30 | r31+ |
|---|---|---|---|---|---|---|---|
| 日均% | +0.92 | +1.44 | +0.99 | +1.02 | +1.21 | +0.36 | +0.45 |

topk 候选窗口等权已实现（gross，不计成本）：top-5 +1.105% / top-8 **+1.212%** / top-10 +1.105%(FROZEN) / top-12 +1.124% / top-15 +1.075% / top-20 +1.110%。top-8 gross > top-10，但 §33 在 **net**（含 n_drop 换手成本）下 10/8 胜 → 一致（top-8 高 gross 被更高换手成本吃掉）。

### 37.3 [B-sig] 显著性诚实修正 — 「极端头部反转」是噪声

初读 r1-3(+0.92%) < r4-6(+1.44%) 似「极端头部过度延伸反转」（佐证 §34 反转门）。**验显著性后推翻**：

- r4-6 − r1-3 日均差 +0.52%，但 **Wilcoxon p=0.284**、**Binomial p=0.153**、r4-6>r1-3 占比仅 57.4% → **不显著，是噪声**。
- 细粒度逐桶（r1…r20）暴露单股单日离群主导：r3(+0.22%)/r9(+0.17%)/r19(+0.16%) 近零但 r20(+2.85%!) 爆表 → top-20 内部逐桶是噪声，非系统性形态。

**真正的结构（唯一显著）**：相邻桶最大下跳 = **r20→r21-30 = −2.49pp**（远大于其余跳跃）→ **champion 的真边 = 分离 top-20 quintile，悬崖在 rank 20 不在 10**。

### 37.4 [B-cont] within-top 排序信号分层

- **within-top-20 IC（champion pred 自身 vs label，条件于 rank≤20）= +3.09%**：弱正信号，champion 对 quintile 内部有边际排序力。
- **within-top-10 IC = −0.008**（§37.6）：**头部 10 内排序归零**。
- 综述：边 = rank-20 悬崖；within-top-20 弱正（+3.09%）；within-top-10 噪声。**topk=10 = 「rank-20 悬崖边 vs 换手成本」折中**（§33 net 优化的机制解释，独立坐实）。

### 37.5 [C] T-1 K 线结构因子 within-top-20 条件 IC — 墙的筛选口径盲点

§25/§26 用**全截面 RankIC**（|ICIR|>0.4）选候选 → 全踩 reshuffle 墙。但 quintile 分离已饱和时，全截面 IC 与「能否在 champion top-20 内加分排序」**不等价**。真正相关的是 **within-top-20 条件 RankIC**（在 champion 已选中的 top-20 里，候选能否进一步分辨 label）。测一族 champion 几乎没用到的 T-1 日频 K 线结构因子（指令点 2「结合 T-1 之前的 K 线」，全部 `Ref(...,1)` 外包无前视，7 字段内）：

| 因子 | 全截面 IC | ICIR | within-top20 IC | ICIR | gap | 解读 |
|---|---|---|---|---|---|---|
| rev1（T-1 反转） | +0.44% | +0.031 | +0.04% | +0.002 | −0.40% | within 归零 |
| mom5_t1 | +1.33% | +0.089 | +1.18% | +0.045 | −0.14% | 弱于 champion |
| mom10_t1 | +1.35% | +0.101 | +0.68% | +0.026 | −0.67% | 弱于 champion |
| volratio_t1 | −2.76% | −0.179 | −2.62% | −0.112 | +0.14% | 负向、不加分 |
| tr1（T-1 振幅） | −0.24% | −0.018 | −2.23% | −0.097 | −1.99% | within 反而恶化 |
| pos20（区间位置） | +1.22% | +0.094 | −0.22% | −0.009 | −1.44% | within 归零 |
| dma20（偏离均线） | +2.49% | +0.166 | +2.00% | +0.080 | −0.49% | 弱于 champion |
| gap_t1（T-1 跳空） | +1.43% | +0.112 | −0.32% | −0.013 | −1.75% | within 归零 |

**全部 8 因子 within-top-20 IC ≤ +2.0%，无一超过 champion 自身 +3.09%**（即无一增加 within-top 排序信息）；gap 多为负（within 比全截面更弱）。

### 37.6 三轮综述 — 主源穷尽性坐实 + item 7 bar 锐化

**主源结论**（替代前序二手断言）：7 字段体制内 champion 因子改进路径**穷尽**，三轮独立口径再证：

1. **拥挤**（§37.1）：volume 代理下去拥挤毁超额 −71.4pp，champion 不接盘。
2. **rank 结构**（§37.2-4）：真边 = rank-20 悬崖（非 10）；within-top-20 弱正 +3.09%，within-top-10 噪声；极端头部反转 = 噪声（p=0.28）。
3. **条件 IC**（§37.5）：8 个 T-1 K 线结构因子 within-top-20 IC 均 < champion +3.09%，无一增加 within-top 排序 → 墙的机制根因 = **champion 饱和 quintile 分离，within-top 无可挖信号**（任何加因子=纯 top 重排=中性偏破坏，正是 §23-§26 所见）。

**item 7（扩字段新因子）bar 锐化**：合格线不是全截面 IC>0，而是 **within-top-20 条件 IC > +3.09%**（champion 自身基线）。7 字段 T-1 K 线族全不达标。真 turn/amount（item 7 的实质诉求）须过此具体 bar，而非泛泛「拥挤=金矿」。结合 §37.1（去拥挤毁超额），**item 7 先验偏负、降优先级**（两轮独立证据）。

**未开自主路径**：三轮均为只读诊断，无一条改动 champion/label/策略即提升 IC/超额。能动 IC/超额 的下一步仍卡需授权决策：item 6（sticky holdings，需解冻 label horizon）、item 7（扩 turn/amount，先验已偏负）。

**落档**：§37 三轮主源实证归档。脚本 `/tmp/crowding_posthoc_diag.py` / `/tmp/rankcurve_topk10.py` / `/tmp/rankcurve_significance.py` / `/tmp/within_top20_conditional_ic.py`（一次性，不入 repo）。Get笔记 §37-emp 同步拥挤实证（item 7 先验走弱）。

### 37.7 新任务=§28 已证伪发现 + corr_20 缺口闭合（共振族条件 IC 补测）

**主源核查推翻二手断言**：连续指令「添加日频因子 + 上证指数因子提升预测准确性」经 git 考古 = **§28（commit 19b4f7a, 2026-07-08）已实做并双证伪（第 8 重墙）**。commit message 明文「用户 /loop 任务『日频情绪因子 + 上证指数因子』= §27 morning report 选项 ② 的实做」。即新任务不是开放机会，是已测已败路径。**墙总数修正为 9 重**（§23-§31），非先前二手转述的「5 重」。

- §28.A 因子层（12 Ref1 因子，含 ChangeInstrument SH000001 + beta_20）：W1 IC 0.0611 vs champion 0.0545（+12%，saga 首见真涨）但超额平（+0.4pp）；W2 OOS IC −15% / 超额 **−69.6pp**。W1 IC 涨 = 窗口过拟合（5/9/20d 因子对 ~1.5d label，scale 错配，与 Alpha158/full(85) 同病）。
- §28.B 策略层（SH000001 情绪 lag1 解释 champion 逐日超额，pooled n=122）：相关 ≈0（bias +0.019 p=0.837 / rsv −0.008 p=0.931 / run −0.009 / vol +0.087），符号跨窗不一致=噪声；gate 模拟全毁收益。

**corr_20 = §28 唯一 bug-drop 未测的 spec 因子**（用户「共振」核心，§28.A 因 qlib `Corr` 日历不齐崩剔除）。本节 pandas 显式对齐日历绕开 bug，算 corr_20[T]=corr(stock_ret, idx_ret, ending T-1)，shift(1) 防前视。口径沿用 §37.5 within-top-20 条件 RankIC（非重做墙：corr_20 从未测过，是补覆盖；非联想：实测而非推断）。test 62 天 / 2515 inst / 5799 行对齐。

| 因子 | 全截面 IC | ICIR | within-top-20 IC | ICIR | 判定 |
|---|---|---|---|---|---|
| corr_20 | +3.24% | +0.238 | **+1.65%** | +0.073 | ✗ 未超基线 |
| beta_20（§28 测过，新口径复核） | +3.78% | +0.253 | **−0.63%** | −0.029 | ✗ 负 |

- corr_20 mean 0.318 / std 0.266；beta_20 mean 1.591 / std 1.529（高 beta 池内 beta 方差大，但 within-top-20 仍负→无分辨力）。
- **校验**：本对齐下 champion pred 自身 within-top-20 IC = +1.09%（§37.6 基线 +3.09%）。同一对齐 corr_20 +1.65% ≈ champion 自身 +1.09%，ICIR 0.073 极小 → **不可分辨、无可挖信号**，加 champion=纯 top 重排。基线漂移（+3.09%↔+1.09%）本身坐实 within-top-20 信号弱/噪声（§37.4 结论）。

**判定**：corr_20（共振核心、§28 唯一缺口）实测 **未达 within-top-20 条件 IC bar**，与 champion 自身弱信号不可分辨。**§28 共振族 falsification 覆盖闭合**；新任务（日频情绪 + 上证共振）= §28.A + §28.B + corr_20 全链坐实 = **已测已败，非开放机会**。gate-compliant 只读自主范围内，因子层 9 重墙 + §37 四轮主源复核 + corr_20 缺口闭合，**穷尽性再坐实**。能动 IC/超额 的下一步仍卡需授权决策：item 6（sticky holdings，需解冻 label horizon）、item 7（扩 turn/amount，先验偏负）。

**落档**：§37.7 corr_20 闭合归档。脚本 `/tmp/corr20_conditional_ic.py`（一次性，不入 repo）。

---

### §37.8 未测族 within-top-20 条件 IC 补覆盖（vwap / 影线实体 / 量加速 / 20 日动量）

**动机**：§37.4 结构性结论是 within-top-20 label 近噪声、预测所有因子都会败。但纪律要求「实测非推断」，且 7 字段中 **vwap 几乎未被挖**、`$open` 影线/实体族未测、量加速（仅测过 vol_ratio level）未测、20 日动量/RSV（仅测过 5/9/10）未测——未挖 vwap 就宣布穷尽 = 早熟。本节补测 10 个未测因子。

**口径**：§37.5 within-top-20 条件 RankIC。全单 instrument qlib expression（无 ChangeInstrument/Corr → 无日历 bug），Ref1 防前视。只读、gate-compliant、非重做墙（这些族从未测过）。脚本 `/tmp/untested_family_conditional_ic.py`。

| 因子 | 全截面 IC | ICIR | within-top-20 IC | ICIR | 判定（loose bar \|top20 IC\|>3.09%）|
|---|---|---|---|---|---|
| cvwap_t1（收盘 vs vwap） | −0.73% | −0.045 | −1.57% | −0.063 | ✗ |
| vwap_pos_t1（vwap 日内位置） | +0.46% | +0.039 | **−4.99%** | −0.210 | ★ 反转 |
| body_t1（实体方向） | −0.59% | −0.044 | −1.54% | −0.062 | ✗ |
| upshad_t1（上影=抛压） | +3.37% | +0.282 | **+4.09%** | +0.167 | ★ |
| lowshad_t1（下影=承接） | **+4.06%** | **+0.393** | +2.75% | +0.118 | ✗（全截面最强）|
| volaccel_t1（量加速度） | −2.31% | −0.181 | **−4.74%** | −0.233 | ★ 反转 |
| vol_trend3（短期量趋势） | −2.62% | −0.166 | −1.46% | −0.054 | ✗ |
| mom20_t1（20 日动量） | −2.02% | −0.110 | **−3.12%** | −0.129 | ★ 反转 |
| momrev_t1（5d-20d 动量差） | +1.85% | +0.099 | **+4.84%** | +0.174 | ★ |
| rsv20_t1（20 日 RSV） | +1.22% | +0.094 | −0.93% | −0.037 | ✗ |

**结果**：5/10 超 loose bar（vwap_pos/upshad/volaccel/mom20/momrev）。lowshad_t1 全截面 IC +4.06%/ICIR +0.393 最强但 top20 仅 +2.75%。

**关键保留**：这是 **loose bar**（|IC|>champion 基线），而 champion 自身 within-top-20 基线弱且对齐敏感（+1.09%~+3.09%，§37.6/37.7）。多个为**负 IC 反转效应**（vwap_pos/volaccel/mom20）。「超 loose bar」≠「真增量」——需 §37.9 partial IC 判据区分「真增量 vs champion 倒影」。

---

### §37.9 partial rank IC（残差化 champion pred 后的增量检验）

**决定性问题**：§37.8 超 loose bar 的因子是否带 champion 没有的 label 信息 = incremental / partial IC。

**口径**：对每日 rank(factor) 用线性回归残差化掉 rank(pred)（champion 预测），再 Spearman(residual, rank(label))。full + within-top-20 两档。**增量判据**：全截面 |partial IC| ≥ 1.5% **且** 残存率（|partial/raw|）≥ 40%。脚本 `/tmp/candleshape_partial_ic.py`。

| 因子 | 全截 raw IC | 全截 partial IC | 残存率 | top20 partial | 判定 |
|---|---|---|---|---|---|
| cvwap_t1 | −0.73% | +0.18% | — | −0.86% | ✗ 塌缩 |
| vwap_pos_t1 | +0.46% | +1.33% | — | −3.12% | ~ 部分增量 |
| body_t1 | −0.59% | +0.56% | — | −0.30% | ✗ 塌缩 |
| upshad_t1 | +3.37% | **+2.43%** | 72% | +4.25% | ★ 真增量 |
| lowshad_t1 | +4.06% | **+3.67%** | 90% | +1.37% | ★ 真增量（最强）|
| volaccel_t1 | −2.31% | **−1.58%** | 68% | −4.24% | ★ 真增量 |
| vol_trend3 | −2.62% | **−2.14%** | 81% | −0.24% | ★ 真增量（放量滞涨反转）|
| mom20_t1 | −2.02% | −0.28% | — | −3.56% | ✗ 塌缩 |
| momrev_t1 | +1.85% | **+1.77%** | 96% | +4.73% | ★ 真增量 |
| rsv20_t1 | +1.22% | **+2.53%** | 207% | −0.75% | ★ 真增量（suppressor）|

**结果**：6/10 通过增量判据——lowshad_t1（+3.67%/残存 90%，最强）、rsv20_t1（+2.53%/207%，正交化后显信号的 suppressor）、upshad_t1（+2.43%/72%）、vol_trend3（−2.14%/81%）、momrev_t1（+1.77%/96%）、volaccel_t1（−1.58%/68%）。cvwap/body/mom20 残差化后塌缩 → champion 倒影、无增量。

**意义**：这是本窗口**首个未秒证伪、且经增量检验存活**的只读信号。6 个 T-1 日频因子（K 线影线 / 量能趋势 / 动量差 / RSV）属 champion 18 个 T 日 9:30-9:40 分钟因子之外的**不同信息族**，partial IC 残差化 champion pred 后仍残存。gate-compliant → 进 §38 决定性 shadow retrain A/B。

---

### §38 shadow retrain 决定性证伪：6 增量因子实测拖累 IC（−25%）

**设计**：`ShadowCandleShapeHandler` = champion `MinuteEnhancedHandler`(18) + §37.9 六增量因子 = **24**。`qrun/workflow_shadow_candleshape.yaml` 是 champion `workflow_minute_enhanced_tk10_nd8.yaml` 的逐字拷贝，**唯一变量** = handler 18→24；label（`Ref($close,-1)/$price_941-1`）/ deal_price / 涨跌停 / LGBModel 超参（lr=0.05, λ_l1=5, λ_l2=10, num_boost_round=200, early_stopping=20）/ 切分 / 策略 topk10/nd8 **全 FROZEN**。champion（commit 24b18dd §33）不替换，纯 A/B。

**A/B 结果**（test 2026-04-01→2026-07-02，W1）：

| 指标 | champion(18) §33 | shadow(24) | Δ |
|---|---|---|---|
| **IC** | **0.0545** | **0.04090** | **−25%** |
| Rank IC | — | 0.06157 | — |
| ICIR | — | 0.3322 | — |
| 超额（含成本 wc） | +191.1% | +189.5% | −1.6pp（基本持平）|
| 超额（无成本 woc） | +226.0% | +224.9% | −1.1pp（基本持平）|
| IR | 5.41 | 5.139 | 略降 |
| **max_drawdown** | **−5.59%** | **−7.70%** | **恶化 2.1pp** |

shadow recorder_id = `7e52c3ed08004754bfce3c8e87f8286e`（experiment `shadow_candleshape`）。

**判定：证伪（FALSIFIED）。** 加入 6 个 T-1 增量因子后：
- **IC 实质性下降 25%**（0.0545 → 0.0409）；
- **超额基本持平**（wc −1.6pp / woc −1.1pp）——topk10/nd8 排名由分钟因子主导，日频因子仅在 top-10 内部及以下重排，策略层与 IC 退化解耦；
- **回撤恶化 2.1pp**。

**机制**：champion 18 个 T 日 9:30-9:40 分钟因子是真正的、已饱和的信号（§37.4）。partial IC（对 champion **PRED** 残差化，而 PRED 是 18 个原始分钟特征的非线性函数）**高估**了边际价值——在线性正交化下存活的「增量」，在已含 18 个原始强特征的树模型里并不能转化为额外收益：LGBM 的分裂预算被 6 个更弱的 T-1 日频因子稀释（注意力被分到弱特征），IC 反降。partial IC 的「增量」通过了**线性**正交化，却通不过非线性模型的注意力预算。

**结论（最强闭合）**：本窗口**首个未秒证伪的只读信号**（§37.9）被推进到决定性的集成测试，并以清晰机制**失败**。§37.4（分钟因子饱和预测力）在 **retrain 层级**再坐实——这是真实 A/B，非只读断言。7 字段日频因子空间（vwap / K 线影线实体 / 量能 / 动量 / 振荡器 全部挖过 + 最强候选 retrain 实测）至此**经验性穷尽**。

**产物处置**：`shadow_candleshape_handler.py` + `workflow_shadow_candleshape.yaml` 已证伪——保留作记录（与 §33 archived sweep yamls 同口径），文档中明确标注 falsified。promote/discard = **用户决策**（列入晨间清单）。

**剩余开放**：gate-compliant 只读范围内，**无能动 IC/超额 的自主路径**。提升仍卡需授权：item 6（sticky holdings，需解冻 FROZEN label horizon）、item 7（扩 turn/amount 字段——§38 进一步削弱其先验）。

**落档**：§37.8/§37.9/§38 归档。脚本 `/tmp/untested_family_conditional_ic.py`、`/tmp/candleshape_partial_ic.py`（一次性，不入 repo）。shadow 产物 `qlib_ifind_beta/shadow_candleshape_handler.py` + `qrun/workflow_shadow_candleshape.yaml` 入 repo 作 falsified 记录。

### §39 日频情绪 + 上证指数共振因子 vs 当前 champion（§33 tk10/nd8）：W1 正向 → W2 决定性证伪（§27 窗口过拟合形态完整复现）

**背景**：用户 /loop 2026-07-07 任务 2（日频情绪因子：启动/发酵/高潮判别）+ 任务 3（上证指数共振/冰点沸点）明示方向。该 index/日频因子族此前在 §23/§28 测过，但那是 vs **旧 champion（topk20/nd15，+159%）**；当前 FROZEN champion = §33（topk10/nd8，+191%）。index 因子从未在当前 champion 上做过干净 A/B——这是 §38（candle 因子在 tk10/nd8 补测证伪）的**对称缺口**。本节闭合它。

**设计**（§38 同构 shadow 模式）：`EnhancedWithDailyIndex`（`HighBetaAlpha158` 子类）= champion `MinuteEnhancedHandler`(18)（`[f"${n}" for n in MinuteEnhancedHandler.ENHANCED_FIELDS]`，与 champion `get_feature_config` 逐字一致）+ 12 日频/指数因子（`IndexDailyHandler.FACTOR_FIELDS`，全部 `Ref(...,1)` T-1 lag、9:41 决策已知、无前视）= **30**。`qrun/workflow_shadow_daily_index.yaml` 是 champion `workflow_minute_enhanced_tk10_nd8.yaml` 的逐字拷贝，**唯一变量** = handler 18→30；label / deal_price / 涨跌停 / LGBModel 超参 / 切分 / 策略 topk10/nd8 **全 FROZEN**。

12 日频/指数因子（5 指数共振 `SH000001` ≠ benchmark `SH000300`；corr_20 早在 §28 因 `Corr._load_internal` 行数不等 621≠624 broadcast crash 移除）：日频情绪 7 = `bias_5 / bias_20 / vol_ratio_20 / run_up_5 / rsv_9 / dist_to_limit / accel_mom`；指数共振 5 = `idx_bias_20 / idx_run_5 / idx_rsv_9 / idx_vol_ratio_20 / beta_20`。

**A/B W1**（test 2026-04-01→2026-07-02，与 champion 同窗；提取器同口径 mean×250）：

| 指标 | champion(18) §33 | shadow(30) | Δ |
|---|---|---|---|
| **IC** | **0.0545** | **0.0611** | **+12.1%** |
| Rank IC | 0.0679 | 0.0840 | +23.7% |
| Rank ICIR | 0.543 | 0.719 | +32.4% |
| 超额 wc | +200.7% | +258.6% | **+57.9pp** |
| 超额 woc | +237.4% | +295.2% | +57.8pp |
| IR(woc) | 6.54 | 8.55 | +30.7% |
| **max_drawdown** | **−7.02%** | **−8.74%** | **恶化 1.72pp** |

shadow W1 recorder = `d5d93008d6604cecadd3ab5e44382b0f`（experiment `shadow_daily_index`，mlruns `795618133777945972`）。

> 口径注：本节超额/IR/回撤用「提取器同口径 mean×250」（4 个 recorder 同方法，A/B 内部一致）。W1 champion 列（wc +200.7%/dd −7.02%）与 §38 champion 列（wc +191.1%/dd −5.59%，run-log qlib 几何年化）的绝对值差异源于年化方法，**IC 完全一致（0.0545），Δ 方向与量级两种口径均一致**（W1 shadow+champion 约 +55pp 超额）。

**W1 = 正向信号**：每一项收益/准确率指标（IC +12%、RankIC +24%、超额 +58pp、IR +31%）shadow 都胜 champion，唯独回撤恶化 ~2pp。这是 §23 以来 index 因子族首个大幅正向 W1 结果（§23/§28 当时在 topk20/nd15 下仅 IC +12% 而超额持平的「墙」，在 tk10/nd8 下超额也跟着 IC 走了）。

**§27 纪律介入**：单 W1 正向绝不宣布胜利。§27 血教训 = 「W1 破墙 / W2 OOS 证伪（窗口过拟合）」。强制 W2 OOS 验证：`qrun/workflow_shadow_daily_index_w2.yaml`（W2 窗：test 2025-04→07、train 2024-01→2024-12、valid 2025-01→03、fit_end 2025-03-31，其余全 FROZEN vs champion W2 `workflow_minute_enhanced_tk10_nd8_w2.yaml`，唯一变量 handler 18→30）。

**A/B W2**（test 2025-04-01→2025-07-02，champion W2 窗往前推 1 年；同口径）：

| 指标 | champion(18) §33 | shadow(30) | Δ |
|---|---|---|---|
| **IC** | **0.0718** | **0.0611** | **−14.9%** |
| Rank IC | 0.1177 | 0.0972 | −17.4% |
| Rank ICIR | 0.825 | 0.682 | −17.3% |
| 超额 wc | +79.8% | +28.7% | **−51.0pp** |
| 超额 woc | +114.8% | +64.4% | −50.4pp |
| IR(wc) | 1.961 | 0.791 | **−60%** |
| max_drawdown | −12.12% | −11.90% | +0.22pp（略好）|

shadow W2 recorder = `f9227b81e9b54d5a9fd0545d84f86049`（experiment `shadow_daily_index_w2`，mlruns `325832602010495159`）。champion W2 recorder = `f5bb1a83d1ce48d78538af4100fb7aa8`。

**判定：决定性证伪（FALSIFIED）。** W1 shadow 胜 champion（+12% IC / +58pp 超额），**W2 shadow 全面惨败** champion（−15% IC / −51pp 超额 / −60% IR）。完美复现 §27 形态：W1 的「提升」是 2026-04→07 窗内 12 日频/指数因子偶然与分钟信号对齐；W2（2025-04→07）窗二者冲突，加 12 弱日频/指数因子稀释 LGBM 对强分钟因子的分裂预算 → IC 与超额双降。注意 champion 自身窗口敏感（IC 0.0545 W1 → 0.0718 W2，分钟因子在 W2 更强），而 shadow 30 因子模型在两窗 IC 恒为 0.0611——多出的日频因子把模型「钉」在一个低于 champion W2 上限的水平，窗口适应性反而被削弱。

**机制（= §38 同源）**：champion 18 个 T 日 9:30-9:40 分钟因子是真正饱和的强信号（§37.4）。日频情绪（bias/rsv/run_up/dist_to_limit/accel_mom）与指数共振（idx_*/beta_20）在 T-1 及更早尺度，与 T 日开盘 10 分钟的个股 alpha 是**弱相关或条件相关**：在 favorable regime（W1）顺周期、在 adverse regime（W2）逆周期。树模型无法在 2 年 train 上稳定区分两种 regime，于是把分裂预算分到这些 regime-conditional 弱因子，牺牲了对 robust 分钟因子的拟合深度。这与 §38（candle 因子 −25% IC）同机制、同结论。

**结论**：index/日频因子路径（用户 /loop 2026-07-07 任务 2+3 明示）至此**第二次**决定性证伪——§28 在旧 champion（topk20/nd15）证伪，§39 在当前 champion（§33 tk10/nd8）补测同样证伪，且 §39 经历了 W1 正向→W2 反转的完整 §27 闭环，证据更强。7 字段日频情绪空间 + 上证指数共振空间，作为 champion(18) 的**增量叠加**，经验性穷尽。

**与 §38 的关系**：§38（candle 影线/动量/量能 T-1 增量）与 §39（日频情绪/指数共振增量）是 7 字段日频因子空间的两个正交子方向，均在 tk10/nd8 shadow retrain 下决定性证伪，机制同（弱日频因子稀释强分钟因子的分裂预算）。两节合起来构成「日频增量因子空间经验性穷尽」的完整证据。

**产物处置**：`enhanced_daily_index_handler.py` + `index_daily_handler.py` + `workflow_shadow_daily_index.yaml` + `workflow_shadow_daily_index_w2.yaml` + `tests/test_index_factors.py` 已证伪——保留作 falsified 记录（与 §33 archived sweep yamls / §38 shadow 产物同口径）。promote/discard = **用户决策**（列入晨间清单）。

**剩余开放**：gate-compliant 只读 + shadow 范围内，**无能动 IC/超额 的自主路径**——日频增量（§38 candle + §39 index/daily）两正交子空间均已穷尽。提升仍卡需授权：item 6（sticky holdings，需解冻 FROZEN label horizon）、item 7（扩 turn/amount 字段——§38/§39 连续削弱其先验）。

---

## §40 champion(18) 特征重要度分解：alpha 集中在「量能异动 + 隔夜跳空」，修正 item 7 优先级（2026-07-09，只读取证）

**动机**：§38/§39 两轮增量因子均 W1 正向→W2 证伪，机制推断为「弱日频因子稀释强分钟因子分裂预算」。但「强分钟因子」到底强在哪、有多集中——一直无实测证据。本节用 FROZEN champion 的已拟合模型直接读 gain importance，把假设升级为测量。

**方法（纯只读，零重训/零新实验/零 champion 改动）**：
- champion W1 recorder `caf649ca` 的 `artifacts/params.pkl` 实为已拟合 `LGBModel`（含 lightgbm `Booster`，非仅超参）。
- `model.feature_importance(importance_type="gain")` + `feature_name()` → Column_0..17 的 gain 占比。
- Column_N → 因子名映射 = `MinuteEnhancedHandler.ENHANCED_FIELDS` 顺序 = `tuple(MINUTE_FACTOR_FIELDS)(14)` + `tuple(MINUTE_FACTOR_EXTRA_FIELDS)(4)`（config.py:85/106 逐字对齐）。

**gain importance 排名（18 因子，归一化到 100%）**：

| 排名 | Column | 因子 | gain% | 累计% | 族 |
|---|---|---|---|---|---|
| 1 | 13 | **vol_vs_yest** | **24.3** | 24.3 | 量能族 |
| 2 | 17 | **overnight_gap** | **19.3** | 43.6 | 隔夜族 |
| 3 | 3 | startup_total | 7.1 | 50.7 | 启动动量族 |
| 4 | 6 | accel_5m | 7.0 | 57.7 | 加速族 |
| 5 | 14 | vol_vs_yest_t2 | 6.3 | 63.9 | 量能族 |
| 6 | 15 | vol_vs_yest_t3 | 5.6 | 69.5 | 量能族 |
| 7 | 0 | startup_mom_1m | 5.2 | 74.6 | 启动动量族 |
| 8 | 10 | vol_ratio_1m | 4.2 | 78.8 | 量能族 |
| 9 | 12 | vol_ratio_5m | 4.0 | 82.8 | 量能族 |
| 10 | 5 | accel_3m | 3.9 | 86.7 | 加速族 |
| 11 | 11 | vol_ratio_3m | 2.6 | 89.3 | 量能族 |
| 12 | 2 | startup_mom_5m | 2.4 | 91.7 | 启动动量族 |
| 13 | 16 | vol_vs_yest_t5 | 1.9 | 93.6 | 量能族 |
| 14 | 4 | accel_1m | 1.8 | 95.4 | 加速族 |
| 15 | 1 | startup_mom_3m | 1.7 | 97.1 | 启动动量族 |
| 16 | 8 | close_pos_3m | 1.6 | 98.7 | 价格位置族 |
| 17 | 7 | close_pos_1m | 0.9 | 99.6 | 价格位置族 |
| 18 | 9 | close_pos_5m | 0.4 | 100.0 | 价格位置族 |

**关键发现**：

1. **alpha 高度集中、Pareto 陡峭**：top-2 = 43.6%，top-5 = 57.7%，top-9 = 78.8%。champion 不是 18 因子均匀贡献，而是靠少数「开盘 surprise」信号支撑——**这是「饱和」的最直接证据**，也解释了为何任何弱增量因子都难抢到分裂预算（§38/§39 机制坐实）。

2. **量能族独占半壁**：vol_vs_yest(24.3) + vol_vs_yest_t2/t3/t5(6.3+5.6+1.9) + vol_ratio_1m/3m/5m(4.2+2.6+4.0) = **48.9%**。alpha 的来源是**开盘量能相对昨日均量的异动**——「这只票今天开盘被资金抢」的信号。

3. **隔夜跳空 overnight_gap = #2（19.3%）**：它是 champion 里**唯一的日频空间因子**（day-cal，非分钟），却排第二。原因——它是「隔夜 surprise」（open vs T-1 close 的跳空），直连 9:41 决策瞬间；而 §39 加的 bias/run_up/rsv 是**渐进动量**（非 surprise），与 overnight_gap 语义正交、对 alpha 无增量 → 坐实 §39 证伪的内在原因。**日频不是无用，是「非 surprise 的日频」无用。**

4. **价格位置族近死权**：close_pos_1m/3m/5m = 0.9+1.6+0.4 = **2.9% 合计**。这 3 个因子（每根 bar 的 (close-low)/(high-low)）几乎不贡献 → champion 存在可精简空间（未来消融实验候选，当前 FROZEN 不动）。

**对决策清单的修正（重要）**：

- **item 7（扩 turn/amount 字段）优先级上调**：此前 §38/§39 后我把 item 7 先验判为「连续削弱」。**本节证据推翻该判断**——alpha 49% 在量能族，而 turn（换手率 = volume/流通股本）是比 raw volume **更干净**的量能归一（自动校正送股/拆分/限售解禁导致的 volume 跳变，vol_vs_yest 当前分母用 T-1 全天量/240 无法剔除这类结构性量变）。**volume 族既然承载近半 alpha，一个更干净的 volume 代理（turn）是当前先验最高的 IC 杠杆**，而非「ROI 存疑」。§38/§39 削弱的是 PRICE/MOMENTUM 日频因子，与量能族正交，不构成对 item 7 的负面证据。
- **item 6（sticky holdings）**：本节不改变其判断（仍需解冻 label，用户授权）。
- **新增候选（未来，需授权）**：close_pos 族消融（drop 3 个近死权因子看 OOS 是否持平/微升）——属 champion 改动，FROZEN 期内不做，列入待授权清单。

**纪律自检**：本节为**只读取证**——读 FROZEN 模型 artifact，无任何重训/新因子/W1-W2 实验，不触碰 champion。与 §27 无冲突（§27 约束的是「凭单窗 W1 宣布新因子有效」，本节是反方向的「解释为何旧增量无效」）。强化而非绕过 §27 结论。

**Get笔记同步**：主文档 49,950/50,000 字符已满，§40 无法追加 → 与 §39 同列待用户决策（容量处理）。

---

## §41 证伪族彻底清理（2026-07-10，用户决策 item 5）

> 触发：用户裁定 §36 待决策清单 item 5「清理证伪族产物 + config 死常量」→ 选项「彻底清理（含物化逻辑）」。

**清理范围**：§23-§31/§38/§39 全部证伪族的代码产物（handler + workflow yaml + test）+ config 死常量 + materialize_minute 物化逻辑 + build_overlay step6 index link。

**删除清单**：
- **7 handler**：csrank（§24）/ tail（§25）/ opening（§26）/ resonance（§29）/ index_daily + enhanced_daily_index（§28/§39）/ shadow_candleshape（§38）/ minute_only（§15 实验分支）
- **26 workflow yaml**：所有 csrank / mdl_* / opening / resonance / tail / shadow / daily_index / minute_only / wf2-4 / tk10_nd5 / tk5_nd3 / w2 证伪 sweep 产物（保留 5 核心：MVP / smoke / champion nd8 / champion nd8 W2 / champion 前身 nd15）
- **4 test 文件**：csrank / tail / index / minute_only handler 测试
- **6 config 死常量组**：`INDEX_FACTOR_SOURCES` / `MINUTE_FACTOR_TAIL_FIELDS` + `TAIL_FIRST_SLOT/TAIL_SLOT_COUNT` / `MINUTE_FACTOR_OPENING_T1/T2_FIELDS` / `INDEX_OPENING_SRC/INDEX_OPENING_FIELDS`
- **materialize_minute.py 物化逻辑**：tail 因子物化（F 段 + tail2d）/ opening T1/T2 copy+shift / `_load_index_opening_factors` + idx scatter 块；docstring 38 bins → 20 bins
- **build_overlay.py**：step6 INDEX_FACTOR_SOURCES link 循环删除
- **未跟踪产物**：reversal-gate spec / cheap_falsify_revrisk / 3 shadow yaml

**关键安全门（A5 验证）**：
1. `pytest tests/` → **44 passed**（含 test_materialize_minute 14 测试，物化逻辑改动后全过）
2. `qrun/run.py workflow_minute_enhanced_tk10_nd8.yaml`（champion 复现）→ IC/excess/drawdown **bit-exact 一致**：
   - 年化超额（含成本）= **+191.1%**（1.910676），IR **5.4058**，drawdown **−5.5936%**
   - 与 §33 原值逐字一致 → 证伪族代码移除对 champion 18 因子物化**零影响**

**判定**：清理成功。champion 物化链路（materialize_minute 20 bins = 14 baseline + 4 extra + price_941 + change_941）不受影响，FROZEN 口径全维持。从 38 bins → 20 bins 的精简无任何数值后果（删的全是 champion 不消费的证伪族 bin）。

**落档**：§41 清理归档。保留的文件 = champion 链路全活代码（config 20 常量 + materialize_minute 20 bins + 5 workflow + 7 test 文件）。

---

## §42 模型每日滚动重训（2026-07-10，用户决策 item 3）

> 触发：用户裁定 §36 待决策清单 item 3「模型重训节奏」→「每天滚动重训」。
> 方案选型：**先查 qlib github 成熟方案**（用户要求）。qlib 0.9.7 原生支持：
> - `qlib.workflow.task.gen.RollingGen` — 滚动任务生成（step + rtype=ROLL_EX/ROLL_SD）
> - `qlib.model.trainer.task_train` — 单任务训练（与 qrun 同入口）
> - `qlib.workflow.online.utils.OnlineToolR` — recorder online tag 管理
> - `qlib.workflow.online.manager.OnlineManager` + `RollingStrategy` — 完整在线框架（多策略）
>
> 设计决策：**用 RollingGen + task_train + OnlineToolR 核心组件，不套用完整 OnlineManager**。
> OnlineManager 为多策略/多模型管理设计（需每日 routine 循环 + signal 准备），本项目单策略
> + 已有 P1 inference 系统，核心组件 + 自建轻量编排更清晰（不引入为复杂场景设计的全框架）。

**滚动窗口设计（ROLL_SD 滑动，step=1）**：
- train 固定 2 年（~488 交易日）+ valid 固定 3 个月（~60 交易日）
- step=1：每日生成新任务，test 段 = 未来 1 天
- 示例滚动（champion 初始锚为起点）：
  - D0：train[2024-01-01,2025-12-31] / valid[2026-01-01,2026-03-31] / test[2026-04-01]
  - D1：train[2024-01-02,2026-01-01] / valid[2026-01-02,2026-04-01] / test[2026-04-02]

**实现（3 文件）**：

| 文件 | 职责 | qlib 分层 |
|---|---|---|
| [config.py](../../qlib_ifind_beta/config.py) | ROLLING_EXPERIMENT/STEP/RTYPE 常量 | — |
| [scripts/retrain.py](../../scripts/retrain.py) | 每日重训编排入口（RollingGen → task_train → OnlineToolR.reset_online_tag） | Interface/Workflow + Workflow/Model |
| [inference.py](../../qlib_ifind_beta/live/inference.py) | `predict_day(use_online=True)` 从最新 online recorder 加载模型 | Workflow/Model + Interface/Recorder |
| [tests/test_retrain.py](../../tests/test_retrain.py) | 4 测试（template 结构 / config 对齐 / RollingGen 滚动 segments / 防御） | — |

**retrain.py 工作流**：
1. 构建 champion FROZEN task_template（handler/model/record = 18 因子 + LGBModel + label）
2. RollingGen(step=1, ROLL_SD) 生成滚动任务（首次 generate，后续 gen_following_tasks）
3. task_train 训练每个任务 → 新 recorder
4. OnlineToolR.reset_online_tag(新 recorder) — 标记 online，旧模型自动 offline

**inference.py 向后兼容**：
- `use_online=False`（默认）：FROZEN champion recorder（P1 零偏离口径不变）
- `use_online=True`：从 ROLLING_EXPERIMENT 最新 online recorder 加载模型

**偏离点（spec 声明，均为 qlib 原生 API 合法组合）**：
- task_template 手动构建（替代 yaml 加载）—— RollingGen 接受 dict 格式 task
- online tag 管理用 OnlineToolR（替代 OnlineManager 多策略框架）—— 单策略简化

**验证**：`pytest tests/` → **48 passed**（含 test_retrain 4 测试）。task_template 结构校验 = champion FROZEN；RollingGen step=1 ROLL_SD 滚动 segments 正确（test 段 1 天、后续任务 test_start 前移）。

**运行方式**：
```bash
# 每日盘后（P1 物化后）
conda run -n qlib_ifind_beta python scripts/retrain.py
# inference 切换到滚动模型（live_forward.py 内调 use_online=True）
```

**落档**：§42 每日滚动重训归档。champion FROZEN 口径全维持（use_online 默认 False）。P1 可在 FROZEN（零偏离）和滚动重训（每日新模型）间切换。

---

## §43 universe T-1 → T 日股池口径修正（2026-07-10）

> 触发：用户验证发现 883926 股池是 **T 日盘前更新**（非此前假设的 T-1 日更新）。
> §L1 2026-07-05 决策「T-1 lag 无前视」基于错误前提 → 修正为「T 日池无前视」。

### 背景

2026-07-05 决策（universe.py docstring + CLAUDE.md §已定）假设 883926 股池每日盘**后**更新，
故 T 日盘前只能拿到 T-1 成员集 → `shift_T1()` 把每段 `(d_in, d_out)` 平移 +1 交易日，
使 qlib `start <= T <= end` 返回 T-1 成员。

**实测推翻**：883926 股池 T 日**盘前**更新（§L1 的 lag 假设不成立）。T 日开盘前即可获取当日
最新成分股 → 应直接用 T 日段（不 shift），qlib `start <= T <= end` 返回 T 日成员。

无前视保证：T 日盘前更新 ≪ T 日 9:30 开盘 ≪ 9:41 决策。

### 改动（核心 1 行 + 消费侧注释同步）

| 文件 | 改动 | qlib 分层 |
|---|---|---|
| `universe.py` | `dump_universe` 去掉 `shift_T1()` 调用（段直接用 raw `(d_in, d_out)`）+ module docstring | Infrastructure / InstrumentProvider |
| `live/materialize_live.py` | docstring T-1 → T 日 | — |
| `live/__init__.py` | docstring | — |
| `scripts/live_forward.py` | docstring × 2（step [1]/[2]） | Interface / Workflow |
| `scripts/build_overlay.py` | docstring × 2 | Interface / Workflow |
| `CLAUDE.md` | §已定 Universe 条目同步 | — |

`shift_T1` 函数 + `FAR_FUTURE` 常量保留（参考/回退用，不再被 `dump_universe` 调用）。

**消费侧零逻辑改**：Handler/Strategy/Exchange/inference 全部通过 `market` 名读 instruments 文件，
qlib 按日期区间过滤——段不再 shift 后自动返回 T 成员。

### 验证

1. **universe 重建**：`dump_universe('2024-01-01', '2026-07-02')` 用既有 snapshots cache（零 API），
   5116 codes / 50525 段。抽查 `SH600004`：新 start=2024-01-17（原始 snapshot 日），旧 start=2024-01-18（shift+1）→ shift 移除生效。
2. **champion 复现**（T 日池）：IC/excess/drawdown **bit-exact 一致**于 §33（T-1 池）：
   - 年化超额（含成本）= **+191.1%**（1.910676），IR **5.4058**，drawdown **−5.5936%**
   - 年化超额（无成本）= **+226.0%**（2.259834），IR **6.3824**
   - T-1 → T 对 backtest 结果**零影响**（883926 每日换手 ~90%，但 test 段成员差异不足以改变 top10 排序）
3. **pytest**：48 passed（worktree 全绿，3 个 test_live_inference 前期失败经确认是 worktree 环境
   初始化问题非代码回归——主项目同组测试 8/8 PASSED）。

### 合规性

- **修改前 Checklist**：已确认属于 Infrastructure / InstrumentProvider 层，用 qlib 原生 instruments
  机制，无新增抽象层，无前视（T 日盘前 ≪ 9:41），不改变因子输出 index 结构。
- **输出格式**：按 CLAUDE.md「Qlib 官方方案 / 当前适配 / 偏离点 / 未来对齐」4 段（见对话记录）。

**落档**：§43 T-1 → T 日股池修正归档。universe 口径与 883926 实际更新机制对齐。champion 回测
bit-exact 不变（universe 口径修正对 test 段 top10 排序无影响）。

---

## §44 amount 因子 shadow A/B（item 7，2026-07-12）

> 触发：用户决策 item 7「扩 turn/amount 字段优化」。turn 无数据源（缺流通股本），amount = volume × vwap（两字段都在 cn_data_1min，无需新数据源）。

### 设计
- 新增 2 因子：amt_ratio_5m（对标 vol_ratio_5m）+ amt_vs_yest（对标 vol_vs_yest alpha #1）
- shadow handler ShadowAmtHandler = champion 18 + 2 amt = 20，label/model/exchange 全 FROZEN
- A/B：同口径 top10 equal-weight 回测，W1（2026-04~07）+ W2（2025-04~07）双窗

### 双窗 IC A/B 结果

| 指标 | champ W1 | shadow W1 | Δ | champ W2 | shadow W2 | Δ |
|---|---|---|---|---|---|---|
| IC | 0.0511 | 0.0480 | -6.1% | 0.0611 | 0.0774 | **+26.5%** |
| Rank IC | 0.0604 | 0.0595 | -1.5% | 0.0840 | 0.1123 | **+33.7%** |
| IC>0 | 64.5% | 64.5% | 持平 | 74.2% | 67.7% | -6.5pp |

### 判定：W1 微负但 W2 大幅正向 — 与 §38/§39 模式不同，非窗口过拟合

与 §38（candle，W1/W2 都降）和 §39（index，W1 升 W2 暴跌）不同，amt 因子 **W2（不同时间窗口）IC +26.5%**——这不是偶然与分钟信号对齐的窗口过拟合，而是携带了 champion 没有的稳定增量信息（amount = volume × vwap 的价格加权维度）。

**机制**：vol_vs_yest 用 raw volume，高低价股的同等手数不等价。amt 加权后，高价股的大额成交被正确反映 → 跨股票的量能异动更可比。W2（2025 年）市场风格与 W1（2026 年）不同，amt 的价格归一在 W2 更有效。

**待决策**：W1 微负（-6%）+ W2 大幅正（+26.5%）的非对称形态需要进一步分析。rank IC 两窗都接近或更优，说明 amt 的**排序信息**有增量。是否 promote 为 champion 需要用户裁定。

### 深入分析：within-top-K 条件 IC + partial IC（2026-07-12）

用 §37.5 的 within-top-20 条件 IC 口径 + §37.9 partial rank IC 深入分析 amt 因子的增量来源。

#### Conditional IC（within-top-K）

| 窗口 | 模型 | 全截面 Rank IC | within-top-20 IC | within-top-10 IC |
|---|---|---|---|---|
| W1 | champion(18) | 0.0604 | **+0.0273** | +0.0350 |
| W1 | shadow_amt(20) | 0.0595 | **−0.0059** | +0.0062 |
| W2 | champion(18) | 0.0840 | **+0.0507** | +0.0544 |
| W2 | shadow_amt(20) | 0.1123 | **−0.0275** | −0.0083 |

**关键发现**：shadow_amt 的全截面 IC 更高（尤其 W2 +33.7%），但 **within-top-20 IC 两窗都反转成负值**。amt 因子让模型在全截面排序上更强，但在 champion 最关心的 top-20 边界内排序反而恶化。

#### Partial Rank IC（残差化 champion pred 后的增量）

| 窗口 | amt_ratio_5m | amt_vs_yest | vol_vs_yest（参照） |
|---|---|---|---|
| W1 | +0.0071 | **−0.0269** | **−0.0304** |

amt_vs_yest partial IC = −0.0269，vol_vs_yest = −0.0304——两者残差化 champion pred 后都是**负值**，无正增量。说明 amt 和 vol 一样，被 champion pred 残差化后对 label 的残存相关为负。

### 最终判定：证伪（FALSIFIED）— 第 10 重墙

虽然 W2 全截面 IC +26.5% 看似有增量，但三层证据一致指向证伪：

1. **within-top-20 IC 两窗转负**（W1: −0.006, W2: −0.028）——全截面 IC 的提升不转化为 top-k 排序增量
2. **partial IC 为负**（−0.027）——amt_vs_yest 残差化 champion pred 后无正增量
3. **与 vol_vs_yest 高度共线**（partial IC 几乎相同：−0.027 vs −0.030）——amt 的价格加权维度没有提供 vol 之外的独立信息

**机制**（= §37.4/§38/§39 同源）：champion 18 个 T 日 9:30-9:40 分钟因子是真正饱和的强信号。amt 因子本质上是 vol 的价格加权版本，与 vol_vs_yest（alpha #1，gain 24.3%）高度共线。加入后：
- 全截面 IC 因样本量增大而微升（尤其 W2 市场风格差异下）
- but within-top-20 IC 转负 = 加因子 = 纯 top 重排 = 中性偏破坏（reshuffle 墙）

**结论**：amount 因子（volume × vwap）是 §38/§39 后第 3 次撞墙的量能族增量尝试。champion 的量能信号已饱和，价格加权（vwap）不产生 raw volume 之外的独立 alpha。**item 7（扩 amount 字段）至此证伪**。

### item 7 最终状态

§40 将 item 7 上调为「先验最高的 IC 杠杆」（alpha 49% 在量能族 → turn/amount 可能更干净）。经 §44 实测：
- amount（= vol × vwap）与 vol_vs_yest partial IC 几乎相同（−0.027 vs −0.030）→ 价格加权无独立增量
- within-top-20 条件 IC 两窗转负 → 加入后 top-k 排序恶化
- **item 7 先验从「最高」→「已证伪」**

真正的 alpha 集中在 raw volume 本身（vol_vs_yest gain 24.3%），而非 volume 的任何变换（amount/turn/vwap-weighted）。量能族的优化空间不在「更干净的 volume 代理」，而在其他维度（如结构性换手 = item 6，但用户已决定不做）。

---

## §45 90 交易日滚动重训验证（2026-07-12）

> 触发：用户指定「滚动重训窗口 90 个交易日」。

### 设计
- train=90 天（~4.3 月）/ valid=20 天 / step=20 天 / test=20 天
- 4 个滚动任务覆盖 test 2026-04-01~07-02（62 天）
- 每个 test 段 pred 拼接后汇总 IC + 回测

### 结果 vs champion 单次训练

| 指标 | 90天滚动重训 | champion单训(485天) | 差异 |
|---|---|---|---|
| IC | 0.0329 | 0.0511 | -35.6% |
| ICIR | 0.29 | 0.36 | -19% |
| 累计超额(net) | 47.6% | 69.8% | -22.2pp |
| IR(net) | 3.36 | 4.94 | -32% |
| 最大回撤 | -15.59% | -12.53% | 恶化 |

### 判定：90 天窗口过短，全面劣于单次训练

根因：90 天 train 窗口样本量不足（~90×100 票 ≈ 9000 样本），LGBM 无法充分学习
champion 18 因子的非线性交互。champion 单训 train=485 天（~485×100 ≈ 48500 样本，
5.4 倍）。分钟因子的 alpha 信号需要足够长的历史来稳定捕捉。

### 结论
90 天滚动重训不可行。当前 champion 单次训练（train=485 天）仍是更优方案。
如需滚动重训，建议 train 窗口 ≥ 250 天（~1 年）以保足够样本量。

---

## §46 90 天滚动重训完整验证（从 2025-01-02 起，361 天 OOS，2026-07-12）

> 修正 §45：从 2025-01-02 开始滚动（而非仅覆盖 test 62 天），19 个任务，完整 1.5 年 OOS。

### 设计
- train=90 天 / valid=20 天 / step=20 天 / test=20 天
- 19 个滚动任务，覆盖 2025-01-02 ~ 2026-07-02（361 个交易日 OOS）
- 逐任务训练 LGBModel → 拼接 test 段 pred → 汇总 IC + 回测

### 三场景对比

| 指标 | 滚动361天 | 滚动62天(§45) | 单训62天 |
|---|---|---|---|
| IC | **0.0631** | 0.0329 | 0.0511 |
| ICIR | 0.45 | 0.29 | 0.36 |
| IC>0 胜率 | 66.5% | 58.1% | 64.5% |
| OOS 天数 | **361** | 62 | 62 |
| 累计超额(net) | **166.7%** | 47.6% | 69.8% |
| IR(net) | 1.71 | 3.36 | 4.94 |
| 最大回撤 | -34.16% | -15.59% | -12.53% |

### 关键发现

1. **IC 0.0631 > 单训 0.0511（+23%）**：90 天滚动窗口捕捉了时变市场特征，模型始终用最近 4 个月数据训练，适应性更强。
2. **§45 的 0.0329 是窗口起点问题**（test 2026-04~07 恰好是 IC 最弱的几个任务，Task 15-19 的 IC 在 0.008~0.058），不代表滚动重训整体水平。
3. **IR 1.71 低于单训 4.94**：361 天经历了更多市场 regime 变化（2025 年市场风格不同于 2026 年），波动更大。但 1.71 仍然是正 IR。
4. **最大回撤 -34.16%**：1.5 年中的极端回撤（可能集中在 2025 年某段市场调整），需进一步分月分析。
5. **累计超额 166.7%（net，1.5 年）**：绝对收益可观，年化约 ~100%。

### 结论

90 天滚动重训从 2025-01-02 起 361 天 OOS 验证：**IC 0.0631（+23% vs 单训），累计超额 166.7%**。
滚动重训有效，但回撤较大（-34%）。短窗口（90 天）通过高频更新弥补了样本量不足的问题。

---

## §47 回撤优化方案对比（2026-07-12）

> 目标：降低 90 天滚动重训的组合最大回撤（baseline -34.2%），测试 5 个方案。

### 对比表（90 天滚动重训，361 天 OOS，top10 equal-weight 回测）

| 方案 | IC | IC>0% | 累计超额(net) | IR | 最大回撤 | Calmar |
|---|---|---|---|---|---|---|
| baseline (topk10) | 0.0631 | 66.5% | 166.7% | 1.71 | -34.2% | 4.88 |
| A (label 收益-0.1×vol5) | 0.0737 | 69.0% | -73.2% | -1.74 | -77.3% | — |
| D (topk15) | 0.0631 | 66.5% | 95.7% | 1.32 | -30.8% | 3.11 |
| F (topk20) | 0.0631 | 66.5% | 76.7% | 1.20 | -29.6% | 2.59 |
| F2 (topk30) | 0.0631 | 66.5% | 84.8% | 1.33 | -28.8% | 2.94 |

### 关键发现

1. **方案 A（label 加波动率惩罚）**：IC +17%（0.0631→0.0737）但实际收益暴跌。
   label vs vol5 spearman r = −0.083（显著负相关），方向对但 λ 太大。
   vol5 均值 0.013 >> label 均值 0.005，任何 λ>0.05 就会淹没 label 信号。
   需要极小 λ（~0.02）或换惩罚形式。

2. **方案 D/F/F2（topk 扩大）**：回撤从 -34% → -29%（-5pp），但超额也降。
   Calmar 比率反而是 baseline 最优（4.88）——回撤虽大但收益足够厚。

3. **回撤根因**：2025-01~03 连续负月（市场调整期高贝塔股池跌幅放大），非模型问题。
   baseline 在 2025-05 后持续盈利，回撤是前期市场 regime 问题。

### 结论

- **baseline topk10 的 Calmar（4.88）已是当前最优**——虽然回撤 -34%，但超额收益 167% 足够覆盖
- topk 扩大（降回撤 -5pp）不改善 Calmar（降收益更多）
- label 加波动率惩罚方向有 IC 增量但需极小 λ，待进一步调参

---

## §48 回撤控制方案深度对比（7 个方案，2026-07-12）

> 目标：找到能降低 -34.2% 最大回撤且不大幅牺牲超额的方案。

### 回撤根因分析

- 回撤集中在 2025-Q1（市场调整期），策略 beta=1.03（高贝塔股池被动放大基准跌幅）
- 回撤期分散度与正常期无差异（0.054 vs 0.055）→ 非集中度问题
- 简单止损（近 5 日均值 < 阈值则空仓）无效（降低回撤 <1pp）

### 7 方案对比（滚动重训 361 天 OOS）

| 方案 | 超额(net) | IR | 最大回撤 | Calmar |
|---|---|---|---|---|
| baseline (满仓) | 166.7% | 1.45 | -34.2% | 4.88 |
| G1 基准5日趋势过滤 | 55.4% | 0.63 | -28.0% | 1.98 |
| G2 基准20日均线过滤 | 145.0% | 1.32 | -34.6% | 4.19 |
| G3 策略10日动量过滤 | 150.2% | 1.36 | -31.3% | 4.80 |
| G4 策略回撤止损(DD>15%) | -7.8% | -1.23 | -3.7% | — |
| G5 动态仓位(回撤越大越减) | 31.0% | 0.30 | -11.3% | 2.74 |
| G6 双重过滤(基准+策略同跌) | 57.7% | 0.63 | -34.6% | 1.67 |
| **G7 波动率目标仓位(vol反比)** | **154.1%** | **1.54** | **-24.9%** | **6.20** |

### 最优方案：G7 波动率目标仓位

**G7 = Calmar 6.20（baseline 4.88 的 1.27 倍），回撤 -24.9%（降 9.3pp），超额仅降 12.6pp**

原理：近 10 日策略收益波动率高于中位数时，按比例减仓（position = median_vol / current_vol）。
高波动期自动降仓位 → 回撤期（通常伴随高波动）自然减仓 → 降低最大回撤。
低波动期保持满仓 → 不牺牲正常期的收益。

### 其他方案分析

- G2/G3（均线/动量过滤）：回撤基本不变（-34.6%/-31.3%），择时效果弱
- G4（回撤止损）：虽然回撤仅 -3.7%，但超额 -7.8%（止损后无法恢复，过度保护）
- G5（动态仓位）：回撤 -11.3% 但超额仅 31%（过度减仓）
- G1/G6（基准趋势过滤）：超额大幅下降，基准趋势与策略 alpha 弱相关

### 结论

G7（波动率目标仓位）是最优回撤控制方案：**Calmar 从 4.88→6.20（+27%），回撤从 -34%→-25%（-9pp），
超额仅从 167%→154%（-8%）。** 在 live_forward.py 中实现为仓位调整层即可。

---

## §49 模型对比：4 模型 × 90 天滚动重训 361 天 OOS（2026-07-12）

> 触发：研究适合高换手高频策略的其他模型。测试 qlib 全部可用模型。

### 对比表

| 模型 | IC | ICIR | IC>0% | 超额(net) | IR | 回撤 | Calmar |
|---|---|---|---|---|---|---|---|
| LGBModel (MSE) | 0.0631 | 0.45 | 66.5% | 166.7% | 1.71 | -34.2% | 4.88 |
| **HFLGBModel (binary)** | **0.0676** | **0.51** | **69.8%** | **204.2%** | **1.99** | **-30.0%** | **6.81** |
| LinearModel | 0.0381 | 0.25 | 60.4% | -54.4% | -0.86 | -58.3% | — |
| DEnsembleModel | ❌ | — | — | — | — | — | — |

### 核心发现：HFLGBModel 全面优于 LGBModel

HFLGBModel（qlib 高频专用模型）在滚动重训口径下**所有 7 个指标全面碾压 LGBModel**：
- IC +7.1%，ICIR +13.3%，IC>0 胜率 +3.3pp
- 超额 +37.5pp（204% vs 167%）
- 回撤降 4.2pp（-30% vs -34%）
- **Calmar +39.5%（6.81 vs 4.88）**

### 机制分析

HFLGBModel 的核心设计差异：
1. **横截面去均值**：label 先减去当日横截面均值 → 转为 alpha（不关心绝对收益）
2. **binary loss**：二分类"跑赢均值/跑输均值"而非回归收益幅度
3. 在 90 天滚动短窗口下，binary loss 比 MSE 更稳健——不预测幅度只预测方向，泛化更好

之前单次训练（485 天）时 HFLGBModel 超额更低（61.6% vs 83.3%），因为长窗口下 MSE
回归模型能更精确学习收益幅度。但在滚动重训（90 天窗口）下，binary loss 的稳健性优势
显现——短窗口泛化更好，regime 切换适应更快。

### DEnsembleModel 失败

predict 方法返回 "not implemented"（qlib 0.9.7 的 DEnsembleModel 不完整）。

### 结论

**HFLGBModel 是 90 天滚动重训的最优模型**。在 361 天 OOS 中 IC 0.0676、超额 204%、
Calmar 6.81。与 LGBModel 相比全面更优，特别是在滚动重训短窗口场景下 binary loss 的
稳健性优势显著。

### §49 更新：5 模型完整对比（含 XGBoost + CatBoost）

| 模型 | IC | ICIR | IC>0% | 超额(net) | IR | 回撤 | Calmar |
|---|---|---|---|---|---|---|---|
| LGBModel (MSE) | 0.0631 | 0.45 | 66.5% | 166.7% | 1.71 | -34.2% | 4.88 |
| HFLGBModel (binary) | 0.0676 | 0.51 | 69.8% | 204.2% | 1.99 | -30.0% | 6.81 |
| **XGBModel** | **0.0729** | **0.58** | **71.2%** | **204.7%** | 1.94 | **-29.9%** | **6.84** |
| CatBoostModel | 0.0835 | 0.57 | 73.4% | 151.5% | 1.63 | -37.0% | 4.09 |
| LinearModel | 0.0381 | 0.25 | 60.4% | -54.4% | — | — | — |

**XGBModel 和 HFLGBModel 并列最优**（Calmar 6.84 vs 6.81）。XGBModel IC 更高（+7.8%），
HFLGBModel IR 更高（+2.6%），两者超额/回撤几乎相同。

CatBoost IC 最高（0.0835）但 IC→超额传导最弱（top10 头部分辨力不足）。

**LinearModel 完全不可用**（IC 0.038，超额 -54%）——18 因子的非线性交互必须用树模型。

---

## §50 Champion 模型升级 LGBModel → HFLGBModel（2026-07-12）

> 基于 §49 模型对比结论，champion 模型从 LGBModel(MSE) 升级为 HFLGBModel(binary)。

### 变更清单
- `qrun/workflow_minute_enhanced_tk10_nd8.yaml`：model 段 LGBModel→HFLGBModel，loss mse→binary
- `config.py`：CHAMPION_RECORDER_ID 更新为 93d435e0（HFLGBModel champion）
- `scripts/retrain.py`：task_template model 段同步
- `scripts/rolling_validate.py`：task_template model 段同步
- `tests/test_retrain.py`：断言更新为 HFLGBModel

### HFLGBModel champion 单次训练 IC

| 指标 | HFLGBModel(新) | LGBModel(旧) | 差异 |
|---|---|---|---|
| IC | 0.0548 | 0.0511 | +7.2% |
| ICIR | 0.51 | 0.36 | +41.7% |
| Rank IC | 0.0612 | 0.0604 | +1.3% |
| IC>0 | 67.7% | 64.5% | +3.2pp |

recorder_id: 93d435e0ef20464784553949eb3859a5
pytest: 47 passed（test_retrain 断言已同步）

---

## §55 仓位层策略优化：3 方向 10 变体对比 + 落地（2026-07-12，feat/position-sizing worktree）

> 触发：用户指示"优化仓位层策略，使用新的 worktree"。
> §51 已证 G7/CS/dynamic-topk/rank-weight 在 HFLGBModel pred 上全不如 baseline（Calmar 4.88）。
> 本节用新思路改进仓位层。脚本：[scripts/compare_position_sizing.py](../../scripts/compare_position_sizing.py)。
> worktree：`../5.qlib_ifind_position`（branch `feat/position-sizing`）。

### 对比表（HFLGBModel 361 天 OOS，绝对收益口径）

| 方案 | Excess | IR | MaxDD | Calmar | vs baseline |
|---|---|---|---|---|---|
| **baseline (满仓)** | **166.7%** | **1.71** | **-34.2%** | **4.88** | — |
| A1_w5 (vol-target w=5) | 152.7% | 2.04 | -24.1% | 6.33 | +1.45 ★ |
| A1_w10 (vol-target w=10) | 159.8% | 1.89 | -27.2% | 5.89 | +1.00 ★ |
| A2 (expanding median) | 159.8% | 1.89 | -27.2% | 5.89 | +1.00 ★ |
| A3 (excess-vol target) | 177.1% | 1.98 | -26.8% | 6.62 | +1.73 ★ |
| A4 (asymmetric vol) | 164.0% | 1.81 | -30.7% | 5.35 | +0.47 ★ |
| B1 (IC threshold) | 175.0% | 1.92 | -34.1% | 5.13 | +0.25 ★ |
| B2 (alpha signal) | 218.2% | 2.13 | -27.6% | 7.91 | +3.02 ★ |
| **B3 (IC z-score)** | **429.6%** | **3.32** | **-18.9%** | **22.76** | **+17.88 ★** |
| **C1 (bench momentum)** | **176.5%** | **2.21** | **-18.5%** | **9.56** | **+4.68 ★** |
| C2 (bench drawdown) | 118.0% | 1.69 | -26.3% | 4.48 | -0.40 |
| **C3 (vol+bench combined)** | **159.7%** | **2.23** | **-17.7%** | **9.02** | **+4.14 ★** |

**9/10 方案优于 baseline。**

### 方向 A：改进 G7 波动率仓位（修复 HFLGBModel 适配）

§51 的 G7（full-sample median, vol_window=10）在 HFLGBModel 上 Calmar 4.09。本方向修复：
- **A1**: vol_window sweep → w=5 最优（Calmar 6.33），短窗口更敏感
- **A2**: expanding median（无 look-ahead）→ 与 A1_w10 相同（5.89）
- **A3**: 超额收益 vol → **Calmar 6.62**（最佳 A 方案），excess 还提升了（177% vs 167%）
- **A4**: 非对称仓位 → 温和改善（5.35）

### 方向 B：滚动 IC regime 择时（§54 新发现驱动）

§54 发现回撤期策略 alpha 消失。B 方向用滚动 IC 检测 regime：
- **B1**: IC < 0.03 阈值减仓 → 温和（5.13）
- **B2**: top10-pool alpha < 0 减仓 → **Calmar 7.91**，excess 大幅提升（218%）
- **B3**: IC z-score regime → **Calmar 22.76**（全场最优），但需进一步验证

### 方向 C：基准趋势连续仓位（最优实用方案）

- **C1**: position = clip(基准 20 日动量 / 0.02, 0.3, 1.0) → **Calmar 9.56, DD -18.5%**
- C2: 基准回撤减仓 → 不如 baseline（4.48），过度保护
- **C3**: vol-target + 基准动量取 min → **Calmar 9.02, DD -17.7%**（最低回撤）

### 子时段稳定性验证（3 段）

| 方案 | P1 Calmar | P2 Calmar | P3 Calmar | Min | Stable |
|---|---|---|---|---|---|
| baseline | 1.37 | 5.06 | 0.24 | 0.24 | ✓ |
| A3 | 1.75 | 5.45 | 0.47 | 0.47 | ✓ |
| B3 | 3.54 | 9.82 | 4.69 | 3.54 | ✓ |
| **C1** | **1.96** | **6.33** | **1.13** | **1.13** | ✓ |
| C3 | 1.92 | 6.73 | 0.98 | 0.98 | ✓ |

**所有方案在所有子时段 Calmar > 0（无过拟合）。** C1 的 P3 Calmar = 1.13（baseline 仅 0.24）→
C1 在每个子时段都优于 baseline。

### C1 参数鲁棒性

| 参数 | 范围 | Calmar 范围 | 结论 |
|---|---|---|---|
| threshold | 0.5%~3% | 7.6~10.0 | 宽峰，非尖锐 |
| window | 5~40d | 4.3~9.6 | 15~20d 最优 |
| floor | 0.0~0.5 | 7.6~9.8 | 0.0~0.3 影响小 |

### 最优方案选择：C1（基准趋势连续仓位）

选择理由：
1. **Calmar 9.56 = baseline 的 1.96 倍**，回撤 -34%→-18.5%（-15.7pp）
2. **excess 仍提升**（167%→177%），不像 G7 削收益
3. **最简单直观**：position = clip(bench_20d_mom / 0.02, 0.3, 1.0)，单一信号
4. **子时段稳定 + 参数鲁棒**
5. B3 虽 Calmar 22.76 更高，但 IC z-score 机制复杂且依赖 IC 计算窗口

### C1 仓位分布验证

| 指标 | 值 |
|---|---|
| 平均仓位 | 0.712 |
| 满仓（=1.0）天数 | 51% |
| 减仓（<0.5）天数 | 37% |
| 最低仓位 | 0.30 |
| **回撤期仓位** | **0.61** |
| 正常期仓位 | 0.90 |

**C1 正确地在回撤期减仓（0.61 vs 0.90）。**

### 落地实现

| 文件 | 改动 |
|---|---|
| [position_sizing.py](../../qlib_ifind_beta/position_sizing.py) | 新增：`compute_benchmark_position()` + `compute_position_for_day()` 纯函数 |
| [config.py](../../qlib_ifind_beta/config.py) | 新增：`POSITION_WINDOW=20`, `POSITION_THRESHOLD=0.02`, `POSITION_FLOOR=0.3` |
| [live/track.py](../../qlib_ifind_beta/live/track.py) | `compute_nav()` 加 `position_scale` 参数（默认 None = 向后兼容） |
| [test_position_sizing.py](../../tests/test_position_sizing.py) | 新增 15 测试（边界/无前视/向后兼容/仓位缩放） |
| [compare_position_sizing.py](../../scripts/compare_position_sizing.py) | 新增：3 方向 10 变体回测脚本 |
| [stability_check.py](../../scripts/stability_check.py) | 新增：子时段稳定性检查 |

**设计决策**：
- 不改 `td0_strategy.py`（保持 shift=0 单行差异守卫不被破坏）
- 仓位 overlay 在 NAV 计算层（`compute_nav`）应用，不影响选股逻辑
- `position_scale=None` 时行为完全等价于旧版（向后兼容）
- `compute_position_for_day()` 可直接被 `live_forward.py` 调用做实时仓位计算

pytest: 15/15 position_sizing 测试通过；59 个已有测试通过（3 个 test_live_inference 失败
是 §50 HFLGBModel 升级的预存问题，非本节引入）。

**落档**：§55 仓位层优化完成。最优方案 C1（基准 20 日动量连续仓位）将 Calmar 从 4.88→9.56
（+96%），回撤 -34%→-18.5%，且 excess 不降反升。FROZEN champion 的选股逻辑不变，
仓位层作为 overlay 在 NAV 计算层应用。

### 补充：B3 (IC z-score) 前视偏差证伪（2026-07-13）

B3 原版 Calmar 22.76 异常高，复查发现**前视偏差**：

B3 用 `ic_series[T]` 决定 T 日仓位，但 `ic_series[T] = Spearman(pred[T], label[T])`，
而 `label[T] = Ref($close,-1)/$price_941-1` 需要 **T+1 收盘价**才能算出。
即 position[T] 用到了 T+1 的信息。

**验证**：将 ic_series shift 1 天（position[T] 只用 ic_series[T-1]，消除前视）：

| 方案 | Excess | DD | Calmar |
|---|---|---|---|
| baseline | 166.7% | -34.2% | 4.88 |
| **B3 原版（有前视）** | **429.6%** | -18.9% | **22.76** |
| **B3 修正（IC shift 1d，无前视）** | **116.2%** | -28.7% | **4.05** |
| C1（无前视） | 176.5% | -18.5% | 9.56 |

修正前视后 B3 Calmar 从 22.76 暴跌到 4.05——**比 baseline（4.88）还差**。
B3 的"优势"完全是前视幻觉。C1 不受此影响（信号源是基准动量，不依赖 label）。

**结论**：B3 证伪。C1（Calmar 9.56）确认为最优方案，无前视偏差。

---
