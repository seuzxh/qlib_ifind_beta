---
layout: default
title: qlib-factor-zoo 六库广谱筛选与分钟域移植（Phase 0–2 执行记录）
parent: "验证与研究"
nav_order: 9
---

# qlib-factor-zoo 六库广谱筛选与分钟域移植（Phase 0–2 执行记录）

> 日期：2026-09-24（分支 `research/factor-zoo-20260924`，方案 v1.1）
> 状态：**执行记录（筛选层 + 冻结协议 timebox）**；Champion 18 因子不变，
> 无任何因子晋升——按方案 Phase 3 规则，晋升须 purged 19 段终审 + 人工审批
> 产物：`reports/factor_zoo/`（本地产物，不入 Git）：`factor_zoo_smoke.json`、
> `factor_zoo_parse.json`、`factor_zoo_data_audit.json`、`day_screen.csv`、
> `minute_b1_metrics.csv`、`minute_b2_t1_metrics.csv`、`accel_state.pkl`；
> 实验 mlrun `factor_zoo_augment`
> 上游：<https://github.com/JustinF8/qlib-factor-zoo> @ `ea21f315`（MIT，vendor
> 于 `qlib_ifind_beta/factor_zoo/`，出处见其 SOURCE.md）

## 0. Phase 0 基建与审计

### 0.1 vendor 与算子兼容（0.2 冒烟）

- 六库共 **1007** 个表达式（Alpha360×360 / Alpha158×158 / Alpha101×101 /
  GTJA191×191 / TDXGS×88 / JQ110×109），表达式与 50+ 自定义算子逐字节 vendor。
- fork 算子类覆写 `__init__` 且不设 `self.feature`，与 pyqlib 0.9.7 的窗口
  递归协议不兼容（`AttributeError`）→ `ops_compat.py` 为 61 个算子类生成
  同名子类，补 `get_extended_window_size`（子表达式窗口并集 + 整数参数）
  与 `get_longest_back_rolling`；模块级注入保证 joblib pickle 可用。
- **冒烟结果：999/1007（99.2%）可求值**。数值对照：`SMA` 与手工 numpy 逐位
  一致（max|diff| 7.6e-6）、`TsArgmax` 完全一致（0.0）、`RSI` Spearman
  0.998（残差=ewm 预热深度差异，见口径注意）。8 个不可用因子均为上游缺陷：
  ALPHA002（单参 Rank）、GTJA043/059/069/084/157/166（表达式书写错误）、
  TDXGS_EMA20_CSRANK（fork 专属 CsRank 算子未 vendor）。
- 口径注意：①shim 的窗口扩展是**有限窗口近似**——上游对 ewm/SMA 类声明
  无限记忆，本实现按窗口参数扩展预热，对 RSI 等无界指标有 O(1%) 量级影响
  （筛选口径足够，若进生产需精确复刻）；②Alpha101 全库用 `Rank(X,1)` 近似
  WorldQuant 截面 rank（窗口 1 时序 rank，退化为常数 1），系上游自身口径，
  筛选结果如实继承。

### 0.3 graveyard 排雷

`excluded.json` 结构化 10 个已证伪族 + 4 条方法论规则（同批基线重跑、手动
口径只排序、延迟执行对照、purged 19 段终审）。路线 A 筛选的 grave 匹配按
表达式裸名匹配：Alpha158/Alpha360 的 MIN/MAX/LOW/CLOSE/ROC/STD/RANK 等
列名族命中 `alpha158_y`（71 个 Alpha158@T-1 曾在四层漏斗全灭）。

### 0.4 数据审计

- T-k 完整 240bar 覆盖率（T 日成分在 T-k 有完整分钟数据）：T-1 均值
  **98.35%**（P5 95.0%）、T-2 97.89%、T-3 97.54%、T-5 96.98% → 多日族可行，
  缺失交 DropnaProcessor。
- 截面时点：universe 由 08:30 盘前 cron 拉取，T 日成分 09:41 前可得 ✓。
- 复权口径：分钟域一律 1min 原值（名义口径），与 label v2 / overnight_gap
  决策一致（2026-09-11 gap A/B 结论）。
- 1min 字段：7 字段齐备（抽样 50 股 1 只全缺，与 89/5146 无 volume bin 一致）。

### 0.5 静态解析（B1/B2 可实例化）

窗口参数扫描：**B1（当日盘初，窗口≤8）202 个；B2（多日全天，窗口≤120）
982 个**（B1⊆B2）；too_long 17、unusable 8。

## 1. 路线 A：六库日频 T-1 lag 广谱筛选

999 个 `Ref(expr,1)` 在 universe 内对 label v2 的双轨指标（2024-01-02→
2026-09-11，n_days≈653；2024 年初长窗因子有 NaN 预热，覆盖率如实列示）。
**screened 总数 = 999**（多重检验视角：653 日样本下 |t|≥3 即越过 Bonferroni
校正后约 t≥3.9 的量级，见 §5 注意）。

门槛：|IC|≥0.015 且 |IR|≥0.3 且覆盖≥95% 且双半窗同号 且 残差保持≥50%
（残差 IC = 逐日对 18 champion 因子 OLS 残差化后的 Rank IC）。

**漏斗：65 个过 IC/IR/覆盖/双半窗 → 27 个过残差保持 → 15 个非 graveyard →
相关矩阵去重后 2 个独立信号族。**

15 个未测幸存者几乎全部属于同一潜因子（互相 |spearman|≈1，残差 IC 均在
0.025–0.032）：**"T-1 收盘在近期价格区间的位置"** 族；唯一独立的是 5 日
均量水平族。代表与关键指标：

| 代表 | 表达式族 | IC | IR | half1→half2 | 残差 IC | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| `alpha101__ALPHA006` | 开盘-成交量相关性 | +0.0526 | 0.409 | +0.068→+0.037 | +0.0298 | 稳定性最佳（half2 最高） |
| `jq110__JQ110_VOL_005` | 5 日均量水平 | -0.0486 | 0.340 | -0.055→-0.042 | -0.0296 | half2 反而更强 |
| （已排除）`alpha158__MIN5` 等 12 个 | 低价位/quantile 族 | +0.058 | 0.354 | +0.086→+0.031 | +0.039 | **alpha158_y graveyard**（71 因子漏斗全灭，禁重跑） |
| （残差消失）`alpha360__VOLUME4x/5x` | 量比 60 日回溯 | +0.046 | 0.34 | 稳定 | +0.014 | 残差保持 27%——量能信息 champion 已覆盖 |

共性观察：几乎所有幸存者 half1→half2 衰减 40–60%（信号在减弱但方向稳定）。

## 2. 多日分钟加速状态（Phase 1C，风险轨）

10 个 T-1 全天分钟分量 + 4 分量合成 `accel_score`（day_ret/tail30_mom/
max_gain/vol_ratio_d1 的横截面 rank 均值），n=650 日：

- **加速类分量全部负 IC 且双半窗同号**：close_pos -0.040（IR -0.30）、amp
  -0.036、path_dd -0.031、day_ret -0.027、max_gain -0.022、near_limit
  -0.017；量能分量正（vol_ratio_d1 +0.014）。合成 accel_score -0.018。
- **状态罕见且不持续**：accel_top（≥0.8 分位）占比 4.43%，T-2 连续概率仅
  0.28%（高换手池生态）。
- **尾部无区分度（判据 b' 不通过）**：已加速组 P(label<-5%)=20.06% vs
  未加速组 20.06%，bottom-decile -7.77% vs -7.89%——"已经加速→尾部崩跌"
  在本状态定义下**不成立**；已加速组均值甚至更高（+0.23% vs +0.02%），
  与 2026-09-12 研究文档"高动量样本均值不低"一致。
- **Champion Top10 与已加速状态零交集**：pred 覆盖期（2026-04→07，62 日）
  Top10 内 accel_top 占比 **0.00%**（全池 4.43%）——模型天然从不买已加速股。

## 3. 风险轨 overlay（Phase 2-risk）：拒绝

冻结 Champion pred × accel_score 软惩罚（Top20 内 `rank(pred) - λ·rank(accel)`），
λ 在训练半窗网格 {0.25..4} 按 Calmar 标定（选 λ*=0.5），测试半窗对照
（手工重放口径，仅排序参考不终判）：

| | 测试半窗累计 | Calmar |
| --- | --- | --- |
| λ=0（champion 原选择） | +12.5% | 11.02 |
| λ*=0.5 | +1.0% | 0.52 |

加速市/平静日 Δ日均 -0.21% / -0.18% 双双为负。**结论：拒绝**。机制与 §2
自洽：Top10 里没有加速股可惩罚，惩罚项只是扰动排名引入噪声；且状态本身
尾部无区分度。这与 graveyard"均匀追高惩罚已证伪"一致，本实验把结论扩展到
"条件化软惩罚 + 多日分钟状态"版本。

## 4. 分钟域 B1/B2（Phase 1B/1D）

收敛集（B1 全部 202 个 + B2 仅未测三库 alpha101/tdxgs/jq110 共 231 个；
alpha360/158 的 B2 与 graveyard/路线 A 语义重复、成本超 timebox 搁置）。

计算在全连续分钟序列上进行（GTJA/TDX 慢算子是成本大头，全量收敛集约
3–4 小时）。**收尾已自动化**：后台 finisher 等 `minute.py feat` 结束后自动
运行 `scripts/validate_factor_zoo_minute_metrics.py` 并把双门槛摘要写入
`reports/factor_zoo/minute_digest.txt`（同目录另有 minute_b1_metrics.csv /
minute_b2_t1_metrics.csv 全量指标）。本节结论以 digest 为准补写；判据同
§1（alpha 轨全门槛 + 残差保持≥50%），风险轨参照 §2 已整体拒绝。

## 5. 冻结协议 augment（Phase 2-alpha，timebox）：拒绝

选因子：路线 A 去重后两独立族代表 `alpha101__ALPHA006`（价量相关结构）
与 `jq110__JQ110_VOL_005`（5 日均量水平），物化为 `fz_*.day.bin` 研究 bin
（additive），18+2 冻结协议（同 HFLGB 超参/切分/label）：

| | IC | Rank IC | 手工重放 62 测日（top10 等权-成本） |
| --- | --- | --- | --- |
| 同批基线（18 因子重训） | 0.0603 | 0.0718 | 累计 +62.6%，Calmar 36.9 |
| augment（18+2） | **0.0496** | **0.0553** | 累计 +48.5%，Calmar 28.4 |

IC 劣化 0.0107，**超出 ±0.006 运行噪声带**；重放前半 Δ日均 -0.34%（显著
差）、后半 +0.04%（打平）。**判据 (e) 不通过——拒绝**。单因子层的残差增量
（§1）在模型层被稀释，复现 graveyard 的"单窗/单层亮眼、组合层蒸发"模式。

执行备注（本机环境坑，已修入脚本）：①task_train 按 `scripts.*` 路径二次
导入脚本模块，与 `__main__` 全局不共享（首跑 augment 与基线逐位相同即此
因）；②qlib 表达式引擎按**小写**定位 bin 文件且双下划线字段名解析失败
（逐段 bisect 实测），研究 bin 字段名须全小写+单下划线；③task_train 的
PortAnaRecord 回测段在本机两次死锁（0% CPU futex 等待），timebox 内改用
SignalRecord + 自带重放口径（只排序不终判），全漏斗（PortAna + purged
19 段）留待后续环境排查后执行。

## 6. 初步结论与边界

1. **zoo 对本策略的增量集中在"T-1 日频价格位置/量水平"两族**——它们是
   graveyard 未覆盖的表达式形态（价量相关结构、量水平），单因子层有真实
   残差增量（~0.03），但 half2 普遍衰减，且 71 个 Alpha158 同类前科表明
   单因子层通过 ≠ 漏斗通过；
2. **"已经加速"风险轨在当前定义下无效**——分量 IC 虽负，但尾部无区分度、
   状态零持续、champion 零暴露，软惩罚 overlay 测试半窗全面劣化。若后续
   仍想保留此方向，需要换状态定义（如 T-1 涨停炸板路径、竞价分歧）而非
   调 λ；
3. 多重检验声明：本日 screened=999（路线 A）+ 982×2（分钟域）+ 14（状态
   分量），任何"入选"都只进入 Phase 2 排序，不构成采纳证据；
4. 运行噪声提醒（excluded.json 规则 1）：§5 的单次对照结论均在 ±10pp
   PortAna 噪声带内解读。

## 环境备注

- 全部计算在 worktree `.worktrees/factor-zoo-research`，data/mlruns 软链主仓
  运行资产；`fz_*.day.bin` 研究 bin 以 additive 方式写入 overlay（沿用
  xd*/e94* 先例，去留待用户决定）。
- 测试：`tests/test_factor_zoo.py` 11 例离线通过（parse/ops_compat/screen_lib）。
