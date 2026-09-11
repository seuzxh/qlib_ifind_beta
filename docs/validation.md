---
layout: default
title: 验证与研究
nav_order: 10
has_children: true
---

# 验证与研究

本页汇总候选模型准入、风险叠加和已证伪的研究方向。核心原则：
**任何候选改动只有在前置验证段全面不差于基线才允许进入下一环节；验证结论存档，淘汰方向记录在案避免重跑。**

## 滚动验证方法论（purged rolling）

所有候选模型共用一套无泄漏窗口（`scripts/retrain.py`、`scripts/validate_xgb_purged_rolling_gate.py`）：

| 参数 | 值 | 说明 |
| --- | --- | --- |
| train 窗口 | 90 交易日 | 固定长度滑动 |
| valid 窗口 | 20 交易日 | 门控指标的评估段 |
| embargo | 1 交易日 | 隔离 valid label 与 test 起点（valid 末日 label 用 T+1 收盘 = embargo 日） |
| test 段 | 20 交易日 | 冻结段，段内不换模型 |
| 滑动步长 | 20 交易日 | 与 test 段对齐 |

embargo 隔离日的标签泄漏原理（T 日 label 依赖 T+1 收盘价）与防篡改校验，
详见[模型说明](models.md)。

## 候选门控（validation gate）

`qlib_ifind_beta/model_ensemble.py` 实现三项门控，XGBoost 候选只有在**全部三项**不低于
HFLGB 主模型时才启用固定 25% 混合权重：

1. Pearson IC；
2. Spearman RankIC；
3. 精确 TD0 口径的年化超额收益。

安全回退链：gate metadata 缺失或损坏 → 权重置 0 只用 HFLGB → 无日期匹配的 online
recorder → 回退冻结 Champion。任何异常都不会猜测权重。

## 治理流程：CANDIDATE → 人工晋升

```bash
# 盘后训练候选（产出 CANDIDATE 审阅材料，不触碰 online tag）
conda run -n qlib_ifind_beta python scripts/retrain.py --test-start YYYY-MM-DD

# 人工审阅 data/model_governance/candidate-<rid>.json 后，显式晋升
conda run -n qlib_ifind_beta python scripts/retrain.py \
    --promote-manifest data/model_governance/candidate-<rid>.json \
    --approved-by <reviewer>
```

- 训练只产出 `CANDIDATE` 状态的 manifest 和 validation 报告，**不修改 Qlib online tag**；
- `promote_candidate` 会先校验 gate metadata 与 recorder 来源，再切 online；
- 项目红线：禁止自动晋升候选模型（见 [AGENTS.md](../AGENTS.md)）。

## 风险叠加（risk_overlay，研究态）

`qlib_ifind_beta/risk_overlay.py` 把 T 日 09:40 前已知的信息转成软惩罚，只作用于模型
Top20 候选池内部，不修改 alpha 分数：

- `compute_chase_risk`：5 日动量 × 连续阳线 × 隔夜跳空-开盘延续 × 加速度的联合"追高"风险；
- `compute_tracking_risk`：跟踪风险分量；
- `apply_top_pool_penalty`：池内排名惩罚，输出仍可回 TopkDropoutStrategy。

研究结论：未通过生产准入，当前默认不启用。验证脚本 `scripts/validate_risk_overlay_purged.py`、
`scripts/validate_risk_overlay_quarters.py`。

## 已存档的研究结论

| 方向 | 结论 | 存档 |
| --- | --- | --- |
| 反弹穿越 + 回调后共同启动（23 特征合并模型） | IC 0.0621 超 Champion 0.0593，但超额回撤 -26.34% 不达标，未准入 | [joint TVT](backtest-log/2026-07-17-rebound-pullback-joint-tvt.md) |
| 自动特征合成（525 qlib 算子组合 + LGBM） | IC 0.039 低于 Champion 0.056，广撒网稀释强因子 | 会话记录（worktree 已清理） |
| 遗传规划因子挖掘（gplearn） | valid IC≈0，小规模过拟合 | 会话记录 |
| Kronos 基础模型 zero-shot（日线/5min、base/small） | 四次 IC 全负，方向相反 | 外部项目 `quant_projects/kronos`（有 finetune checkpoint） |
| Alpha158 日频因子（85 维） | 稀释 9:41 分钟信号，§19.1 证伪 | [backtest-log](backtest-log/2026-07-06-l1-full-backtest.md) |
| overnight_gap 后复权口径变体 | 信息量与名义口径持平（IC 差为 label 混合基准噪声所致），组合收益持平，不切换；**衍生发现：数据源 factor 逐日漂移 + label 基准噪声待立项** | [gap 口径 A/B](backtest-log/2026-09-11-gap-caliber-ab-and-factor-drift.md) |

## 验证脚本索引

| 脚本 | 用途 |
| --- | --- |
| `scripts/validate_xgb_purged_rolling_gate.py` | XGB 门控 purged rolling 验证（含 `purged_segments` 工具函数） |
| `scripts/validate_index_stage_joint_models.py` | 指数阶段反弹/回调联合模型验证 |
| `scripts/validate_factor_challengers.py` | 挑战者因子验证 |
| `scripts/validate_gap_adjusted.py` | overnight_gap 口径对照（名义 vs 后复权） |
| `scripts/validate_prediction_blend.py` | 预测混合验证 |
| `scripts/validate_risk_overlay_purged.py` / `_quarters.py` | 风险叠加验证（purged / 分季度） |
| `scripts/diagnose_index_stage_tail_association.py` | 指数阶段与尾部收益关联诊断 |
| `scripts/diagnose_minute_basket_resonance.py` | 分钟篮子共振诊断 |

另有 `scripts/rolling_validate.py`（滚动重训验证）与 `scripts/compare_models.py`
（LGBM/XGBoost/CatBoost/Linear 消融对比，非生产）。
