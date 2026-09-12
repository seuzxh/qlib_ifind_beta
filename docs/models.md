---
layout: default
title: 模型说明
nav_order: 5
---

# 模型说明

## 生产 Champion

当前生产模型是 Qlib `HFLGBModel`，配置位于
`qrun/workflow_minute_enhanced_tk10_nd8.yaml`。

| 项目 | 现役配置 |
| --- | --- |
| 实验名 | `minute_enhanced_tk10_nd8` |
| 模型 | `qlib.contrib.model.highfreq_gdbt_model.HFLGBModel` |
| 损失 | `binary`，用于横截面 alpha 二分类排序 |
| learning rate | `0.05` |
| max depth / num leaves | `6` / `64` |
| L1 / L2 | `5.0` / `10.0` |
| 训练区间 | 2024-01-01–2025-12-31 |
| 验证区间 | 2026-01-01–2026-03-31 |
| 测试区间 | 2026-04-01–2026-07-02 |
| Champion recorder | `93d435e0ef20464784553949eb3859a5` |

模型输出的是股票横截面分数，不直接输出仓位或订单。策略按分数排序取 Top10，
`n_drop=8` 控制单次最多替换 8 个持仓；涨跌停、T+1 和缺失行情约束由交易层处理。

## Label 与推理对齐

```text
Ref($close1500, -1) / $close0941 - 1     # label v2（2026-09-12 采纳）
```

v2 两腿全部取 1min 序列原值（`close0941` = 09:41 收盘原值、`close1500` = 15:00
收盘原值），对应 T 日 09:41 买入、T+1 15:00 卖出，消除旧版
`Ref($close,-1)/$price_941-1` 的混合复权基准（验证与决策记录见
[backtest-log](backtest-log/2026-09-12-label-close1500.md)）。推理使用
`TopkDropoutStrategyTD0` 的同日执行语义。这一组时间对齐是模型合同的一部分，
不能单独替换模型而不重新验证。冻结 Champion（93d435e0）按旧 label 训练，
label 不进推理，生产评分不受影响；一切后续训练已切换 v2。

## 滚动候选与组合模型

`scripts/retrain.py` 维护一个固定窗口的候选流程：

- HFLGB 主模型：训练 90 个交易日，验证 20 个交易日，测试段前留 1 个交易日 embargo；
- 每 20 个交易日冻结一个测试模型段，目标日期优先加载覆盖该日期的 online recorder；
- XGBoost 候选使用相同数据、Handler、label 和切分，只替换模型为 `XGBModel`；
- 只有 Pearson IC、Spearman RankIC 和精确 TD0 年化超额收益三项都不低于 HFLGB，
  才允许使用固定 25% 的 XGBoost 分数权重；否则权重为 0；
- 任意 metadata、recorder 或候选加载失败，都安全回退到 HFLGB，缺少日期匹配模型时
  再回退冻结 Champion。

### embargo：测试段前的隔离日

滚动切分 `train 90 / valid 20 / embargo 1 / test ≤20` 中的那 1 个交易日隔离日
（实际示例：valid 至 03-30、embargo 为 03-31、test 从 04-01 起），用于防止
**标签泄漏**。label（v2 同理）的 T 日样本要用 T+1
收盘价才算完整：

```text
valid: … → 03-30 | 03-31（隔离日，谁都不用） | test: 04-01 → …
                   ↑ valid 末日样本的 label 本会"看到" 03-31 的收盘价
```

若 valid 紧贴 test，valid 尾部样本的 label 就提前"看到"了 test 起点的行情，
门控三项指标会被虚增。空开一日后，前段尾部样本的信息边界完全落在 test 开始
之前。该概念出自 purged cross-validation（López de Prado《Advances in
Financial Machine Learning》）。代码层还有防篡改校验：
`validate_gate_metadata` 会检查 `ensemble_gate.pkl` 中的 `embargo_days`，
无隔离证明的 gate 元数据无法启用 XGB 混合权重。

截至最新回放记录，候选状态为 `CANDIDATE/REVIEW`，因为绝对最大回撤恶化，
尚未自动或人工晋升为生产模型。项目禁止自动晋升候选模型；人工晋升命令与门控细节见
[验证与研究](validation.md)。

## 非生产研究模型

`scripts/compare_models.py` 还提供 LGBM、XGBoost、CatBoost、Linear 等模型的滚动
对比入口。这些用于研究和消融，不代表生产模型，也不能覆盖 Champion recorder。
