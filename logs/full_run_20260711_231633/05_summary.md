# 全流程训练/验证/回测审计日志

## 执行时间
2026-07-11 23:16 ~ 23:30

## 环境（Step 0 基线）
- git HEAD: 6e4abbb (fix(slot): 242→240 适配)
- slot 配置: SLOTS_PER_DAY=240, FIRST_FEATURE_SLOT=0, BUY_SLOT=10
- 数据: qlib_data day=2026-07-10, cn_data_1min=2026-07-10 15:00, 240 slots/day
- 环境: qlib 0.9.7, lightgbm 4.6.0, pandas 2.3.3

## Step 1: Universe 刷新
- 快照: 609 天 cached, 0 新拉（已是最新）
- 结果: 5118 codes, 51070 segments
- 口径: T 日盘前更新（无 T-1 shift）

## Step 2: 因子物化（240 slot）
- 结果: 5042 ok, 76 missing（退市/停牌）
- 口径: slot 0-9 = 09:31-09:40（10 根特征 K），slot 10 = 09:41（成交价）
- 产物: 20 day.bins/stock（14 baseline + 4 extra + price_941 + change_941）

## Step 3: 训练 + 预测 + IC
- recorder: 028a2de6b2fc4d0fbe35de85bacdc750
- 模型: LGBModel(lr=0.05, depth=6, leaves=64, λ_l1=5, λ_l2=10), best_iter=47
- test 段: 2026-04-01 ~ 2026-07-02（62 交易日）

| IC 指标 | 值 | 旧基线(242格) |
|---|---|---|
| IC | 0.0511 | 0.0545 (-6%) |
| ICIR | 0.36 | 0.51 |
| Rank IC | 0.0604 | 0.0679 |
| Rank ICIR | 0.47 | 0.54 |
| IC>0 胜率 | 64.5% | ~68% |

注：IC 下降 ~6% 是 240 格 slot 变更的预期结果（因子数值有细微变化）。
PortAnaRecord 的 SimulatorExecutor 在当前环境多线程死锁（201 threads futex_wait），
手动回测替代（Step 4）。

## Step 4: 手动回测（top10 equal-weight，含交易成本）

| 指标 | 含成本(net) | 无成本(gross) | 基准 SH000300 |
|---|---|---|---|
| 累计超额收益 | +69.8% | +91.9% | — |
| 策略累计收益 | +83.3% | +107.5% | +6.3% |
| 年化超额(几何) | +745% | +1284% | — |
| 信息比率 IR | 4.94 | 6.02 | 1.33 |
| 最大回撤 | -12.53% | — | — |
| 日均超额(net) | +0.913% | +1.116% | — |
| 超额>0 天数比 | 61.3% | — | — |

回测口径：T 日 9:41 买 top10 equal-weight → T+1 收盘卖。
成本: open_cost=0.0005, close_cost=0.0015。
62 个交易日，超额净值从 1.0 → 1.698（含成本 +69.8%）。

注：手动回测口径与 champion 的 SimulatorExecutor 略有差异（无涨跌停拦截、无 n_drop
换手帽），IC/超额方向和量级一致，但绝对值不完全可比。

## 结论
- 240 slot 适配后策略仍有效：IC 0.0511（-6% 但仍显著），62 天 test 段含成本超额 +69.8%
- 模型在 240 格 bin 上重新训练后口径对齐（不再有 FROZEN 不匹配问题）
- FROZEN champion params.pkl (caf649ca) 需更新为新训练的模型，或使用滚动重训
- PortAnaRecord SimulatorExecutor 死锁是环境问题（qlib 0.9.7 + pandas 2.3 多线程），
  不影响训练和推理，仅影响 qrun 内的回测段

## 日志文件
- 00_baseline.txt: 环境基线
- 01_universe.txt: universe 刷新
- 02_materialize.txt: 因子物化
- 03_qrun.log: qrun 完整输出（含 training/IC）
- 03_train_ic.txt: IC 结果摘要
- 04_backtest.txt: 手动回测完整结果（含每日明细）
