# 全流程训练/验证/回测审计日志

## 执行信息
- 时间：2026-07-11 23:47 ~ 23:54
- 日志目录：logs/full_run_20260711_234725/

## Step 1: 环境检查
- git: 6e4abbb (240 slot 适配)
- slot: SLOTS_PER_DAY=240, FIRST_FEATURE_SLOT=0, BUY_SLOT=10
- 数据: qlib_data→2026-07-10, cn_data_1min→2026-07-10 (240 slots/天)
- 环境: qlib 0.9.7, lgbm 4.6.0, pandas 2.3.3

## Step 2: Universe 刷新
- 快照: 609 天 cached, 0 API
- codes: 5118, segments: 51070
- 07-10 在册: 100 只 (T 日盘前口径)

## Step 3: 因子物化
- 5042 ok / 76 missing (退市/停牌)
- 口径: slot 0-9=09:31~09:40, slot 10=09:41
- 产物: 20 day.bins/stock
- 07-10 全池 clean=100/100

## Step 4: 训练 + 预测 + IC
- recorder: 14b9e5db88f24238b0f537127c0dadb3
- best_iter: 47 (train l2=0.931, valid l2=0.985)
- test: 2026-04-01~07-02, 62 天, pred 5776 行

| IC 指标 | 240格(新) | 242格(旧 caf649ca) | 变化 |
|---|---|---|---|
| IC | 0.0511 | 0.0545 | -6.2% |
| ICIR | 0.36 | 0.51 | -29% |
| Rank IC | 0.0604 | 0.0679 | -11% |
| IC>0 胜率 | 64.5% | ~68% | -3.5pp |

## Step 5: 回测（top10 equal-weight, test 62 天）

| 指标 | 含成本(net) | 无成本(gross) | 基准300 |
|---|---|---|---|
| 累计超额 | +69.8% | +91.9% | — |
| 策略累计 | +83.3% | +107.5% | +6.3% |
| IR | 4.94 | 6.02 | — |
| 最大回撤 | -12.53% | — | — |
| 日均超额 | +0.913% | +1.116% | — |
| 超额>0 | 61.3% | — | — |

月度：4月 +0.81%/日, 5月持平, 6月 +1.88%/日, 7月(2天)微负

### 同口径对比（top10 equal-weight gross）
| 口径 | 累计(gross) | 天数 | IR |
|---|---|---|---|
| 242格 caf649ca | 122.1% | 61 | 9.35 |
| 240格 14b9e5db | 107.5% | 62 | 6.45 |

同口径下超额差 ~14pp（122%→108%），与 IC -6% 基本一致。

### 口径差异说明
§33 champion 原始 +191% 超额是 SimulatorExecutor 回测（含涨跌停拦截+n_drop换手帽+双边成本），
与本回测（top10 equal-weight 手动，简化成本）口径不同，绝对值不可直接比。

## 结论
1. 240 slot 口径下策略仍有效：IC 0.0511, 含成本超额 +69.8%/62天
2. 同口径对比旧 242 格：gross 超额 122%→108%（-14pp），与 IC -6% 一致
3. FROZEN champion caf649ca 需更新为 14b9e5db（240 格口径）
4. PortAnaRecord SimulatorExecutor 死锁是环境问题（qlib 0.9.7 + pandas 2.3 多线程）

## 日志文件清单
- step1_env.txt: 环境检查
- step2_universe.txt: universe 刷新
- step3_materialize.txt: 因子物化
- step4_qrun.log: qrun 完整输出
- step4_train_ic.txt: 训练+IC 摘要
- step5_backtest.txt: 回测完整结果
