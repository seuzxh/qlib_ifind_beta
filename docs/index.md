---
layout: default
title: qlib_ifind_beta 项目文档
---

# qlib_ifind_beta

基于 Qlib 的 883926 高贝塔指数增强项目。系统在交易日早盘使用当日成分快照和
09:31–09:40 的十根分钟 K 线生成信号，09:41 生成买入价格与 CSV，最终由人工下单。

> 本站面向项目交接、研究复核和生产操作。模型、因子和运行状态以仓库当前代码与
> `qrun/workflow_minute_enhanced_tk10_nd8.yaml` 为准；日期标记为 2026-08-17。

## 现役结论

- 因子：18 维，14 个 T 日开盘分钟因子 + 3 个多日量能因子 + 1 个隔夜跳空因子。
- 模型：Qlib `HFLGBModel`，`loss=binary`，冻结 Champion recorder。
- 组合：Top10，`n_drop=8`，使用 `TopkDropoutStrategyTD0` 在 T 日 09:41 执行。
- 交易合同：label 为 `Ref($close, -1) / $price_941 - 1`，即 T 日 09:41 买入、
  T+1 收盘卖出；第一阶段只输出 CSV，不自动提交券商订单。

## 快速导航

- [全流程实跑导读](pipeline-walkthrough.md)：2026-09-06 全流程重跑实录，逐环节输入/输出与数据全旅程（新接手先读这篇）。
- [项目总览](project-overview.md)：数据、时序、目录和当前边界。
- [因子说明](factors.md)：18 个现役因子、计算窗口和前视约束。
- [模型说明](models.md)：Champion、滚动候选、XGBoost 配对与晋升门控。
- [生产基线](production-baseline.md)：日内流程、输入输出和人工操作合同。
- [验证与研究](validation.md)：purged rolling 门控、风险叠加、已证伪方向存档。
- [产物地图](artifacts.md)：训练/验证/回测/生产各环节产物位置与字段。
- [运维手册](operations.md)：生产子命令、故障排查、红线。
- [架构与数据流](architecture.md)：MVP 期架构快照（历史，带弃用标记）。
- [历史影子回放](backtest-log/2026-07-21-intraday-shadow-replay.md)：62 日回放验收与待办。

## 运行入口

```bash
conda run -n qlib_ifind_beta python -m pytest -q
conda run -n qlib_ifind_beta python qrun/run.py qrun/workflow_minute_enhanced_tk10_nd8.yaml
conda run -n qlib_ifind_beta python scripts/intraday_production.py --help
conda run -n qlib_ifind_beta python scripts/replay_intraday_shadow.py
```

## 文档分层

`factors.md`、`models.md` 和 `production-baseline.md` 是当前状态的面向读者文档；
`superpowers/specs/`、`superpowers/plans/` 和 `backtest-log/` 保留研究设计、过程记录
和历史证据，不自动代表现役方案。
