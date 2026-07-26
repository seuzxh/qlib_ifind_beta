# CLAUDE.md

本文件只补充 Claude Code 的项目约束；通用且权威的项目规则见
[AGENTS.md](AGENTS.md)，当前状态和操作入口见 [README.md](README.md)。

## 项目定位

这是基于 Qlib 的 883926 高贝塔指数增强项目。生产基线为 18 个开盘分钟因子、
HFLGB Champion、Top10/n_drop=8。09:31–09:40 形成特征，09:41 起按方案 B
先卖旧仓再买新仓。第一阶段只生成 CSV，由人工下单。

## 硬约束

1. 只使用 conda 环境 `qlib_ifind_beta`，禁止调用系统 Python。
2. 日频与一分钟源目录只读；可写叠加层为 `data/qlib_root/`。
3. 所有网络调用必须在测试中 mock，测试必须离线可运行。
4. 禁止未来数据、自动提交券商订单、硬编码凭证、自动晋升候选模型。
5. 涉及 Qlib API 或扩展机制时，先查询当前官方文档，再决定 Handler、
   Processor、DatasetH、Model、Strategy、Executor 或 Recorder 的归属。
6. 保留无关的未提交改动；清理运行资产或研究残留前先取得用户确认。

## 当前 Qlib 分层

| 层级 | 项目实现 |
|---|---|
| 数据叠加层 | `qlib_ifind_beta/overlay.py`、`materialize.py`、`materialize_minute.py` |
| 时变股票池 | `qlib_ifind_beta/universe.py` |
| 因子与 Handler | `minute_factors.py`、`highbeta_handler.py`、`minute_enhanced_handler.py` |
| 模型与候选门控 | Qlib HFLGB/XGBoost Recorder、`model_ensemble.py`、`evaluation.py` |
| 调仓策略 | `qlib_ifind_beta/td0_strategy.py` |
| 盘中生产 | `qlib_ifind_beta/live/intraday.py`、`realtime/` |
| 历史回放 | `qlib_ifind_beta/live/historical_replay.py` |
| 工作流配置 | `qrun/workflow_minute_enhanced_tk10_nd8.yaml` |

## 常用验证

```bash
conda run -n qlib_ifind_beta python -m pytest -q
conda run -n qlib_ifind_beta python qrun/run.py qrun/workflow_minute_enhanced_tk10_nd8.yaml
conda run -n qlib_ifind_beta python scripts/intraday_production.py --help
conda run -n qlib_ifind_beta python scripts/replay_intraday_shadow.py
```

## 文档与记忆

- 生产合同：
  `docs/superpowers/specs/2026-07-20-intraday-production-signal-design.md`
- 历史验证：
  `docs/backtest-log/2026-07-21-intraday-shadow-replay.md`
- 得到笔记主文档仍为“同花顺高贝塔值指数增强策略”
  (`note_id=1914664125624050528`)；只追加关键决策和里程碑，不写任何 secret。
- 代码和本地运行事实优先于外部笔记；修改外部笔记前需先读取最新正文。

## 修改前检查

- 已读取目标模块并确认 Qlib 分层。
- 能用 Qlib 原生扩展点时不另造抽象层。
- 因子只使用决策时点已知数据，输出索引保持 `(instrument, datetime)`。
- 新增分钟因子覆盖正常、空数据、单股和边界时间测试。
- 候选模型仅生成审阅材料，不自动替换 Champion。
