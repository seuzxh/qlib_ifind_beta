# qlib_ifind_beta

基于 Qlib 的 883926 高贝塔指数增强项目。生产基线为：

- 当日成分快照构成观察池；
- 09:31–09:40 十根闭合分钟K线生成18维特征；
- 冻结 HFLGB Champion 评分；
- Top10 / n_drop=8，方案B上午先卖旧仓再买新仓；
- 第一阶段只生成CSV，人工下单；候选模型必须人工审核后晋升。

当前不是自动交易系统，也不负责生产行情数据。

## 当前状态

- Champion：`minute_enhanced_tk10_nd8`，recorder `93d435e0ef20464784553949eb3859a5`。
- 2026-04-01～2026-07-02 的62日历史影子回放已通过：公共样本分数零漂移、逐日账务对账通过。
- 回放模拟收益 +39.27%、最大回撤 -19.81%；该结果不包含真实延迟和券商滑点。
- 2026-07-02 候选模型保持 `CANDIDATE/REVIEW`，未自动发布。
- 真实交易日影子/纸面跟踪已运行：2026-07-03～07-14 共 7 个信号日、6 个结算日
  （最新结算 2026-07-13，`data/live_nav.csv`）；2026-07-14 后无新信号记录，恢复时间待定。
- 2026-08-15 合并 Codex 研究线：`risk_overlay.py`、purged rolling 验证脚本、
  joint TVT 研究档案（`docs/backtest-log/2026-07-17-rebound-pullback-joint-tvt.md`，
  结论为未通过生产准入，Champion 不变）。

## 环境与数据

仅使用 conda 环境 `qlib_ifind_beta`：

```bash
conda run -n qlib_ifind_beta python -c "import qlib; print(qlib.__version__)"
conda run -n qlib_ifind_beta python -m pytest -q
```

只读行情源：

| 数据 | 路径 | 当前截止 |
|---|---|---|
| 日频 | `/home/zxh/.qlib/qlib_data/cn_data` | 2026-07-20 |
| 1分钟 | `/home/zxh/.qlib/qlib_data/cn_data_1min` | 2026-07-20 |

1分钟日历每天240根真实K线，09:31为 slot 0，09:41为 slot 10。项目通过
`data/qlib_root/` 构建可写 overlay；该目录和 MLflow/报告产物均不进入 Git。

## 现役入口

```bash
# 一次性或数据更新后重建 overlay
conda run -n qlib_ifind_beta python -m scripts.build_overlay

# 冻结 Champion 训练/回测复现
conda run -n qlib_ifind_beta python scripts/materialize_minute.py
conda run -n qlib_ifind_beta python qrun/run.py qrun/workflow_minute_enhanced_tk10_nd8.yaml

# 每日盘中 CSV 工作流（查看分阶段命令）
conda run -n qlib_ifind_beta python scripts/intraday_production.py --help

# 盘后滚动候选训练；默认不发布
conda run -n qlib_ifind_beta python scripts/retrain.py --test-start YYYY-MM-DD

# 历史影子回放
conda run -n qlib_ifind_beta python scripts/replay_intraday_shadow.py
```

## 目录

```text
qlib_ifind_beta/        可复用数据、因子、模型、策略与生产逻辑
  live/                 盘中状态转换、离线历史适配、冻结信号产物
  realtime/             实时行情适配与内存推理
scripts/                CLI 编排入口
qrun/                   Qlib 训练和回测配置
tests/                  离线测试
docs/                   当前生产合同和必要验证结论
data/                   overlay、快照、生产CSV和回放产物（忽略）
mlruns/                 模型与实验记录（忽略）
```

## 权威文档

- [GitHub Pages 文档首页](docs/index.md)：项目总览、18 个现役因子、模型与生产基线。
- [盘中生产设计](docs/superpowers/specs/2026-07-20-intraday-production-signal-design.md)：输入、产物、时间点和人工操作合同。
- [历史影子演练](docs/backtest-log/2026-07-21-intraday-shadow-replay.md)：62日回放、风险和候选训练结论。
- [项目规则](AGENTS.md)：下次开发必须遵守的最小约束。

其余早期架构、计划和研究日志是历史材料，不应作为现役操作依据。
