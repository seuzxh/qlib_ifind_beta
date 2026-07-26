# qlib_ifind_beta 项目规则

## 项目定位

基于 Qlib 的 883926 高贝塔指数增强项目。当前生产基线是 18 个开盘分钟因子、
HFLGB Champion、Top10/n_drop=8；第一阶段只生成 CSV，由人工下单。

## 环境与常用命令

- 只使用 conda 环境 `qlib_ifind_beta`，不要调用系统 Python。
- 测试：`conda run -n qlib_ifind_beta python -m pytest -q`
- Champion 复现：`conda run -n qlib_ifind_beta python qrun/run.py qrun/workflow_minute_enhanced_tk10_nd8.yaml`
- 盘中生产入口：`conda run -n qlib_ifind_beta python scripts/intraday_production.py --help`
- 历史影子回放：`conda run -n qlib_ifind_beta python scripts/replay_intraday_shadow.py`

## 数据和技术栈

- Python 3.12、pyqlib 0.9.7、pandas、NumPy、LightGBM/XGBoost、MLflow。
- 日线只读源：`/home/zxh/.qlib/qlib_data/cn_data`。
- 1分钟只读源：`/home/zxh/.qlib/qlib_data/cn_data_1min`，每天240根真实K线。
- `data/qlib_root`、`mlruns`、`reports` 和生产 CSV 都是运行资产，不提交 Git。

## 目录与约定

- `qlib_ifind_beta/`：可复用领域代码；`live/` 是生产与历史回放核心。
- `scripts/`：CLI 编排；现役生产入口只有 `intraday_production.py`。
- `qrun/`：Qlib 训练/回测配置；Champion 配置为 `workflow_minute_enhanced_tk10_nd8.yaml`。
- `tests/`：所有网络调用必须 mock；测试必须离线可运行。
- `docs/superpowers/specs/2026-07-20-intraday-production-signal-design.md`：生产合同。
- 禁止未来数据、自动提交券商订单、硬编码 token、自动晋升候选模型。
- 保留并避开无关的未提交改动；删除研究残留前必须先获得用户确认。

## 当前状态

2026-07-21 已完成 62 日历史影子回放：公共样本 Champion 分数零漂移、逐日对账通过；
仍需至少5个真实交易日影子运行。同分候选二级排序、旧持仓行情覆盖和回撤晋升门槛待完善。
