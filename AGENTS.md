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

- 研究分支命名（2026-09-26 用户约定）：**「模块名｜持续目标」**，如
  `factor-zoo｜分钟因子匹配与替代挑战`；worktree 目录用模块名短路径。
- `qlib_ifind_beta/`：可复用领域代码；`live/` 是生产与历史回放核心。
- `scripts/`：CLI 编排；现役生产入口只有 `intraday_production.py`。
- `qrun/`：Qlib 训练/回测配置；Champion 配置为 `workflow_minute_enhanced_tk10_nd8.yaml`。
- `tests/`：所有网络调用必须 mock；测试必须离线可运行。
- `docs/superpowers/specs/2026-07-20-intraday-production-signal-design.md`：生产合同。
- 禁止未来数据、自动提交券商订单、硬编码 token、自动晋升候选模型。
- 保留并避开无关的未提交改动；删除研究残留前必须先获得用户确认。

## 当前状态

2026-07-21 完成 62 日历史影子回放（零漂移、逐日对账通过）；2026-07-03～07-14 完成
真实交易日纸面跟踪 7 个信号日 / 6 个结算日（达到"至少 5 日"门槛，最新结算 07-13），
2026-07-14 后无新信号记录、恢复待定。2026-08-15 合并 Codex 研究线（risk_overlay、
purged rolling 验证脚本、joint TVT 档案——未通过准入，Champion 不变）。
候选模型保持 `CANDIDATE/REVIEW`，禁止自动晋升。

2026-09-13 上线全自动影子模拟盘：回放引擎日增量化（`paper_shadow.py` + 16:30 cron），
评分恒用冻结 Champion。回放补齐 7/21→9/11 共 39 日完成（净 +12.17%，最大回撤 -11.11%，
期末 1,121,694）；其中 8/27、9/11 为 rally 日封板股致特征 <80，门槛已改为机制下限
（数据完整性仍由 bar gate 把守）。9/14 起 cron 前向，`report` 查看净值。

2026-09-24 开出因子研究分支 `factor-zoo｜分钟因子匹配与替代挑战`（原名 research/factor-zoo-20260924，
2026-09-26 按用户新约定「模块名｜持续目标」重命名；worktree
`.worktrees/factor-zoo-research`）：探索 qlib-factor-zoo 六库 ~1013 因子与
09:41 分钟策略的匹配度，核心路线是表达式移植到分钟域（当日盘初 10 根 +
多日 T-1/2/3/5 全天 240 根；多日族用于识别"已经加速"状态并做尾部风险
控制）。方案见
`docs/research/2026-09-24-factor-zoo-minute-research-plan.md`。

执行结论（详见 `docs/backtest-log/2026-09-24-factor-zoo-screen.md`）：
999/1007 表达式在 pyqlib 0.9.7 可求值（vendored+shim）。2026-09-25 替代
挑战战役：**C1（close_pos×3 → b1 域 MAX5/MIN5/QTLD5）以 purged 16 段胜率
92%、双窗 IC 翻倍成为史上最强挑战者，但 TD0 官方协议两窗合计净输
（W1 +16.7% vs +12.2% 胜、W2 +9.9% vs +21.1% 败）——机制：C1 Top10 中
49% 在 09:41 已涨停（禁买）vs Champion 仅 0.4%，排序 alpha 集中在生产
合同买不到的名单上。无可替代方案成立，Champion 18 因子不变。**
PortAna 死锁已三层修复（threading 后端/dayok universe/显式 codes），
TD0 回测通道恢复可用。盘初极值类因子今后应先过"Top10 撞涨停率"闸门。
工作区 data/mlruns 为软链主仓运行资产。
