# 全自动影子模拟盘设计（paper shadow account）

> 日期：2026-09-13
> 状态：设计已获用户批准（含 ponytail 最简化修正），待 spec 审阅
> 前置决策（用户 2026-09-13 确认）：全自动影子撮合 / 先回放补齐历史再前向 /
> 前向全自动 cron / 纯冻结 Champion（retrain 不进循环）

## 1. 目标与非目标

**目标**：把已通过 62 日验证的回放状态机转为无人值守的模拟盘——先批量回放
2026-07-21 → 最近已同步交易日（预计 2026-09-11），再由 cron 每交易日自动续跑，
持续累积净值样本。

**非目标（明确跳过）**：
- 9:41 实时拉 bar——9:31-9:41 bar 在 15:30 同步后即入 bin，与实时逐字节相同；
  实时路径验证是现役六步链（`intraday_production.py`）的职责
- 微信/邮件告警——cron 日志够用，需要时再加
- 盘后 retrain 循环、候选模型晋升——用户否决，评分恒用冻结 Champion
  `93d435e0`（`load_model_bundle(use_online=False)`）

## 2. 核心简化（ponytail）

模拟盘成交为系统模拟，非真实下单。**bin 同源等价**：15:30 数据同步后，
`HistoricalReplaySource` 读到的 T 日 9:31-9:41 bar 与 9:41 时实时拉取的完全一致，
且 T 日 factor 已就绪（无 stale-factor 问题）。因此前向日**零实时数据代码**，
完全复用已验证零漂移的 bin 读取路径。

## 3. 架构

```
回放补齐（一次性，手动，零新代码）:
  conda run -n qlib_ifind_beta --no-capture-output python -W ignore \
    scripts/replay_intraday_shadow.py \
    --start 2026-07-21 --end <最新已同步交易日> \
    --allow-unreferenced-scores \
    --output data/paper_shadow/

前向（每交易日一个 cron）:
  16:30  scripts/paper_shadow.py day --date T
         └─ 数据门禁 → run_paper_day(结转上日账本) → 追加 nav.csv
```

**账本 = 最新日期目录**：不新建 state.json。跨日结转 = 找
`data/paper_shadow/` 下 T 之前最新的 `<date>/`，读其
`positions_after_close.csv` + `cash_after_buy.json`。

## 4. 代码改动

| 改动 | 内容 |
|---|---|
| `scripts/replay_intraday_shadow.py` 重构 | 将 `run()` 内日循环体提取为模块级 `run_paper_day(...)`，批量回放与新日入口共用同一状态机（含 hold_days/sellable 结转、两阶段现金、模拟成交、收盘估值） |
| `scripts/paper_shadow.py` 新建 | `day --date T`：加载上日账本 → 数据门禁 → `run_paper_day`（`require_stored_parity=False`）→ 追加 `data/paper_shadow/nav.csv`；`report`：打印累计指标 |
| cron 一行 | `30 16 * * 1-5`（15:30 同步历史耗时 ~12 分钟，留 1 小时裕量） |
| `docs/paper-trading.md`、`AGENTS.md` | "已删除/暂停"表述改为新模拟盘流程与现状 |

净新增约 100 行（脚本 + 提取函数的签名胶水）。

## 5. 前向日数据流

```
08:30  既有 cron_update_universe 拉当日快照（不动）
16:30  paper_shadow day --date T
  ├─ 门禁（轮询，超时 60 分钟）：
  │    ① T 在 data/qlib_root/calendars/day.txt（同步完成的交易日）
  │    ② 样本股 1min bin 存在 T 日 09:41 bar（防止 daily 先于 1min 完成）
  │    非交易日（日历永不出现 T）→ 静默 exit 0
  ├─ bars / daily_info / prev_volumes ← HistoricalReplaySource（bin）
  ├─ 评分 ← 冻结 Champion bundle（predict_bundle_matrix）
  ├─ plan_topk_dropout(10/8) → 模拟成交（涨停/无bar拦截，费率 0.05%/0.15%，
  │   min 5 元）→ 先卖后买两阶段现金
  ├─ 收盘估值 valuation_close(T)（T 的 day.bin 已同步）
  └─ 落盘 <date>/ 全套产物（沿用回放文件名，可审计）+ 追加 nav.csv
```

`nav.csv` 列：`date, cash, market_value, nav`，追加式；日级明细在
`<date>/shadow_day_result.json`。

## 6. 错误处理（fail-closed）

- 任一步失败 → 当日目录留 FAIL 痕迹、**账本不推进**；次日结转仍用更早
  的最新完整日（缺失日事后 `paper_shadow day --date <缺失日>` 手动补跑，
  同日幂等覆盖）
- universe 快照 ≠ 100 只 → 当日失败（沿用回放闸门）
- 门禁超时（同步事故）→ exit 1，cron 日志留痕
- **宁缺日不编数据**：与 62 日回放引擎一致的语义

## 7. 测试与验证

- 新增 1 个单测：合成两个相邻日目录 fixture，断言账本结转
  （hold_days+1、sellable=quantity、cash 读取）与 nav.csv 追加正确
- 真验证 = 回放补齐实跑：引擎连续跑 ~38 个交易日（7/21→9/11）本身即
  回归测试；`replay_report.json` 的 `all_reconciled` / 状态字段全 PASS 为门槛

## 8. 已知边界

- 模拟盘验证的是 模型+策略+状态机 的逐日正确性，**不**验证盘中实时
  拉取路径（后者由六步链承担）——两者数据最终同源
- 回放段 7/21 之后无 pred.pkl 参照，走 `UNREFERENCED_FORWARD` 分支
  （7/02 之前的有参照日已由 62 日回放覆盖）
- 若 7/21→9/11 期间某日 1min 数据源缺失，引擎当日 fail-closed，
  补齐数据后重跑即可（报告会列出失败日）
