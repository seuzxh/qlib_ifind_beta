---
layout: default
title: 运维手册
---

# 运维手册

日常操作、故障排查与红线。环境硬约束：只使用 conda 环境 `qlib_ifind_beta`，
禁止系统 Python；测试必须离线可运行。

## 盘中生产（`scripts/intraday_production.py`）

可审计的 6 步子命令 + 2 个治理命令，逐段运行、产物落盘：

```bash
P="conda run -n qlib_ifind_beta --no-capture-output python -W ignore scripts/intraday_production.py"

$P preflight              # ① 盘前检查：数据、模型 manifest、日历
$P universe               # ② 读取 T 日 883926 成分快照，生成观察池
$P collect-factor-bars    # ③ 收集 09:31-09:40 分钟 bar + 质量门槛
$P score-and-plan-sells   # ④ Champion 打分 + 生成卖出计划
$P build-buy-orders       # ⑤ 生成买入订单（Top10 / n_drop=8，方案 B）
$P reconcile              # ⑥ 对账：订单/成交/持仓/现金
```

治理命令（不改变生产信号）：

```bash
$P retrain                # 盘后滚动候选训练（产出 CANDIDATE，不发布）
$P promote-model          # 显式晋升（人工审核后，写 approved_by）
```

## 历史影子回放

```bash
conda run -n qlib_ifind_beta python scripts/replay_intraday_shadow.py
```

62 日回放已通过（零漂移、逐日对账 PASS），产物见[产物地图](artifacts.md)。

## Champion 复现

```bash
conda run -n qlib_ifind_beta python -m scripts.build_overlay   # ① overlay（一次性）
conda run -n qlib_ifind_beta python scripts/materialize_minute.py  # ② 分钟因子物化
conda run -n qlib_ifind_beta python qrun/run.py qrun/workflow_minute_enhanced_tk10_nd8.yaml  # ③ 训练回测
conda run -n qlib_ifind_beta python -m pytest -q               # ④ 全量测试
```

## 常见故障排查

| 症状 | 原因 | 处置 |
| --- | --- | --- |
| 某股卖出 `fill_status=NO_BAR` | 当日无 9:41 bar（停牌/缺数） | 正常拦截，持仓顺延次日；检查 `bar_quality.json` |
| 旧持仓掉出股池后无行情 | 退出成分股不在默认查询集 | 已修（§36.3）：`D.features(list, ...)` 直查 bin，不走池过滤 |
| 分数全 NaN / 样本骤减 | `DropnaProcessor(feature)` 前视护栏生效 | 查该日 1min 数据源是否事故（如 2026-07-01 全市场缺失） |
| iFinD 401 / token 失败 | access_token 过期 | 自动刷新重试；持续失败查 `/home/zxh/qlib_data/.ifind_token` 与 refresh token |
| qrun 报 `limit_threshold` 类型错误 | YAML list 被 qlib 判为非法 | 用 `qrun/run.py` 入口（自动 list→tuple），勿直接 `qrun` |
| mlflow 报 FileStore 弃用异常 | mlflow 3.x 收紧 | 同上，`run.py` 已预置 `MLFLOW_ALLOW_FILE_STORE=true` |

## 数据与目录约定

- 只读源：`/home/zxh/.qlib/qlib_data/cn_data`（日线）、`/home/zxh/.qlib/qlib_data/cn_data_1min`（1 分钟）；
- 可写叠加层：`data/qlib_root/`（symlink 7 base + 自有衍生/分钟 bin）；
- 运行资产不入库：`data/`、`mlruns/`、`reports/`、`logs/`。

## 红线（违反即事故）

1. 禁止未来数据：特征只用 09:41 前闭合的分钟 K 线；日频字段进模型必须 lag 到 T-1；
2. `$price_941` 缺失不得回退 T 日收盘价（未来函数）；
3. 不自动提交券商订单；不自动晋升候选模型（见[验证与研究](validation.md)）；
4. 不硬编码 token / 凭证。

## 测试

```bash
conda run -n qlib_ifind_beta python -m pytest -q   # 全量，须 100% 通过
```

网络调用一律 mock；新增因子需覆盖正常、空数据、单股和边界时间四类用例
（见 [AGENTS.md](../AGENTS.md)）。
