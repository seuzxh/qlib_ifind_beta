---
layout: default
title: 配置说明
nav_order: 7
---

# 配置说明：qrun YAML 与运行链路

训练与回测的最后一步由 `qrun/run.py <配置.yaml>` 驱动，**yml 是训练/回测口径的
单一事实来源**：Handler 窗口、label、模型超参、策略参数、交易成本全部由它声明。
本文说明现役 Champion 配置的逐段含义、qrun/ 下其余 yml 的定位，以及 yml 与
盘中生产链路的关系。

> 注意区分：`docs/_config.yml` 是本文档站的 Jekyll 主题配置，与策略运行无关。

## qrun/ 配置文件一览

| 文件 | 定位 | 状态 |
|---|---|---|
| `workflow_minute_enhanced_tk10_nd8.yaml` | ★ **现役 Champion**：18 因子 + Top10/n_drop=8 | 生产基线，冻结 |
| `workflow_minute_enhanced_tk10_nd8_w2.yaml` | W2 OOS 对照窗（test 2025-04→07），防 W1 单窗过拟合 | 研究对照，可重跑 |
| `workflow_minute_enhanced.yaml` | 上代 champion（experiment `minute_enhanced`，n_drop=15 时代） | legacy，勿用于生产 |
| `workflow.yaml` | MVP：日频 Alpha158 全链路 | 历史（其 Universe 注释仍是 T-1 lag 旧口径，现役为 T 日盘前更新，见下） |
| `workflow_smoke.yaml` | 2025 子窗口烟雾测试（train 9 月/valid 1 月/test 2 月） | 验证后可删 |

研究新想法的正确姿势是**复制成新 yml**（如 W2 那样），不要改动 Champion 配置。

## Champion YAML 逐段图解

### ① qlib_init / 实验名

```yaml
qlib_init:
    provider_uri: /home/zxh/projects/3.qlib_ifind_beta/data/qlib_root   # overlay 叠加层
    region: cn
experiment_name: minute_enhanced_tk10_nd8        # MLflow 实验 = recorder 归属地
```

### ② 数据窗口与 label（`data_handler_config`）

| 字段 | 值 | 含义 | 冻结 |
|---|---|---|---|
| `start_time / end_time` | 2024-01-01 → 2026-07-02 | Handler 取数窗口（约 2.5 年） | ✅ |
| `fit_start_time / fit_end_time` | 2024-01-01 → 2025-12-31 | 学习型 processor 拟合窗口（= train 段） | ✅ |
| `instruments` | `highbeta883926` | 时变股池：T 日在册集（每段 `[d_in, d_out]` 逐日精确） | ✅ |
| `label` | `Ref($close,-1) / $price_941 - 1` | **FROZEN label**：T 日 09:41 买入、T+1 收盘卖出的收益 | 🔒 改=新 Champion |

> 股池无前视：883926 是每日重平衡高贝塔榜，instruments 按历史每日真实名单写入；
> qlib 取 T 日数据时自动只返回 T 日在册的约 100 只（详见[因子说明](factors.md)与
> [产物地图](artifacts.md)）。

### ③ 策略与回测（`port_analysis_config`）

```yaml
strategy: TopkDropoutStrategyTD0   # topk=10 / n_drop=8 / hold_thresh=1（T+1）
backtest: 2026-04-01 → 2026-07-01  # test 段（末日收在 07-01，避免日历末越界）
account: 100000000                 # 初始资金 1 亿
exchange_kwargs:
    deal_price: ["$price_941", "$close"]          # 买=09:41 价，卖=T+1 收盘
    limit_threshold:                              # 涨跌停拦截（表达式型）
        - $change_941 >= $limit_up                #   买：09:41 已涨停 → 禁买
        - $change <= $limit_down                  #   卖：当日跌停 → 禁卖
    open_cost: 0.0005   close_cost: 0.0015   min_cost: 5
```

| 参数 | 值 | 含义 |
|---|---|---|
| `topk` | 10 | 每日目标持仓 10 只（按预测分取头部） |
| `n_drop` | 8 | 单日最多替换 8 只（80% 换手帽） |
| `hold_thresh` | 1 | 最少持有 1 日（A 股 T+1 制度约束） |
| `forbid_all_trade_at_limit` | true | 涨跌停禁交易（依赖 change/limit_up/limit_down 物化字段） |

### ④ 模型（`task.model`）

`HFLGBModel`（qlib 高频 LightGBM 封装）：`loss=binary`、`learning_rate=0.05`、
`max_depth=6`、`num_leaves=64`、`lambda_l1=5`、`lambda_l2=10`、`num_threads=20`。
实测产物为单 Booster（约 12 棵树 × 18 特征），毫秒级推理。与 XGBoost 候选的
关系见[模型说明](models.md)。

### ⑤ 切分（`task.dataset.segments`）

| 段 | 区间 | dropna 后样本 | 用途 |
|---|---|---|---|
| train | 2024-01-01 → 2025-12-31 | 42,122 行 / 480 天 | 训练 |
| valid | 2026-01-01 → 2026-03-31 | 5,097 行 / 56 天 | 早停与门控 |
| test | 2026-04-01 → 2026-07-02 | 5,776 行 / 62 天 | 样本外评估（62 日影子回放同窗） |

### ⑥ 记录链（`task.record`）

`SignalRecord`（pred/label.pkl）→ `SigAnaRecord`（逐日 IC/RankIC）→
`PortAnaRecord`（TD0 策略回测报告），全部落进 `mlruns/<实验>/<recorder>/`。

## 运行时如何加载 yml（qrun/run.py）

`run.py` 等价 qlib 官方 `qlib.cli.run.workflow`，额外修两个本机坑：

1. **limit_threshold list → tuple**：qlib `Exchange._get_limit_type` 用
   `isinstance(..., tuple)` 判分支，YAML 的 `[a, b]` 加载成 list 会落到
   `NotImplementedError`；run.py 加载后把它转成 tuple，表达式型涨跌停拦截才能生效。
2. **`MLFLOW_ALLOW_FILE_STORE=true`**：mlflow 3.12 把 file store 列入维护模式，
   不预置该环境变量时 `task_train` 落盘会抛异常。

执行链：`加载 yml → qlib.init(provider_uri=overlay) → task_train → recorder 落
mlruns`。实测全流程（含 PortAna 回测）约 **13 秒**（2026-09-06 重跑）。

## yml 与生产链路的关系（关键）

```text
qrun/workflow_minute_enhanced_tk10_nd8.yaml          scripts/retrain.py（不读 yml）
        │ qrun/run.py 训练                               │ 程序化模板：超参与 Champion 相同，
        ▼                                                │ 窗口按 90/20/embargo1/test≤20 滚动
mlruns/1568…/93d435e0…/  ←—— 冻结 Champion ———          ▼
  params.pkl / pred.pkl / task                    minute_enhanced_rolling(_xgb)
        │                                              （CANDIDATE，人工晋升前不生效）
        ▼
qlib_ifind_beta/config.py  ←  CHAMPION_RECORDER_ID / CHAMPION_EXPERIMENT /
        │                    CHAMPION_LABEL_EXPR / CHAMPION_TOPK 等冻结常量
        ▼
盘中生产（intraday_production / 影子回放）按 recorder_id 加载冻结产物
```

要点：

- **生产不直接读 yml**。`scripts/intraday_production.py` 与回放通过
  `config.py` 的冻结常量找到 Champion recorder，从 `params.pkl` 加载模型本体；
  yml 的作用是"定义怎么训练出这个 recorder"。
- **滚动重训也不读 yml**。`retrain.py` 用 `_build_task_template()` 程序化生成
  任务（模型超参与 Champion 完全一致，仅窗口滚动），XGB 任务是对 HFLGB 任务
  深拷贝后只替换 model 段。
- 因此改 yml ≠ 改生产：改 yml 只影响你**下一次训练**的产物；生产切换模型的唯一
  途径是 Candidate → 人工批准晋升（见[模型说明](models.md)）。

## 修改守则

| 想改什么 | 能否直接改 Champion yml | 后果/正确姿势 |
|---|---|---|
| 注释、格式 | ✅ | 无 |
| `experiment_name` | ⚠️ | 会把新产物记到别的实验；研究应复制新 yml |
| 切分窗口 | ⚠️ | 属于新实验设计；复制新 yml（参考 W2 的做法） |
| label / deal_price / 涨跌停口径 | 🔴 FROZEN | 任何变动都构成新交易合同，必须完整重验证 |
| topk / n_drop / hold_thresh | 🔴 FROZEN | 同上（策略层 sweep 结论已冻结 10/8/1） |
| 模型超参 | 🔴 FROZEN | 改后属于候选模型，走 retrain + 门控 + 人工晋升 |
| 因子公式（yml 外的 minute_factors.py） | 🔴 FROZEN | 改后需 `materialize_minute` 重物化 + 零漂移检查 |

一句话：**Champion yml 是合同文本，不是调参面板**。要实验，复制新文件；
要换生产基线，走验证与晋升流程。
