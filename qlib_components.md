# Qlib 组件速览与示例

> 基于 qlib 官方文档整理。按"数据 → 模型 → 组合回测 → 分析 → 实盘"流水线组织。
> 所有示例均为可直接运行的最小代码（数据需提前准备）。

## 目录

1. [Data Layer 数据层](#1-data-layer-数据层)
2. [Data Handler 因子加工](#2-data-handler-因子加工)
3. [Forecast Model 预测模型](#3-forecast-model-预测模型)
4. [Portfolio Management & Backtest 组合与回测](#4-portfolio-management--backtest-组合与回测)
5. [Nested Decision Execution 嵌套决策执行](#5-nested-decision-execution-嵌套决策执行)
6. [Workflow Management 工作流管理](#6-workflow-management-工作流管理)
7. [Recorder 实验记录](#7-recorder-实验记录)
8. [Analysis 分析评估](#8-analysis-分析评估)
9. [Meta Controller 元学习](#9-meta-controller-元学习)
10. [Online Serving 实盘衔接](#10-online-serving-实盘衔接)
11. [附录：风险模型数据接口（外挂 Barra 用）](#11-附录风险模型数据接口)

---

## 1. Data Layer 数据层

**职责**：自有二进制数据存储 + 表达式引擎，负责行情/因子的读取与计算。

- 数据按 instrument 逐文件存储，几十GB A股数据秒级读取
- `D.features` 支持字符串表达式，因子计算无需手写循环
- 支持日频/分钟频（`freq="day"` / `"1min"`）

### 示例

```python
import qlib
from qlib.data import D

qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")

# 取收盘价
df = D.features(["SH600519"], ["$close"], start_time="2020-01-01", end_time="2020-12-31")

# 表达式引擎：直接在字符串里写因子
df = D.features(
    D.instruments("csi300"),
    ["$close", "Mean($close, 10) / Ref($close, 10) - 1"],  # 10日均线偏离度
    start_time="2020-01-01",
)

# 分钟数据
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data_1min", region="cn")
df_min = D.features(["SH600519"], ["$close", "$volume"], freq="1min")
```

### 数据准备

```bash
# 日频数据
python -m qlib.cli.data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
# 分钟数据
python -m qlib.cli.data qlib_data --target_dir ~/.qlib/qlib_data/cn_data_1min --region cn --interval 1min
```

---

## 2. Data Handler 因子加工

**职责**：批量因子库 + 数据预处理流水线（缺失值、去极值、标准化），产出模型训练用的特征矩阵。

- 内置因子库：`Alpha158`（158个量价因子）、`Alpha360`（360个）
- `DatasetH` 管理 train/valid/test 切分，自动防数据泄漏

### 示例

```python
from qlib.utils import init_instance_by_config

dataset = init_instance_by_config({
    "class": "DatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "handler": {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "start_time": "2008-01-01",
                "end_time": "2020-08-01",
                "fit_start_time": "2008-01-01",
                "fit_end_time": "2014-12-31",
                "instruments": "csi300",
            },
        },
        "segments": {
            "train": ("2008-01-01", "2014-12-31"),
            "valid": ("2015-01-01", "2016-12-31"),
            "test":  ("2017-01-01", "2020-08-01"),
        },
    },
})

df_train = dataset.prepare("train", col_set=["feature", "label"])
df_test = dataset.prepare("test", col_set=["feature", "label"])
```

---

## 3. Forecast Model 预测模型

**职责**：对因子矩阵训练，产出股票打分（pred_score）。统一 `fit/predict` 接口。

- 内置：LightGBM/XGBoost、线性模型、LSTM/GRU/Transformer/HIST 等深度模型
- 只负责"打分"，不管交易——交易是 Strategy 的事

### 示例

```python
from qlib.utils import init_instance_by_config

model = init_instance_by_config({
    "class": "LGBModel",
    "module_path": "qlib.contrib.model.gbdt",
    "kwargs": {
        "loss": "mse",
        "learning_rate": 0.05,
        "num_leaves": 210,
        "max_depth": 8,
    },
})

model.fit(dataset)
pred_score = model.predict(dataset)  # pd.Series, MultiIndex(datetime, instrument)
```

---

## 4. Portfolio Management & Backtest 组合与回测

**职责**：策略（下什么单）+ 交易所规则（能否成交/价格/费用）+ 执行引擎（推进回测进程）+ 账户（记账）。

| 组件 | 职责 |
|------|------|
| Strategy | 每个bar决定买卖什么、多少量，产出订单 |
| Exchange | 撮合定价、停牌/涨跌停/成交量检查、费用计算 |
| Executor | 驱动回测循环，`time_per_step` 决定节拍（day/1min） |
| Account | 现金、持仓、净值记账 |

### 示例：经典日频回测

```python
from qlib.backtest import backtest, executor
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import risk_analysis

strategy = TopkDropoutStrategy(topk=50, n_drop=5, signal=pred_score)
exec_obj = executor.SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True)

report, positions = backtest(
    start_time="2017-01-01", end_time="2020-08-01",
    account=100_000_000,
    benchmark="SH000300",            # 可换成自定义池指数（见附录）
    exchange_kwargs={
        "freq": "day", "deal_price": "close",
        "open_cost": 0.0005, "close_cost": 0.0015, "min_cost": 5,
        "limit_threshold": 0.095,
    },
    strategy=strategy, executor=exec_obj,
)

report_df, positions_dict = report.get("1day")
print(risk_analysis(report_df["return"] - report_df["bench"]))
```

### 示例：自定义权重策略（动态仓位 / GC001 阀门）

```python
from qlib.contrib.strategy import WeightStrategyBase

class DynamicPositionStrategy(WeightStrategyBase):
    """权重和 < 1 的部分自动留现金；risk_degree 即仓位阀门。"""

    def get_risk_degree(self, trade_date):
        # 返回 0~1 的目标仓位，由择时/波动率信号决定
        ...

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time, **kwargs):
        w = self.get_risk_degree(trade_start_time)
        top = score.sort_values(ascending=False).iloc[:50]
        return {code: w / 50 for code in top.index}
```

---

## 5. Nested Decision Execution 嵌套决策执行

**职责**：两层策略协同——外层日频选股，内层分钟级执行（拆单/择时成交）。

```text
外层（day）：TopkDropoutStrategy → 目标持仓
    └─ 内层（1min）：SBBStrategyEMA → 逐分钟判断如何执行
```

### 示例

```python
from qlib.backtest import backtest, executor

# 内层：分钟级执行策略 + 分钟 executor
inner_strategy = ...  # 如 SBBStrategyEMA / 自定义盘中条件策略
inner_exec = executor.SimulatorExecutor(time_per_step="1min")

# 外层：日频组合策略
outer_strategy = TopkDropoutStrategy(topk=50, n_drop=5, signal=pred_score)

nested = executor.NestedExecutor(
    time_per_step="day",
    inner_executor=inner_exec,
    inner_strategy=inner_strategy,
    generate_portfolio_metrics=True,
)

report, positions = backtest(
    ..., executor=nested, strategy=outer_strategy,
    exchange_kwargs={"freq": "1min", ...},  # 底层行情用分钟
)
```

> 内层策略同样在每分钟被调用 `generate_trade_decision()`，可实现"9:31~9:40 区间内符合条件立即交易"。

---

## 6. Workflow Management 工作流管理

**职责**：一个 YAML 定义"数据→模型→回测→记录"全流程，`qrun` 一条命令跑完，便于批量实验。

### 示例 workflow.yaml

```yaml
qlib_init:
    provider_uri: "~/.qlib/qlib_data/cn_data"
    region: cn

market: &market csi300
benchmark: &benchmark SH000300

task:
    model:
        class: LGBModel
        module_path: qlib.contrib.model.gbdt
        kwargs: { loss: mse, learning_rate: 0.05, num_leaves: 210 }

    dataset:
        class: DatasetH
        module_path: qlib.data.dataset
        kwargs:
            handler:
                class: Alpha158
                module_path: qlib.contrib.data.handler
                kwargs:
                    start_time: 2008-01-01
                    end_time: 2020-08-01
                    fit_start_time: 2008-01-01
                    fit_end_time: 2014-12-31
                    instruments: *market
            segments:
                train: [2008-01-01, 2014-12-31]
                valid: [2015-01-01, 2016-12-31]
                test:  [2017-01-01, 2020-08-01]

    record:
        - class: SignalRecord
          module_path: qlib.workflow.record_temp
        - class: PortAnaRecord
          module_path: qlib.workflow.record_temp
          kwargs:
            config:
                strategy:
                    class: TopkDropoutStrategy
                    module_path: qlib.contrib.strategy.strategy
                    kwargs: { topk: 50, n_drop: 5, signal: <PRED> }
                backtest:
                    start_time: 2017-01-01
                    end_time: 2020-08-01
                    account: 100000000
                    benchmark: *benchmark
                    exchange_kwargs:
                        limit_threshold: 0.095
                        deal_price: close
                        open_cost: 0.0005
                        close_cost: 0.0015
                        min_cost: 5
```

```bash
qrun workflow.yaml
```

---

## 7. Recorder 实验记录

**职责**：实验管理——每次运行自动归档模型、预测、回测产物，支持检索复现（MLflow 风格）。

- `SignalRecord`：存 pred.pkl（预测打分）、sig_analysis（IC/ICIR）
- `PortAnaRecord`：存 report_normal.pkl、positions_normal.pkl、port_analysis.pkl

### 示例

```python
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord

with R.start(experiment_name="my_strategy"):
    model.fit(dataset)
    SignalRecord(recorder=R.get_recorder(), model=model, dataset=dataset).generate()
    PortAnaRecord(recorder=R.get_recorder(), config=port_analysis_config).generate()

# 读取历史实验
recorder = R.get_recorder(experiment_name="my_strategy")
pred = recorder.load_object("pred.pkl")
positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
```

> `positions_normal.pkl` 是外挂 Barra 归因的数据源。

---

## 8. Analysis 分析评估

**职责**：绩效指标与可视化。

- `risk_analysis`：年化超额、IR、最大回撤
- `analysis_position`：净值曲线、回撤图（plotly）
- `score_ic_graph` / `calc_ic`：信号 IC 分析

### 示例

```python
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.report import analysis_position

# 指标
excess = report_df["return"] - report_df["bench"] - report_df["cost"]
print(risk_analysis(excess))

# 图表
analysis_position.report_graph(report_df)
analysis_position.risk_analysis_graph(...)
```

> 注意：默认年化收益用算术累加（sum），跨工具对比时传 `mode="product"` 对齐口径。
> 局限：无 Barra 归因，需外挂（见附录）。

---

## 9. Meta Controller 元学习

**职责**：用元学习方法自动选择/加权多个模型，样本外自适应。研究性质较浓。

- `MetaModel` / `MetaTaskGen`：多任务样本生成与元模型训练
- 典型场景：市场风格切换时自动切换/加权底层模型

### 示例（示意）

```python
from qlib.contrib.meta.data_selection import MetaModelDS

meta_model = MetaModelDS(task_config=..., meta_config=...)
# 输入多任务训练集 → 学习"何时用哪个模型" → 样本外自动调度
```

> 建议核心链路稳定后再研究。

---

## 10. Online Serving 实盘衔接

**职责**：每日定时"取昨日前瞻数据 → 预测打分 → 更新策略 → 生成今日订单列表"的实盘循环。

### 流程

```text
每日定时触发
  → model.get_data_with_date(前一交易日)   # 取预测输入
  → model.predict(...)                      # 打分
  → strategy.update(score, ...)             # 更新策略
  → strategy.generate_trade_decision(...)   # 生成今日订单列表
  → 落盘供实盘系统消费
```

### 示例（示意）

```python
from qlib.contrib.online.operator import Operator
from qlib.contrib.online.user import UserManager

op = Operator(client=..., )
op.generate(date="2026-09-12", path="~/online_data")  # 生成当日订单列表
```

> 策略回测验证通过后，用这层对接模拟盘/实盘。

---

## 11. 附录：风险模型数据接口

qlib 内置 `qlib.model.riskmodel`（统计因子模型），但**生产建议接入基本面风险模型（如 MSCI BARRA）**。官方预留了标准目录格式：

```text
/path/to/riskmodel
├── 20210101
│   ├── factor_exp.{csv|pkl|h5}      # 因子暴露矩阵（股票 × 因子）
│   ├── factor_cov.{csv|pkl|h5}      # 因子协方差矩阵
│   ├── specific_risk.{csv|pkl|h5}   # 特质风险
│   └── blacklist.{csv|pkl|h5}       # 可选
└── ...
```

**一份数据两用**：

1. 喂 `EnhancedIndexingStrategy`（benchmark=池指数）→ 组合构建层带风格/跟踪误差约束
2. 喂外挂 Barra 归因模块 → 持仓归因读 `factor_exp`

```python
from qlib.contrib.strategy import EnhancedIndexingStrategy

strategy = EnhancedIndexingStrategy(
    riskmodel_root="/path/to/riskmodel",
    market="csi300",
    turn_limit=0.1,          # 换手约束
)
```

### 自定义池指数做 benchmark

benchmark 本质是"一个普通 instrument"——已发布指数直接填代码；自定义池子自己造 NAV 序列灌成 instrument：

```python
# 池内等权合成指数 NAV
nav = pool_daily_ret.mean(axis=1).add(1).cumprod()
# → 灌成 qlib 数据里的一个 instrument，如 "POOLBENCH"
backtest(..., benchmark="POOLBENCH", ...)
```

或事后手动算超额：`report_df["return"] - pool_bench_ret`，效果等价。
