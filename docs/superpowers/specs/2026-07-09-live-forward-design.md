# 883926 champion 实战对接 P1 — 纸面前向跟踪系统设计

> 日期：2026-07-09
> 状态：**已实现（用户 2026-07-09 授权自主完成设计+实施，决策项延后至晨间清单）**
> 关联：technical-design §D6 + backtest-log §22/§33/§35（champion = enhanced(18)@topk10/nd8，commit `24b18dd`）
> 前置决策：§29.1350 列出的 3 个待决项之一「转实战对接」，用户选定此方向。

---

## 1. 背景与目标

champion（`enhanced(18)@topk10/nd8`，commit `24b18dd`）在 W1 test 段（2026-04→07）取得
+191.1% wc 超额 / woc +226.0% / IC 0.0545 / IR 5.41 / DD −5.59% / 换手 147.4%/日 / 成本侵蚀 34.9pp。
但这是**回测 test 段**——验证到此为止无法再深入（§35.5 已证明因子层 + 模型层 + 策略层三大杠杆
在 FROZEN label/strategy 下全部触顶，剩余杠杆全需用户授权：sticky holdings / 扩段 / 转实战）。

**P1 目标：把 champion 从「回测验证」推进到「每日产出可执行交易信号 + 前向纸面跟踪」**，
在**零实盘风险、零未知外部资源依赖**的前提下：

1. **验证 OOS 稳健性**：champion 训练截止 2025-12-31、test 截止 2026-07-02。2026-07-09 之后是
   champion 从未见过的真实未来。P1 每日前向产出信号 → 累积真实 OOS IC/超额 → 验证 champion
   是否真稳健（而非过拟合 test 段）。
2. **验证回测假设**：9:41 价可得性、涨跌停拦截、换手 147%、成本 16pp——这些回测假设在真实
   市场是否成立，P1 用真实 9:41 价 + 真实涨跌停逐日核对。
3. **为 P2/P3 铺路**：把 qlib `R.get_recorder().load_object('params.pkl') → model.predict()`
   这条**推理 pipeline** 跑通（当前项目只有 `qrun/run.py → task_train` 全量重训入口，缺独立推理
   路径），P2 实时 / P3 实盘复用此推理骨架。

### 为什么是 P1（纸面前向）而非 P2（实时）/ P3（实盘）

第一性约束：**项目当前没有任何实时分钟数据源、没有任何券商交易通道**。iFinD 仅有
`history_data`（历史批量）+ `data_pool`（成分快照）两个 API，**无实时推送**。cn_data_1min 是
兄弟项目 15:30 cron 的 T+0 收盘批量同步（非实时）。因此：

- **P2（T 日 9:41 实时下单）不可行**：无实时分钟数据 → 9:41 无法实时算因子。需先引入实时源
  （iFinD 实时 API / kline-fetcher 实时 / 券商 L1/L2）—— **进决策清单，需用户选型**。
- **P3（实盘）不可行**：无券商 API。**进决策清单，需用户选型**。
- **P1（T 日收盘后算 T 日信号）可行且零外部依赖**：cn_data_1min T+0 同步后 T 日 1min 齐全，
  收盘后跑 T 日信号 = 「事后算事前信号」，复用既有全部基础设施。

P1 是**唯一不阻塞于未知外部资源**的选项，故为本次自主实施范围。P2/P3 的外部依赖进决策清单。

---

## 2. Qlib 官方方案（context7 确认）

推理路径（区别于训练路径 `qrun → task_train`）：

```python
from qlib.workflow import R
recorder = R.get_recorder(
    recorder_id="caf649ca6aa44aac8dec8c4e5a252aef",
    experiment_name="minute_enhanced_tk10_nd8",
)
model = recorder.load_object("params.pkl")          # LGBModel 实例
pred = model.predict(dataset, segment="inference")      # pandas.Series, index=(instrument, datetime)
```

- `R.get_recorder` 按 `recorder_id` + `experiment_name` 从 mlflow 后端定位（本项目用
  `mlruns/` 文件后端）。
- `load_object("params.pkl")` 取回训练好的 `LGBModel`（**artifact 名根因**：champion record 未挂
  `ModelRecorder`，故 `LGBModel` 实例以默认名 `params.pkl` 持久化，而非 `trained_model`；spike
  2026-07-09 实测 `type(params.pkl)=qlib.contrib.model.gbdt.LGBModel`，`model.model.booster_` =
  冻结的 lightgbm Booster，num_trees=10 为 valid 早停结果）。
- `model.predict(dataset, segment)` 内部调 `dataset.prepare(segment, col_set="feature",
  data_key=DataHandlerLP.DK_L)` 拿已 transform 的 feature，喂 `booster_.predict`。

**DatasetH 构造**（复刻 champion `workflow_minute_enhanced_tk10_nd8.yaml` 的
`data_handler_config`）：

```python
from qlib.data.dataset import DatasetH
from qlib_ifind_beta.minute_enhanced_handler import MinuteEnhancedHandler

handler = MinuteEnhancedHandler(
    instruments="highbeta883926",
    start_time="2024-01-01",          # 含 train 段（learned processors 需 fit 段数据）
    end_time=T,                        # 扩展到 inference 日
    fit_start_time="2024-01-01",       # FROZEN = champion train 段
    fit_end_time="2025-12-31",         # FROZEN = champion train 段
    label=[...],                       # 仅用于 handler fetch label 列（P1 不读 label 值）
)
dataset = DatasetH(handler=handler, segments={"inference": (T, T)})
```

---

## 3. 项目适配方案

### 3.1 inference 口径正确性证明（P1 生命线）

**核心论断**：P1 的 `pred[T]` 与「把 T 日加入 champion 回测 test 段后产生的 pred[T]」逐位相同。

依据：
1. **handler 参数复刻**：start/fit_start/fit_end/processors 全部 = champion（仅 end_time 扩到 T）。
2. **learned processors fit 一致**：qlib `DataHandlerLP.process()` 在 `fit_start_time~fit_end_time`
   上 fit learned processors（Alpha158 默认含 `RobustZScoreNorm` 等）。P1 的 fit 段 = champion
   train 段（2024-01-01→2025-12-31）**完全相同** → train 段数据不变 → fit 统计量相同 →
   T 日 transform 口径 = champion 训练时 test 段（2026-04→07）transform 口径。**normalize 一致性
   自动保证**，无需 dump/复用统计量。
3. **L1 前视护栏一致**：`HighBetaAlpha158._DEFAULT_SHARED_PROCESSORS =
   [DropnaProcessor(feature)]`，P1 handler 默认挂上 → T 日 feature NaN 的 stock-day 被 drop，
   与回测同口径。
4. **model 冻结**：`load_object("params.pkl")` 取回的就是 champion 训练时的 LGBModel，
   不重训 → 决策边界冻结。

**前视护栏（零未来信息）**：
- 18 因子 = T 日 9:30-9:40 分钟数据 → T 日 9:41 已知（9:40 < 9:41）✓
- pred = 冻结 model × T 日 9:40 前因子 ✓
- **label = `Ref($close,-1)/$price_941-1`**：P1 inference 时**不算 label 值**（label 含 T+1 close，
  T 日收盘时未知）。handler 虽配 label（为 fetch 结构完整），但 P1 只取 feature→pred，不读 label。
  label 仅在 T+1 回填用于事后 IC 评估。
- 结论：**P1 零前视**。它在 T 日收盘后用 T 日 9:40 前因子还原「回测本应在 T 日 9:41 产生的信号」，
  与回测 test 段逻辑同构，仅延伸到训练后真实未来。

**Spike 实测（2026-07-09 `/tmp/spike_inference.py`）—— 零偏离铁证**：
`predict_day("2026-07-02")`（test 末日）vs champion `pred.pkl` 同日逐位对比 →
`matched=98/98`，`max|diff|=0.000e+00`（bit-exact），top10 集合 **10/10 完全一致**。
§3.1 论断经实测坐实，P1 inference 口径正确性成立。

### 3.2 数据流（T 日收盘后，约 15:35 触发）

```
cn_data_1min T+0 同步（兄弟项目 15:30 cron，非本项目维护）
        │
        ▼
[1] 增量更新 universe      → dump_universe 拉 T-1 p03473 快照（T-1 lag 无前视）
        │                    → 扩展 instruments/highbeta883926.txt
        ▼
[2] 增量物化 T 日 day.bins  → 池内每票 materialize_minute_instrument(code)（幂等全量重算）
        │                    → T 日 38 bin 齐全（price_941/change_941/18 因子/limit_up/down）
        ▼
[3] inference              → predict_day(T)
        │                    → load 冻结 model → 复刻 handler → predict → top10 score
        │                    → 附各票 price_941[T]/change_941[T]/limit_up[T]/limit_down[T]
        ▼
[4] settle_prev(T-1)       → 回填昨日信号的 sell_price = close[T]（= T-1 信号的 T+1 收盘）
        │                    → 涨跌停卖出拦截（change[T] <= limit_down[T] → 持有）
        ▼
[5] NAV 累积               → equal-weight top10，compound → live_nav.csv
```

**买入成交判定**（与回测 exchange 一致）：
- `change_941[T] >= limit_up[T]` → T 日 9:41 封涨停，**买不进**（剔除该票，顺延或留空）
- 否则 → 以 `price_941[T]` 成交

**卖出成交判定**（T+1 收盘）：
- `change[T+1] <= limit_down[T+1]` → T+1 封跌停，**卖不出**（持有至 T+2，递延）
- 否则 → 以 `close[T+1]` 成交

**成本**（与回测 exchange 一致）：`open_cost=0.0005 / close_cost=0.0015 / min_cost=5`。

### 3.3 组件设计（文件结构）

```
qlib_ifind_beta/live/                    # 新包（类比 data/ factors/）
├── __init__.py
├── inference.py                         # predict_day() 核心
├── track.py                             # record_signal / settle_prev / compute_nav
└── materialize_live.py                  # 池内增量物化（薄封装 materialize_minute）

scripts/
└── live_forward.py                      # 每日入口（orchestration: universe→materialize→predict→settle→nav）

data/                                    # 运行产物（不入 git，见 §6）
├── live_signals.csv                     # 每日 raw 信号
├── live_settle.csv                      # T+1 回填
└── live_nav.csv                         # NAV 曲线

tests/
└── test_live_inference.py               # CLAUDE.md 测试要求全覆盖
```

#### inference.py
```python
def predict_day(date: str,
                recorder_id: str = CHAMPION_RECORDER_ID,
                experiment_name: str = "minute_enhanced_tk10_nd8",
                topk: int = 10) -> dict:
    """T 日收盘后推理：返回 {date, topk 个 {code,score,price_941,change_941,limit_up,limit_down}, n_candidates}.

    口径与 champion 回测 test 段逐位一致（§3.1 证明）。
    """
    qlib.init(provider_uri=str(OVERLAY_ROOT), region="cn")
    recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=experiment_name)
    model = recorder.load_object("params.pkl")
    handler = MinuteEnhancedHandler(
        instruments="highbeta883926", start_time="2024-01-01", end_time=date,
        fit_start_time="2024-01-01", fit_end_time="2025-12-31",
        label=[LABEL_EXPR],
    )
    dataset = DatasetH(handler=handler, segments={"inference": (date, date)})
    pred = model.predict(dataset, segment="inference")          # Series (inst, datetime)
    row = pred.xs(date, level="datetime").dropna()
    # 附 price_941/change_941/limit_up/limit_down（D.features 单独取，≤ T 无前视）
    aux = D.features(D.instruments(market="highbeta883926"),
                     ["$price_941", "$change_941", "$limit_up", "$limit_down"],
                     start_time=date, end_time=date)
    ...
    return topk_list
```

#### track.py
- `record_signal(date, topk_list)` → 追加 `data/live_signals.csv`
- `settle_prev(prev_date, today)` → 回填 `close[today]` 作 sell_price，涨跌停拦截，写 `live_settle.csv`
- `compute_nav()` → equal-weight compound → `live_nav.csv`（gross/net 双轨）

#### scripts/live_forward.py
```bash
conda run -n qlib_ifind_beta python scripts/live_forward.py --date $(date +%F)
```
按 §3.2 顺序 orchestrate 五步，每步日志可追溯。

### 3.4 调度

P1 用 **cron**（非 qlib 内部调度）：

```cron
# 工作日 15:35（cn_data_1min 15:30 同步后留 5min 缓冲）
35 15 * * 1-5  cd /home/zxh/projects/3.qlib_ifind_beta && \
               conda run -n qlib_ifind_beta --no-capture-output \
               python scripts/live_forward.py --date $(date +\%F) \
               >> data/live_forward.log 2>&1
```

> cron 是否启用 = 决策清单项（用户定启用时机；P1 先实现 + 手动验证，不自动装 cron）。

---

## 4. 偏离点（vs qlib 原生）

| 维度 | qlib 原生 | P1 | 偏离原因 |
|---|---|---|---|
| 推理入口 | `qrun` 全量重训（`task_train`） | `R.get_recorder().load_object` + `model.predict` | qlib 原生无「单日推理」CLI，但 load/predict API 原生；P1 仅组装 |
| dataset segment | train/valid/test | 单 `inference` 段 | qlib DatasetH 原生支持任意 segment 名，零偏离 |
| handler fit 段 | 训练时 = train 段 | 推理时 fit 段 = 同一 train 段（FROZEN） | **严格一致**，零偏离 |
| 信号→成交 | `SimulatorExecutor` + `Exchange` 精确撮合 | P1 简化为 `change_941>=limit_up` 剔除 + `close[T+1]` 卖 | 纸面跟踪无需精确撮合；撮合逻辑与 exchange 判定同源（同表达式），仅简化状态机 |
| NAV | `PortAnaRecord` 全量回测产物 | 手写 equal-weight compound | P1 是前向增量，非批量回测；NAV 公式与 PortAnaRecord 同口径 |

**结论**：P1 仅在「推理入口组装」+「纸面撮合简化」两点偏离，均为 qlib 原生 API 的合法组合，
无自定义抽象层（遵守 CLAUDE.md 禁忌 #4）。

---

## 5. 测试（CLAUDE.md 测试要求）

`tests/test_live_inference.py`：

1. **口径零偏离单测（P1 正确性根基）**：`predict_day("2026-07-02")`（test 末日，回测已知）
   → 与 champion recorder 的 `pred.pkl` 在该日逐位对比，`max|diff| < 1e-6`。
2. **前视护栏**：mock `D.features`，断言所有调用的 `end_time ≤ T`（不读 T+1）。
3. **空数据/单票/边界**（CLAUDE.md 要求）：池为空、单只票、T 日非交易日、T 日 cn_data_1min 事故
   （因子全 NaN → L1 drop → 候选不足）。
4. **涨跌停拦截**：构造 `change_941 >= limit_up` 的票 → 不进 top10 成交集。
5. **settle_prev**：T+1 封跌停 → 持有递延；正常 → `close[T+1]` 成交。
6. **NAV**：equal-weight compound 数值正确（手算对照）。
7. **无网络**（CLAUDE.md 要求）：iFinD API（universe 增量）可 mock/回退——P1 universe 步骤
   独立可跳过（用既有 instruments 文件），inference 本身零网络。

---

## 6. 风险与已知边界

1. **universe 增量依赖 iFinD API + token 刷新**：refresh_token 有效期至 **2026-08-01**（临近！）。
   token 过期 → universe 停更 → P1 用最后一个快照继续（flag，候选池陈旧化）。**进决策清单**。
2. **cn_data_1min 同步事故**：如 2026-07-01 全市场 6/5521 事故 → T 日因子 NaN → L1 drop →
   候选不足。P1 记录事故标记，该日 NAV 持平（不强制交易）。
3. **OOS 分布漂移**：T 日因子分布若偏离 train 段 → RobustZScoreNorm 后失真 → IC 崩塌。
   **这正是 P1 要检测的**：若前向 IC 持续 > 0.03 则 champion 稳健；若崩塌则证明过拟合。
4. **纸面撮合简化**：不做精确撮合（如部分成交、滑点），NAV 与实盘必有偏差。P1 价值在
   「信号 OOS 稳健性 + 回测假设核对」，非「精确盈亏」。
5. **数据产物入 git**：`data/live_*.csv` 是运行产物 → 加 `.gitignore`（避免污染 repo）。

---

## 7. 决策清单（2026-07-09 晨间，需用户拍板）

P1 已自主实施。以下需用户决策（P1 不阻塞）：

1. **实时数据源 P2**：项目当前无任何实时分钟数据源。P2（T 日 9:41 实时下单）需选型——
   iFinD 实时推送 API？kline-fetcher 实时？券商 Level-1/2 行情？**需用户选**。
2. **交易通道 P3**：实盘需券商量化交易 API。**需用户选**。
3. **模型再训练节奏**：P1 用冻结 champion（commit `24b18dd`）。是否周期滚动 retrain（如月度
   扩窗）？**需用户定**（P1 默认冻结）。
4. **universe 增量是否纳入 P1 cron**：倾向纳入（P1 自包含），但依赖 iFinD API + token
   （2026-08-01 临期）。**需用户知情 token 风险 + 定是否启用 cron**。
5. **清理 §28 残留**：`docs/superpowers/{plans,specs}/2026-07-07-daily-index-factors*` 5 文件 +
   `config.py` 中 `MINUTE_FACTOR_TAIL_FIELDS`/`_OPENING_T1`/`_OPENING_T2`/`INDEX_OPENING_FIELDS`
   等 §25/§26/§29 已证伪死常量（rule #7）。**需用户确认是否清理**（不影响 P1）。
6. **sticky holdings 探索**（§35.5 唯一策略层杠杆）：需解冻 FROZEN label horizon（~1.5d），
   与短高频定位冲突。**可选后续，非 P1**。

---

## 8. 未来对齐路径（P1 → P2 → P3）

- **P1（本次）**：纸面前向跟踪，零外部依赖。验证 OOS 稳健性 + 回测假设。
- **P2（实时）**：引入实时分钟源 → T 日 9:41 实时算因子 + 实时 predict + 实时信号推送。
  复用 P1 的 `predict_day()` 推理骨架，仅替换数据源（cn_data_1min 批量 → 实时流）。
- **P3（实盘）**：接券商 API → P2 信号 → 委托下单。复用 P1/P2 的信号 + 涨跌停判定，
  仅替换成交层（纸面 → 真实委托）。

P1 的 `inference.py` / `track.py` 设计为「数据源无关」（inference 只吃 day.bin，不关心 bin 怎么
来）→ P2/P3 可平滑替换数据/成交层而不动推理核心。

---

## 9. 自审（spec self-review）

- [x] 无 TBD/TODO/占位符
- [x] 内部一致（§3.1 口径证明 ↔ §2 API ↔ §5 测试 1 互证）
- [x] 范围聚焦（单 P1 实现，P2/P3 进决策清单/未来路径）
- [x] 无歧义（成交判定、成本、NAV 公式全给出确切表达式）
- [x] qlib 分层归属：推理 = Workflow/Recorder（`R`）+ Workflow/Dataset（`DatasetH`）+
  Workflow/Model（`LGBModel`）+ Workflow/Feature Engineering（`MinuteEnhancedHandler`），
  全 qlib 原生组件组合，无自定义抽象层
