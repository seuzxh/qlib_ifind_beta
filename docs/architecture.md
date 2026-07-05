# 架构文档 · qlib_ifind_beta

> 标的：**883926（同花顺高贝塔值指数）成分股**增强策略 MVP。
> 形态：日频 `Alpha158` 全链路（因子 → 模型 → 回测 → 报告）。
> 本文档描述**当前已实现的 as-is 架构**（基于实际代码，非设计稿）。技术选型与决策依据见 [technical-design.md](technical-design.md)。

---

## 1. 项目定位

一个**数据工程极薄、模型/回测全用 qlib.contrib 原生类**的 A 股因子挖掘 MVP：

- **数据**：只读消费 [`/home/zxh/qlib_data`](../../../qlib_data)（日频，7 字段，26 年深度），不生产行情数据。
- **唯一自研代码**：把只读 qlib_data 之上**叠加一层最小 overlay**（symlink 复用 + 自有衍生 bin），满足 qlib `Exchange` 涨跌停拦截对 `$change / $limit_up / $limit_down` 的字段依赖。
- **因子/模型/策略/执行/记录**：全部 `qlib.contrib` 原生类，零自定义子类。

> 项目状态：**搭建/设计阶段已完成 MVP 跑通**（未初始化 git）。已知妥协与后续方向见技术方案文档第 5、8 节。

---

## 2. 系统分层

```
┌─────────────────────────────────────────────────────────────────┐
│  外部数据源（只读）                                                │
│  ┌───────────────────────────┐   ┌────────────────────────────┐ │
│  │ /home/zxh/qlib_data       │   │ iFinD quantapi (HTTPS)     │ │
│  │ 7 字段 day.bin × 6419 天  │   │  · data_pool p03473        │ │
│  │ instruments/all.txt 等    │   │    （883926 时变成分股）     │ │
│  │ calendars/day.txt         │   │  · history_data（备用）     │ │
│  └─────────────┬─────────────┘   └─────────────┬──────────────┘ │
└────────────────┼────────────────────────────────┼────────────────┘
                 │ symlink 复用                    │ HTTP（token 鉴权）
                 ▼                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  数据工程层  qlib_ifind_beta/（自研包，~806 行）                  │
│  overlay → materialize → universe → ifind → binio → config      │
└────────────────┬────────────────────────────────────────────────┘
                 │ 生成
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Overlay provider_uri  data/qlib_root/（.gitignore，可重建）      │
│  calendars/  → symlink                                           │
│  instruments/  all.txt→symlink + highbeta883926.txt 真实         │
│  features/<code>/  7 base bins→symlink + 3 衍生 bin 真实         │
└────────────────┬────────────────────────────────────────────────┘
                 │ qlib.init(provider_uri=…)
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  qrun 全链路  qrun/run.py + workflow.yaml                        │
│  Alpha158 → DatasetH → LGBModel → TopkDropoutStrategy           │
│           → SimulatorExecutor → SignalRecord/SigAna/PortAna     │
└────────────────┬────────────────────────────────────────────────┘
                 │ 产出
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  mlruns/（.gitignore）  pred.pkl / label.pkl / IC / nav / 回撤   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 目录结构

```
3.qlib_ifind_beta/
├── qlib_ifind_beta/              # 数据工程包（自研，唯一业务代码）
│   ├── __init__.py              #  12 行 · 包说明 + __version__
│   ├── config.py                #  43 行 · 集中配置（路径/字段/代码/URL）
│   ├── binio.py                 #  53 行 · .day.bin 原始读写（FileFeatureStorage 布局）
│   ├── overlay.py               # 103 行 · symlink farm 构建
│   ├── materialize.py           #  89 行 · 衍生字段物化（change/limit_up/limit_down）
│   ├── universe.py              #  98 行 · 883926 成分股（iFinD p03473）
│   ├── dump_index.py            #  97 行 · SH883926 行情 dump（当前未启用，见技术方案 §5）
│   └── ifind.py                 # 211 行 · iFinD HTTP client + token 管理
├── scripts/
│   └── build_overlay.py         #  75 行 · 一次性编排（建 overlay 端到端）
├── qrun/
│   ├── workflow.yaml            # 全量配置（2024-01-01 → 2026-07-02）
│   ├── workflow_smoke.yaml      # 烟雾测试（2025 子窗口）
│   └── run.py                   #  80 行 · qrun 等价入口（绕两个本机坑）
├── data/qlib_root/              # 生成的 overlay（.gitignore，build_overlay 可重建）
├── mlruns/                      # qlib 实验产物（.gitignore）
├── docs/                        # 本文档所在
├── CLAUDE.md                    # 项目工程约束（conda-only / context7 / sequential-thinking）
└── .gitignore
```

---

## 4. 数据工程包模块详解（`qlib_ifind_beta/`）

### 4.1 [config.py](../qlib_ifind_beta/config.py) — 集中配置
所有路径、字段口径、标的代码、iFinD URL 的单一事实源。关键常量：

| 常量 | 值 | 说明 |
|---|---|---|
| `QLIB_DATA` | `/home/zxh/qlib_data` | 只读数据源根 |
| `OVERLAY_ROOT` | `<PROJECT_ROOT>/data/qlib_root` | 生成的 provider_uri |
| `FEATURES_SRC` / `FEATURES_DST` | `qlib_data/features` / `OVERLAY_ROOT/features` | 源/目的 features |
| `INSTRUMENTS_DST` / `CALENDAR_DST` | overlay 下对应目录 | |
| `BASE_FIELDS` | `(open,high,low,close,volume,factor,vwap)` | qlib_data 已有 7 字段 |
| `DERIVED_FIELDS` | `(change,limit_up,limit_down)` | 需物化的 3 个衍生字段 |
| `UNIVERSE_MARKET` | `highbeta883926` | qrun `market` 名 |
| `BENCHMARK` | `SH000300` | qrun `benchmark`（决策见技术方案 §2 D5） |
| `INDEX_CODE_IFIND` / `INDEX_CODE_QLIB` | `883926.TI` / `SH883926` | 标的代码（备用） |
| `FREQ` | `day` | 日频 |
| `IFIND_TOKEN_FILE` | `/home/zxh/qlib_data/.ifind_token` | token 缓存（复用，secret） |

### 4.2 [binio.py](../qlib_ifind_beta/binio.py) — 底层 bin 读写
直接读写 pyqlib `FileFeatureStorage` 的原始字节布局（qlib 0.9.7 无 `DumpBinAll` helper）：

- 格式：little-endian float32（`<f4`）；首 4 字节 = `start_index`（数据起始的日历行号），其后每 4 字节一个值、对齐到日历。
- `read_bin(path) → (start_index, values)`、`write_bin(path, start_index, values)`（覆盖）、`bin_length(path)`。
- 与 qlib 自带 `write` 逐字节 round-trip。

### 4.3 [overlay.py](../qlib_ifind_beta/overlay.py) — symlink farm
只读 qlib_data 不可写，于是建一个新 `provider_uri` 把只读内容 symlink 进来、衍生 bin 写在自己真实目录里：

| 函数 | 作用 | 落点 |
|---|---|---|
| `link_calendars()` | 整目录 symlink `qlib_data/calendars` | `OVERLAY_ROOT/calendars`（→ symlink） |
| `link_instruments(market_files)` | 建 instruments 真实目录；symlink `all.txt`；写入自有 market 文件 | `OVERLAY_ROOT/instruments` |
| `write_market_file(market, records)` | 写 `instruments/<market>.txt`（TSV `code\tstart\tend`，无 header） | |
| `link_stock(code)` | 建**真实** feature 目录，逐文件 symlink 7 base bins；caller 可继续往里加衍生 bin | `features/<code>/` |
| `ensure_stock_dir(code)` | 建**空真实**目录（给 qlib_data 没有的标的，如 SH883926 从 iFinD dump） | |
| `link_benchmark(code)` | **整目录** symlink（dst→src）；当前未用（见下） | |

> **`link_stock` vs `link_benchmark` 的关键差异**：`link_stock` 逐文件 symlink 到一个**真实目录**，因此可在该目录内追加衍生 bin（benchmark 也走这条路）；`link_benchmark` 是整目录 symlink，写入会落到只读 qlib_data 故不可用。

### 4.4 [materialize.py](../qlib_ifind_beta/materialize.py) — 衍生字段物化
为 qlib `Exchange` 涨跌停拦截提供 3 个 qlib_data 缺失的字段：

- **`board_limit(code) → (limit_up, limit_down)`**：按代码前缀分派板块涨跌停档（取"略低于名义限"规避限价处浮点边界）：
  | 前缀 | limit_up / limit_down | 板块 |
  |---|---|---|
  | `SH000/SH88/SZ399/SH999` | `1.0 / -1.0` | 指数（不交易，永不触发） |
  | `BJ*` | `0.295 / -0.295` | 北交所 ±30% |
  | `SZ300/SZ301/SH688/SH689` | `0.195 / -0.195` | 创业板/科创板 ±20% |
  | 其余（含 B 股 SH900/SZ200） | `0.095 / -0.095` | 沪深主板 ±10% |
- **`compute_change(close, factor)`**：不复权价日涨跌幅 `raw=close/factor; out[i]=raw[i]/raw[i-1]-1`，`out[0]=NaN`。按不复权价判定，避免除权日 `factor` 跳变误判涨跌停。
- **`materialize_instrument(code)`**：从 `FEATURES_SRC` 读 `close/factor`，写 `change/limit_up/limit_down` 到 `FEATURES_DST`；返回 `bool`（源缺失/错位则 False）。

### 4.5 [universe.py](../qlib_ifind_beta/universe.py) — 883926 时变成分股（T-1 lag）
通过 iFinD `data_pool` 报表 `p03473` 取 883926 **每日**成分股快照，构建 T-1 lag 时变 instruments（T 日观察池 = 883926 的 T-1 在册集，无前视）：

- `fetch_constituents(iv_date)` → `p03473`（`iv_zsdm=883926.TI`）→ 当日 100 行 `[date, code_ifind, name, code_qlib]`；历史 `iv_date` 已实测可用。
- `fetch_history_snapshots(start, end)`：逐交易日拉快照，可恢复 CSV 缓存 `data/universe_snapshots.csv`（已缓存天跳过，每 50 天 flush）。
- `snapshots_to_segments()`：长表 → `{code: [(d_in, d_out), ...]}` 连续在册段（一只票可多段：进、退、再进）。
- `shift_T1()`：每段 +1 交易日 → `[next(d_in), next(d_out)]`（用完整 day.txt；末日 → `2099-12-31` 哨兵，对齐 qlib IndexBase 开放区间语义）。
- `ifind_to_qlib()`：`000536.SZ → SZ000536`。
- `dump_universe()`：端到端 → `instruments/highbeta883926.txt`（TSV `code\tstart\tend`，每 code×段 一行；返回历史超集 code 列表供 build_overlay 物化）。
- ✅ **无幸存者偏差**（2026-07-05）：时变池消除静态池 hindsight；端到端验证 `D.features(market, T) == T-1 快照` 6/6 PASS（real_missing=0 extra=0）。
- `build_market()`：静态 fallback（按 all.txt 全区间），时变管线不用，留作 ad-hoc 调试。

### 4.6 [dump_index.py](../qlib_ifind_beta/dump_index.py) — SH883926 行情 dump（当前未启用）
通过 iFinD `history_data` 把 `883926.TI` 的 OHLCVW + `factor=1.0` 写成 day-bin。benchmark 切到 SH000300 后此模块**不再被 build_overlay 引用**，留盘待 883926 数据问题解决后可复活（详见技术方案 §2 D5、§5）。

### 4.7 [ifind.py](../qlib_ifind_beta/ifind.py) — iFinD HTTP client
薄封装两个端点（`history_data` + `data_pool`）+ token 生命周期：

- **token 管理**：`load_refresh_token()` 运行时从 `/home/zxh/qlib_data/scripts/daily_update.py` 解析 `IFIND_REFRESH_TOKEN`（**绝不硬编码**）；`get_access_token()` 读 `/home/zxh/qlib_data/.ifind_token` 缓存，过期则刷新（POST `get_access_token`，写回缓存）。
- **`_post()`**：POST + JSON 解码 + token 过期自动刷新重试 + 瞬时错误（5xx/429）指数退避 + 永久错误（4xx 非 429 / 非 JSON / bad errorcode）fail-fast 抛 `IfindError`。
- **`fetch_history_data(code,start,end,indicators,cps="0")`**：单标的（>10 标的会被截断），返回 long DF `[symbol,date,...]`。
- **`fetch_data_pool(reportname,functionpara,outputpara,field_map)`**：报表查询。

---

## 5. 数据流

### 5.1 离线数据准备流（`scripts/build_overlay.py`，一次性 + 幂等）

```mermaid
flowchart TD
    A["build_overlay.build()"] --> B["1. link_calendars<br/>symlink qlib_data/calendars"]
    A --> C["2. link_instruments<br/>symlink all.txt"]
    C --> D["3. universe.dump_universe<br/>iFinD p03473 → highbeta883926.txt<br/>（返回 100 只成分股 codes）"]
    D --> E{"4. 遍历每只 code"}
    E --> F["link_stock(code)<br/>7 base bins → symlink"]
    E --> G["materialize_instrument(code)<br/>change/limit_up/limit_down → 真实 bin"]
    E --> H["5. benchmark SH000300<br/>link_stock + materialize"]
    B --> I["data/qlib_root/<br/>（provider_uri 就绪）"]
    F --> I
    G --> I
    H --> I
```

- `DUMP_START/DUMP_END = 2024-01-01 / 2026-07-02`（覆盖 train/valid/test 全窗）。
- 幂等：`_symlink` 替换既有节点；可安全重跑。

### 5.2 在线训练-回测流（`qrun/run.py qrun/workflow.yaml`）

```mermaid
flowchart LR
    A["run.py<br/>设 MLFLOW_ALLOW_FILE_STORE<br/>加载 YAML<br/>limit_threshold list→tuple"] --> B["qlib.init<br/>provider_uri=data/qlib_root"]
    B --> C["task_train(config)"]
    C --> D["DatasetH + Alpha158<br/>读 overlay features<br/>（7 base + 3 衍生 → 158 因子）"]
    D --> E["LGBModel<br/>train→valid 早停<br/>→test pred"]
    E --> F["SignalRecord<br/>pred.pkl/label.pkl"]
    F --> G["SigAnaRecord<br/>IC / RankIC / ICIR"]
    E --> H["PortAnaRecord<br/>TopkDropoutStrategy<br/>+ SimulatorExecutor<br/>+ LT_TP_EXP 涨跌停"]
    H --> I["nav / 回撤 / 换手 / 年化"]
    F --> J[("mlruns/")]
    G --> J
    I --> J
```

---

## 6. Overlay 内部结构（provider_uri 实际落点）

核验过的 `data/qlib_root/` 真实布局：

```
data/qlib_root/
├── calendars/            → symlink → /home/zxh/qlib_data/calendars（整目录）
│   └── day.txt           （2000-01-04 → 2026-07-02，6419 天）
├── instruments/          （真实目录）
│   ├── all.txt           → symlink → qlib_data/instruments/all.txt
│   └── highbeta883926.txt  （真实，TSV，例：SZ000536  2020-01-02  2026-07-02）
└── features/             （真实目录，104+ 子目录）
    ├── <每只成分股>/       （真实目录）
    │   ├── open.day.bin    → symlink qlib_data
    │   ├── high.day.bin    → symlink
    │   ├── low.day.bin     → symlink
    │   ├── close.day.bin   → symlink
    │   ├── volume.day.bin  → symlink
    │   ├── factor.day.bin  → symlink
    │   ├── vwap.day.bin    → symlink
    │   ├── change.day.bin    （真实，materialize 写）
    │   ├── limit_up.day.bin  （真实，板块常量）
    │   └── limit_down.day.bin（真实，板块常量）
    └── sh000300/          （benchmark，同上：7 symlink + 3 真实）
```

**为何混合 symlink + 真实**：7 个 base 字段直接复用 qlib_data（零拷贝、永远与源同步）；3 个衍生字段是本项目计算产物，必须写在自有可写目录里。qlib `FileFeatureStorage` 对缺失 bin 鲁棒（返回空 Series），但 `FileInstrumentStorage.check()` 要求 market 文件必须存在 → `highbeta883926.txt` 必须先生成。

---

## 7. 外部依赖

| 依赖 | 角色 | 访问方式 |
|---|---|---|
| `/home/zxh/qlib_data` | 只读行情（7 字段 × 6419 天）、instruments、calendars | 文件系统（symlink 复用） |
| iFinD `quantapi.51ifind.com` | 成分股（p03473）/ 行情（history_data，备用） | HTTPS + access_token；token 复用 `/home/zxh/qlib_data/.ifind_token` |
| conda env `qlib_ifind_beta` | 运行环境（Python 3.12.13 + pyqlib 0.9.7） | `conda run -n qlib_ifind_beta …`（CLAUDE.md 硬约束） |
| `qlib.contrib` | Alpha158 / LGBModel / TopkDropoutStrategy / SimulatorExecutor / Record | import（零子类化） |
| lightgbm 4.6 / pandas 2.3 / numpy 2.4 | 模型与数据计算 | conda env 内 |

---

## 8. 模块依赖图

```mermaid
flowchart TD
    config[config.py]
    binio[binio.py]
    overlay[overlay.py]
    materialize[materialize.py]
    ifind[ifind.py]
    universe[universe.py]
    dump_index[dump_index.py<br/>未启用]
    build_overlay[scripts/build_overlay.py]
    run[qrun/run.py]

    binio --> config
    overlay --> config
    materialize --> binio
    materialize --> config
    ifind --> config
    universe --> ifind
    universe --> overlay
    universe --> config
    dump_index --> binio
    dump_index --> ifind
    dump_index --> config

    build_overlay --> overlay
    build_overlay --> materialize
    build_overlay --> universe
    build_overlay --> config

    run -.读.-> qyaml[workflow.yaml]
    run --> qlibrt[qlib runtime]
```

- 包内单向依赖，无循环：`config` 是叶子；`binio/ifind` 仅依赖 `config`；`overlay/materialize` 依赖 `config/binio`；`universe` 依赖 `ifind/overlay`。
- `dump_index` 与 `universe` 之间是**惰性**耦合（`universe.fetch_constituents` 默认 `iv_date` 时才 `from .dump_index import load_day_calendar`），故移除 dump_index 不影响当前 pipeline。

---

## 9. 运行拓扑（典型命令）

```bash
# 0. 环境（CLAUDE.md 硬约束：conda-only）
conda run -n qlib_ifind_beta python -c "import qlib; print(qlib.__version__)"   # → 0.9.7

# 1. 构建 overlay（一次性，幂等可重跑）
conda run -n qlib_ifind_beta python -m scripts.build_overlay

# 2a. 烟雾测试（2025 子窗口，快速跑通全链路）
conda run -n qlib_ifind_beta python qrun/run.py qrun/workflow_smoke.yaml

# 2b. 全量 MVP（2024-01-01 → 2026-07-02）
conda run -n qlib_ifind_beta python qrun/run.py qrun/workflow.yaml
#    产物 → mlruns/<experiment_id>/<recorder_id>/{pred.pkl, label.pkl, …}
```

> 两个 qlib 本机坑由 `run.py` 兜底（`limit_threshold` list→tuple、`MLFLOW_ALLOW_FILE_STORE=true`），详见技术方案文档 §6。
