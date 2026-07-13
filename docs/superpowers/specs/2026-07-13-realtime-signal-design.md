# 实时模拟盘信号生成系统设计（§56）

> 2026-07-13，`feat/realtime-signal` 分支

## 目标

每个交易日 9:41，用实时 9:31-9:40 分钟K线数据生成 top10 标的池信号。

## 架构

### 数据流

```
9:40:00  kline-fetcher 并行拉取 ~100 股 × 11 bars（09:31-09:41）
  ↓
9:40:10  内存构建 c/o/h/l/vol/vwap 数组（length 11）
  ↓
9:40:15  compute_day_factors() → 14 因子 + 4 extra（共 18 champion 因子）
  ↓
9:40:20  写入 overlay day.bins（T 日行）
  ↓
9:40:25  predict_day(T) → FROZEN champion HFLGBModel 预测
  ↓
9:40:30  输出 top10 + 剔除封涨停
  ↓
9:41:00  用户决策 ← CSV + 控制台摘要
```

### 关键设计决策

1. **复用 predict_day 而非绕过 qlib**：将 T 日因子写入 overlay bin 后调用现有 predict_day，
   确保预处理（RobustZScoreNorm/Fillna）与训练完全一致。
2. **实时因子验证**：realtime 因子 vs materialized 因子全部匹配（max diff < 3e-6）。
3. **prev-day 数据来自 cn_data_1min**：T-1 全天分钟量已同步，直接读取。

## 文件清单

| 文件 | 职责 |
|---|---|
| `qlib_ifind_beta/realtime/data_fetch.py` | kline-fetcher 封装 + prev-day 缓存 + universe |
| `qlib_ifind_beta/realtime/signal.py` | 核心信号生成（fetch→factors→write→predict） |
| `scripts/realtime_signal.py` | CLI 入口（--date / --dry-run / --topk） |
| `tests/test_realtime_signal.py` | 11 测试（因子计算/边界/数据读取） |

## 使用方法

### 交易日 9:41 实时运行

```bash
conda run -n qlib_ifind_beta --no-capture-output python -W ignore \
    scripts/realtime_signal.py --date 2026-07-13
```

### Dry-run（非交易日测试）

```bash
conda run -n qlib_ifind_beta --no-capture-output python -W ignore \
    scripts/realtime_signal.py --date 2026-07-10 --dry-run
```

## 验证结果

- **因子精度**：realtime vs materialized 18 因子全部匹配（max diff 2.81e-06）✓
- **Dry-run**：2026-07-10 top10 信号生成成功，候选 97 / 入选 10 ✓
- **测试**：11/11 新测试通过，73/73 全量通过 ✓
- **kline-fetcher**：fetch_min_kline(code, count=-11) 验证可用 ✓

## 依赖检查

| 依赖 | 状态 | 备注 |
|---|---|---|
| kline-fetcher | ✅ 可用 | `KLINE_API_BASE_URL=http://183.242.5.14:7778` |
| FROZEN 模型 | ✅ `93d435e0` | HFLGBModel champion |
| cn_data_1min | ✅ 截至 2026-07-10 | T-1 分钟量已就绪 |
| qlib_data | ✅ 截至 2026-07-10 | 日频 close/factor/open 已就绪 |
| Universe | ✅ highbeta883926.txt | 已更新到 2026-07-10 |
| iFinD token | ✅ 有效至 2026-07-16 | 仅 universe 刷新需要 |
