---
layout: default
title: 生产基线
---

# 生产基线与日内流程

## 日内时序

1. 盘前读取 T 日 883926 成分快照，形成观察池。
2. 收集每只股票 09:31–09:40 的 10 根闭合分钟 K 线。
3. 物化 18 个特征，并检查分钟数据完整性和 NaN 质量门槛。
4. 读取 09:41 价格，加载冻结 Champion 或经过门控的日期匹配模型。
5. 生成模型分数，按 Top10 / `n_drop=8` 形成调仓清单。
6. 产出 CSV；方案 B 是上午先卖旧仓再买新仓，订单由人工提交。
7. 记录成交、现金和持仓状态，盘后可执行候选滚动训练和影子核对。

## 关键安全约束

- 不使用未来数据：T 日日频字段若进入研究模型必须 lag 到 T-1；现役特征只用 09:41
  前已闭合的分钟 K 线。
- `$price_941` 缺失时不能回退到 T 日收盘价；质量护栏应将该股票日排除出交易候选。
- 不自动提交券商订单，不自动晋升候选模型，不在文档或代码中保存 token。
- 退出 883926 成分池的旧持仓仍需要行情覆盖，否则可能无法形成卖出记录；这是当前
  影子运行待完善项。

## 入口命令

```bash
conda run -n qlib_ifind_beta python scripts/intraday_production.py --help
conda run -n qlib_ifind_beta python scripts/replay_intraday_shadow.py
conda run -n qlib_ifind_beta python scripts/retrain.py --test-start YYYY-MM-DD
```

## 非承诺性说明

历史影子回放只验证数据传递、分数复现、状态对账和模拟交易链路。它不等同于真实券商
联调或实盘收益承诺；真实延迟、人工操作、拒单、滑点和行情缺失仍需单独验证。
