---
layout: default
title: 架构与流程图
nav_order: 2.5
---

# 架构与流程图

三张交互图覆盖系统的三个视角：**组件架构**（谁依赖谁）、**每日循环**（影子模拟盘
一个交易日怎么跑）、**盘中时序**（T 日从盘前到收盘谁在什么时刻做什么）。图内支持
缩放、节点聚焦和分章浏览；每张图旁附要点速览，细节以对应文档页为准。

图的源文件是同目录下的 archify JSON（`assets/diagrams/*.json`），可复现渲染。

## 系统架构

<div class="diagram-frame">
<iframe src="assets/diagrams/system-architecture.html" title="系统架构图" loading="lazy"></iframe>
</div>

[全屏打开](assets/diagrams/system-architecture.html){:target="_blank" .btn .btn-purple }

**要点**：外部行情（iFinD）与只读历史源落成本地运行资产（成分快照、qlib_root
overlay）；因子引擎产出 18 维特征，冻结 Champion 打分后经 Top10 / n_drop=8 决策，
同时驱动影子模拟盘（自动撮合、逐日落账）与盘中生产（订单 CSV 人工下单）。
治理红线：不自动下单、不自动晋升候选模型。

## 影子模拟盘每日流程

<div class="diagram-frame">
<iframe src="assets/diagrams/paper-shadow-day.html" title="影子模拟盘每日流程图" loading="lazy"></iframe>
</div>

[全屏打开](assets/diagrams/paper-shadow-day.html){:target="_blank" .btn .btn-purple }

**要点**：16:30 cron 触发后先过运行门禁（节假日、残缺账本 fail-closed 跳过，等待
`paper_shadow.py day --date` 手动补跑）；交易日完成分钟数据同步、当日成分快照、
18 因子组装与质量门槛，评分恒用冻结 recorder；组合决策经 09:41 参考价撮合
（涨跌停拦截），收盘估值对账 PASS 后追加 `nav.csv`。流程细节见
[模拟盘流程](paper-trading.md)。

## 盘中信号时序（T 日）

<div class="diagram-frame">
<iframe src="assets/diagrams/intraday-signal-sequence.html" title="盘中信号时序图" loading="lazy"></iframe>
</div>

[全屏打开](assets/diagrams/intraday-signal-sequence.html){:target="_blank" .btn .btn-purple }

**要点**：盘前两步做日历与成分快照；09:31–09:40 每分钟采集**刚闭合**的分钟K线
（禁止"最近十根"覆盖目标窗口）；09:40 后组装特征送冻结 Champion 打分并生成
卖出计划；09:41 K 线闭合后独立采价、涨停拦截、产出买单 CSV，交人工先卖后买；
盘后对账，16:30 影子线独立结算。时间合同与红线见
[生产基线](production-baseline.md)。

<style>
.diagram-frame { border: 1px solid #d0d7de; border-radius: 8px; overflow: hidden; margin-bottom: 8px; }
.diagram-frame iframe { width: 100%; height: 720px; border: 0; display: block; }
</style>
