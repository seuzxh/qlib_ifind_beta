"""P1 实战对接 — 纸面前向跟踪（live forward）。

子模块：
- inference: T 日收盘后用冻结 champion 模型推理 → top10 信号。
- track: 信号记录 / T+1 结算 / NAV 累积 / 日度 IC（纯 pandas 状态机）。
- materialize_live: T 日池增量物化 day.bins（薄封装 overlay+materialize）。

入口：scripts/live_forward.py（工作日 15:35 触发）。
"""
