"""盘中生产与历史影子回放。

- intraday: 单日目录、分钟K线门禁、调仓计划、CSV/JSON/Parquet 产物。
- historical_replay: 用历史行情适配同一套盘中状态转换。

生产入口为 ``scripts/intraday_production.py``，只生成文件，不提交券商订单。
"""
