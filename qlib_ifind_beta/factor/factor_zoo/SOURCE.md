# Vendored 来源与改动记录

- 上游：<https://github.com/JustinF8/qlib-factor-zoo.git>（MIT，见 `LICENSE.zoo`）
- 快照 commit：`ea21f315f2e6e838513cdec20b0c4298d1cf36ca`（2026-07-17），克隆于 /tmp/factor-zoo
- 拷贝文件（逐字节，未改内容）：
  - `custom_ops.py` ← `qlib/contrib/data/custom_ops.py`（50+ 自定义算子）
  - `loader_alpha101.py` ← `qlib/contrib/data/loader_alpha101.py`（101 表达式）
  - `loader_gtja191.py` ← `qlib/contrib/data/loader_gtja191.py`（191 表达式）
  - `LICENSE.zoo` ← `LICENSE`
- 派生文件（工具生成，非逐字节）：
  - `expressions_tdxgs_jq110.json`：用 ast 从 `qlib/contrib/data/handler.py`
    提取 `TDXGS.get_feature_config`（88 条）与 `JQ110DataHandler.
    get_feature_config`（109 条）的表达式清单（上游方法体为纯字符串构建，
    提取脚本见 `scripts/extract_zoo_exprs.py` 的说明）。
- 本项目新增（非上游内容）：`__init__.py`、`registry.py`、`zoo_libs.py`、
  `parse.py`、`excluded.json` 及后续研究代码。
- 兼容性：上游 import `qlib.data.ops` / `qlib.data.dataset.loader`，与
  pyqlib 0.9.7 原生同名模块兼容（冒烟验证见
  `docs/backtest-log/2026-09-24-factor-zoo-screen.md` Phase 0.2）。
