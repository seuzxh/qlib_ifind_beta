"""qlib_ifind_beta — A-share factor mining on top of readonly qlib_data.

Data-engineering package: builds a minimal symlink-farm overlay over the readonly
`/home/zxh/qlib_data` bin tree, materializes derived feature bins ($change,
$limit_up, $limit_down) required by qlib's Exchange for A-share limit detection,
and dumps the 883926 index benchmark + constituents from iFinD.

The qrun workflow itself uses ONLY qlib.contrib classes (Alpha158 / LGBModel /
TopkDropoutStrategy / SimulatorExecutor / Recorder) — no custom subclasses.
"""

__version__ = "0.1.0"
