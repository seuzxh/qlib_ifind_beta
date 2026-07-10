"""P1 实战对接 — T 日池增量物化 day.bins。

薄封装 overlay + materialize + materialize_minute。零新物化逻辑。
materialize_minute_instrument 幂等全量重算（读 cn_data_1min 所有可得日 → 重写 day.bin），
故 cn_data_1min T+0 同步 T 日后，调一次即自动物化 T 日（含 champion 18 因子 + price_941 +
change_941 + 4 extra；materialize_instrument 补 change/limit_up/limit_down）。
"""
from __future__ import annotations

from qlib_ifind_beta import materialize, materialize_minute, overlay
from qlib_ifind_beta.config import INSTRUMENTS_DST, UNIVERSE_MARKET


def load_pool(target_date: str, market: str = UNIVERSE_MARKET) -> list[str]:
    """读 instruments/<market>.txt → target_date 当日在册 codes（T 日池：查 T 返回 T 日集）。

    文件 TSV：code\\tstart_date\\tend_date（时变段）。过滤 start <= target_date <= end，
    去重返回（~100/日；区别于全历史 5116 unique × 50525 段）。883926 股池 T 日盘前更新，
    故 T 日查询返回的是 T 日当天的成分股（2026-07-10 起生效，此前为 T-1 lag）。
    """
    p = INSTRUMENTS_DST / f"{market}.txt"
    codes: set[str] = set()
    for line in p.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 3 or not parts[0]:
            continue
        code, start, end = parts[0], parts[1], parts[2]
        if start <= target_date <= end:
            codes.add(code)
    return sorted(codes)


def materialize_pool(codes: list[str]) -> dict:
    """对给定 codes：link_stock + materialize change/limit + materialize minute factors。

    materialize_minute_instrument / materialize_instrument 均幂等（全量重算），
    对已物化票重复调用只刷新 T 日增量行，无副作用。
    """
    ok_min = miss_min = ok_chg = miss_chg = 0
    for code in codes:
        overlay.link_stock(code)                      # 7 base bins symlink（新入池票必需）
        if materialize.materialize_instrument(code):
            ok_chg += 1
        else:
            miss_chg += 1
        if materialize_minute.materialize_minute_instrument(code):
            ok_min += 1
        else:
            miss_min += 1
    return {"n": len(codes),
            "minute_ok": ok_min, "minute_miss": miss_min,
            "change_ok": ok_chg, "change_miss": miss_chg}
