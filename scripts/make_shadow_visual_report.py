"""影子模拟盘证据可视化报告（给管理层汇报的单文件 HTML）。

数据来源：data/shadow_analysis/（由 scripts/verify_shadow_basket.py 等四个
分析脚本产出）+ data/paper_shadow/nav.csv。输出 reports/ 下的离线自包含
HTML（plotly.js 内嵌，可直接发给他人浏览器打开）。

生成：conda run -n qlib_ifind_beta python scripts/make_shadow_visual_report.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "shadow_analysis"
NAV = ROOT / "data" / "paper_shadow" / "nav.csv"
OUT = ROOT / "reports" / "shadow_evidence_report.html"

# 汇总结论数字（出处见 docs/research/2026-09-20-shadow-paper-evidence.md，
# 全部可由 data/shadow_analysis 明细复算）
N_DAYS = 44
TOTAL_NET = 0.2078        # 纸面全程净值收益（含成本）
MAX_DD = -0.1111
ACTIVE_TOTAL = 0.1671     # 38 日段内主动收益（相对成分篮子，成本前）
ACTIVE_IR = 3.91
N_SETTLED = 38
FORCE_EXIT = (295, 296)   # 强制退出卖出数 / 总卖出数
SEG_DAY = 0.299           # 买入当天 09:41→收盘
SEG_OVERNIGHT = 0.0263    # 收盘→次日收盘
SEG_NIGHT_EARLY = -0.079  # 收盘→次日 09:41（隔夜早段）
SEG_NEXT_LATE = 0.1134    # 次日 09:41→收盘（次日晚段）
GAP_TOTAL = -0.2521       # 理想口径与实际账本的缺口
GAP_OVERNIGHT = -0.079
GAP_FEE = -0.0642
GAP_STRUCT = -0.1087      # 仓位结构与复利差（残差口径）


def fig_nav_curves(d: pd.DataFrame) -> go.Figure:
    f = go.Figure()
    f.add_scatter(x=d["settle_date"], y=(1 + d["strat_label_ret"]).cumprod(),
                  name="策略（理想持有口径，成本前）", line=dict(width=3))
    f.add_scatter(x=d["settle_date"], y=(1 + d["basket_label_ret"]).cumprod(),
                  name="成分篮子（当日100只等权）", line=dict(width=2, dash="dot"))
    f.add_scatter(x=d["settle_date"], y=(1 + d["nav_ret"]).cumprod(),
                  name="账本净值（真实结算，含成本）", line=dict(width=2))
    f.update_layout(title="三条净值曲线：选股是真本事，但要付过夜和换仓的“税”",
                    yaxis_title="净值（起点=1）", hovermode="x unified",
                    legend=dict(orientation="h", y=-0.2))
    return f


def fig_active_daily(d: pd.DataFrame) -> go.Figure:
    colors = ["#c0392b" if v < 0 else "#27ae60" for v in d["active_ret"]]
    f = go.Figure()
    f.add_bar(x=d["settle_date"], y=d["active_ret"], marker_color=colors,
              name="当日主动收益")
    f.add_scatter(x=d["settle_date"], y=(1 + d["active_ret"]).cumprod() - 1,
                  yaxis="y2", name="累计主动收益", line=dict(width=3, color="#2c3e50"))
    f.update_layout(title="每日主动收益（相对当日成分篮子）——66% 的交易日为正",
                    yaxis=dict(title="当日主动收益"),
                    yaxis2=dict(title="累计", overlaying="y", side="right",
                                tickformat=".0%"),
                    legend=dict(orientation="h", y=-0.2))
    return f


def fig_segments() -> go.Figure:
    names = ["买入当天<br>09:41→收盘", "隔夜到次日<br>收盘→收盘",
             "其中：隔夜早段<br>收盘→次日09:41", "其中：次日晚段<br>次日09:41→收盘"]
    vals = [SEG_DAY, SEG_OVERNIGHT, SEG_NIGHT_EARLY, SEG_NEXT_LATE]
    colors = ["#27ae60", "#95a5a6", "#c0392b", "#f39c12"]
    f = go.Figure(go.Bar(x=names, y=vals, marker_color=colors,
                         text=[f"{v:+.1%}" for v in vals], textposition="outside"))
    f.update_layout(title="利润几乎全部产生在买入当天：过夜基本不赚钱",
                    yaxis=dict(title="38 日累计收益", tickformat=".0%"))
    return f


def fig_waterfall() -> go.Figure:
    f = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["理想持有口径", "隔夜损耗<br>(T+1 制度强制过夜)", "交易成本",
           "仓位结构与复利差", "实际账本"],
        y=[0.3332, GAP_OVERNIGHT, GAP_FEE, GAP_STRUCT, None],
        text=[f"+33.3%", "-7.9", "-6.4", "-10.9", "+8.1%"],
        textposition="outside",
        decreasing=dict(marker=dict(color="#c0392b")),
        increasing=dict(marker=dict(color="#27ae60")),
        totals=dict(marker=dict(color="#2c3e50")),
    ))
    f.update_layout(title="从理想回测到实际账本：25 个百分点的去向（38 日）",
                    yaxis=dict(title="累计收益", tickformat=".0%"))
    return f


def fig_state_alpha(d: pd.DataFrame) -> go.Figure:
    import numpy as np

    q = pd.qcut(d["basket_label_ret"], 4, labels=["最差1/4（大跌日）", "次差1/4", "次好1/4", "最好1/4（大涨日）"])
    g = d.groupby(q, observed=True)["active_ret"].agg(["mean", "count"])
    f = go.Figure(go.Bar(x=list(g.index), y=g["mean"],
                         marker_color="#2980b9",
                         text=[f"日均 {m:+.2%}<br>(n={int(n)})" for m, n in
                               zip(g["mean"], g["count"])], textposition="outside"))
    f.update_layout(title="大跌日选股能力不消失——回撤来自个别极端日，不是常态失效",
                    yaxis=dict(title="日均主动收益", tickformat=".1%"))
    return f


def fig_turnover() -> go.Figure:
    f = go.Figure(go.Pie(
        labels=["持仓掉出成分池<br>被动卖出", "池内分数淘汰<br>主动卖出"],
        values=[FORCE_EXIT[0], FORCE_EXIT[1] - FORCE_EXIT[0]],
        marker=dict(colors=["#e67e22", "#27ae60"]),
        textinfo="value+percent",
    ))
    f.update_layout(title="换仓是被指数规则强制的：296 笔卖出中 295 笔为被动退出")
    return f


def fig_extreme_day(an: pd.DataFrame) -> go.Figure:
    s = an.sort_values("ret")
    f = go.Figure(go.Bar(x=s["code"], y=s["ret"],
                         marker_color=["#c0392b" if v < 0 else "#27ae60" for v in s["ret"]],
                         text=[f"{v:+.1%}" for v in s["ret"]], textposition="outside"))
    f.update_layout(title="8/18 极端日解剖：十只持仓全部下跌（系统性高贝塔回撤，非选股失误）",
                    yaxis=dict(title="当日收益", tickformat=".0%"))
    return f


CSS = """
body{font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;
     max-width:1080px;margin:24px auto;padding:0 16px;color:#2c3e50;background:#fafafa}
h1{font-size:26px} h2{font-size:20px;border-left:5px solid #2980b9;padding-left:10px;margin-top:40px}
.card{display:inline-block;background:#fff;border:1px solid #e0e0e0;border-radius:10px;
      padding:14px 20px;margin:6px;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.card .big{font-size:26px;font-weight:700}
.card .lbl{font-size:13px;color:#7f8c8d}
.note{font-size:13px;color:#7f8c8d;line-height:1.7;background:#fff;
      border-left:4px solid #bdc3c7;padding:10px 14px;border-radius:4px}
"""


def main() -> None:
    d = pd.read_csv(SRC / "basket_decomposition.csv")
    an = pd.read_csv(SRC / "extreme_day_0818_anatomy.csv")

    figs = [
        ("一、三条净值曲线", fig_nav_curves(d),
         "理想持有口径（09:41 买入持有到次日收盘、不含成本）明显高于成分篮子——模型选股能力真实；"
         "账本净值（真实结算）低于两者，差在过夜损耗、交易成本和仓位结构。"),
        ("二、每日主动收益", fig_active_daily(d),
         "主动收益 = 策略持仓收益 − 当日成分篮子收益（同一持有口径、成本前）。"
         f"38 个结算日累计 +16.7%，信息比率 {ACTIVE_IR}，胜率 66%。"),
        ("三、利润产生在哪一段", fig_segments(),
         "把持有区间拆开实测：买入当天（09:41→收盘）累计约 +30%，是策略利润的几乎全部；"
         "收盘持有到次日收盘仅 +2.6%——过夜基本不赚钱，隔夜早段还倒贴约 8%。"),
        ("四、25 个百分点缺口的去向", fig_waterfall(),
         "理想回测 +33.3% 到实际账本 +8.1% 的缺口：隔夜损耗是 A 股 T+1 交易制度强制的持仓过夜代价；"
         "交易成本 6.4 个百分点为逐笔回单实测；仓位结构与复利差指账本只有约七成五资金吃满"
         "当天利润段（两成在效率较低的昨日保留仓上、半成现金闲置）。"),
        ("五、大跌日选股能力", fig_state_alpha(d),
         "按篮子日收益四等分：最差四分之一交易日的日均主动收益（+0.55%）反而高于最好四分之一（+0.27%）。"
         "结论：不需要全程性仓位调节（历史上已多次证伪），需要的是针对个别极端日的条件保护。"),
        ("六、换仓构成", fig_turnover(),
         "883926 指数每天更换约 84% 成分，持仓一旦掉出当日成分池就被强制卖出。"
         "换手率和成本（约 16bp/日）是策略的结构性代价，行业通用的换手优化方法在此不适用。"),
        ("七、极端日解剖", fig_extreme_day(an),
         "8/18 当日篮子 -6.6%、策略持仓 -10.2%：十只全跌、无一幸免，属于高贝塔池整体回撤，"
         "而非模型选错股票。这类日子的保护是下一阶段尾部研究的唯一目标形态。"),
    ]

    parts = [
        "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>",
        "<title>影子模拟盘证据报告</title>",
        f"<style>{CSS}</style></head><body>",
        "<h1>高贝塔指数增强 · 影子模拟盘证据报告</h1>",
        "<p class='note'>数据区间 2026-07-21 ～ 2026-09-18（纸面模拟盘，全自动影子结算，不含真实下单）。"
        "全部数字由落盘的逐日持仓、逐笔成交回单与分钟价格数据复算，明细脚本见 scripts/。</p>",
        "<div>",
        "<div class='card'><div class='big'>+20.8%</div><div class='lbl'>44 日净值收益（含成本）</div></div>",
        "<div class='card'><div class='big'>-11.1%</div><div class='lbl'>最大回撤</div></div>",
        "<div class='card'><div class='big'>+16.7%</div><div class='lbl'>38 日主动收益（相对成分篮子）</div></div>",
        "<div class='card'><div class='big'>3.91</div><div class='lbl'>主动收益信息比率</div></div>",
        "<div class='card'><div class='big'>≈+30%</div><div class='lbl'>利润产生在买入当天</div></div>",
        "<div class='card'><div class='big'>99.7%</div><div class='lbl'>换仓为指数规则强制</div></div>",
        "</div>",
    ]
    first = True
    for title, fig, note in figs:
        parts.append(f"<h2>{title}</h2>")
        parts.append(f"<p class='note'>{note}</p>")
        parts.append(fig.to_html(full_html=False,
                                 include_plotlyjs=True if first else False,
                                 config={"displaylogo": False}))
        first = False
    parts.append("<p class='note'>结论与后续路线详见 "
                 "docs/research/2026-09-20-shadow-paper-evidence.md。"
                 "本报告为研究证据展示，不构成投资建议；模型与生产参数未做任何变更。</p>")
    parts.append("</body></html>")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"written: {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
