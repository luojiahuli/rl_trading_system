#!/usr/bin/env python3
"""可视化图表生成模块"""
import os
from pyecharts import options as opts
from pyecharts.charts import Bar, Line, Pie, Grid, Page
from pyecharts.globals import ThemeType


def create_report_html(context, output_path: str) -> str:
    """生成完整报告 HTML"""
    page = Page(layout=Page.SimplePageLayout)

    # 1. 热门板块排行
    if context.hot_sectors:
        sector_bar = _sector_chart(context.hot_sectors)
        page.add(sector_bar)

    # 2. 交易信号
    if context.rl_signals:
        signal_chart = _signal_chart(context.rl_signals)
        page.add(signal_chart)

    # 3. 策略表现
    if context.strategy_results:
        perf = context.strategy_results.get("strategy_performance", {})
        if perf:
            strategy_chart = _strategy_chart(perf)
            page.add(strategy_chart)

    # 4. 风控指标
    if context.risk_metrics:
        equity = context.risk_metrics.get("current_equity", 0)
        peak = context.risk_metrics.get("peak_equity", 0)
        dd = context.risk_metrics.get("drawdown", {})
        risk_chart = _risk_chart(equity, peak, dd.get("dd_pct", 0))
        page.add(risk_chart)

    # 5. 回测收益曲线
    if context.backtest_results:
        eq_chart = _equity_chart(context.backtest_results)
        page.add(eq_chart)

    page.render(output_path)
    return output_path


def _sector_chart(hot_sectors: list) -> Bar:
    """热门板块柱状图"""
    sectors = [s["sector"][:8] for s in hot_sectors[:10]]
    scores = [s["heat_score"] for s in hot_sectors[:10]]
    return (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="800px", height="400px"))
        .add_xaxis(sectors)
        .add_yaxis("热度", scores, color="#FF6B35")
        .set_global_opts(
            title_opts=opts.TitleOpts(title="🔥 热门板块排行"),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=30)),
        )
    )


def _signal_chart(signals: list) -> Bar:
    """交易信号图"""
    stocks = [s["stock"][:6] for s in signals[:10]]
    confidences = [s["confidence"] for s in signals[:10]]
    colors = ["#FF4444" if s["action"] == "sell" else "#44BB44" for s in signals[:10]]
    bar = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="800px", height="400px"))
        .add_xaxis(stocks)
    )
    for i, (stock, conf) in enumerate(zip(stocks, confidences)):
        bar.add_yaxis(
            signals[i]["action"], [conf],
            color=colors[i],
            label_opts=opts.LabelOpts(position="right"),
        )
    return bar.set_global_opts(
        title_opts=opts.TitleOpts(title="📈 交易信号"),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=30)),
    )


def _strategy_chart(perf: dict) -> Bar:
    """策略表现对比"""
    names = list(perf.keys())
    returns = [perf[n].get("total_return", 0) * 100 for n in names]
    sharpes = [perf[n].get("sharpe_ratio", 0) for n in names]

    bar = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="800px", height="400px"))
        .add_xaxis(names)
        .add_yaxis("收益率%", [round(r, 2) for r in returns], color="#5470C6")
        .add_yaxis("Sharpe", [round(s, 2) for s in sharpes], color="#91CC75")
        .set_global_opts(
            title_opts=opts.TitleOpts(title="📊 策略绩效对比"),
            yaxis_opts=opts.AxisOpts(name="百分比/比率"),
        )
    )
    return bar


def _risk_chart(equity: float, peak: float, dd: float) -> Pie:
    """风控概览饼图"""
    return (
        Pie(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="400px", height="400px"))
        .add("", [
            ("当前净值", max(0, equity)),
            ("回撤金额", max(0, peak - equity)),
        ])
        .set_global_opts(title_opts=opts.TitleOpts(title=f"⚠️ 风控概览 回撤:{dd:.2%}"))
    )


def _equity_chart(results: list) -> Line:
    """回测收益曲线"""
    line = Line(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="800px", height="400px"))
    for result in results[:3]:
        curve = result.get("equity_curve", [])
        if curve:
            line.add_xaxis(list(range(len(curve))))
            line.add_yaxis(
                result.get("strategy", "strategy"),
                [round(v, 2) for v in curve],
                is_smooth=True,
                linestyle_opts=opts.LineStyleOpts(width=1.5),
            )
    return line.set_global_opts(
        title_opts=opts.TitleOpts(title="📉 策略收益曲线"),
        xaxis_opts=opts.AxisOpts(name="交易日"),
        yaxis_opts=opts.AxisOpts(name="净值"),
    )
