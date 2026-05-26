#!/usr/bin/env python3
"""报告生成 Agent"""
from ..agents.base import AgentContext, BaseAgent


class ReportGeneratorAgent(BaseAgent):
    name = "report_generator"
    description = "汇总各 Agent 结果生成综合报告"

    def execute(self, context: AgentContext) -> AgentContext:
        parts = [f"# 🚀 智能量化交易日报\n**{context.date}**\n"]

        # 热门板块
        if context.hot_sectors:
            parts.append("## 🔥 热门板块\n")
            parts.append("| 板块 | 热度 | 相关股票 |\n|------|------|---------|\n")
            for s in context.hot_sectors[:8]:
                stocks = ", ".join(s.get("stocks", [])[:3]) or "—"
                parts.append(f"| {s['sector']} | {s['heat_score']} | {stocks} |\n")

        # 市场状态
        if context.regime:
            parts.append(f"\n**市场状态**: {context.regime}\n")

        # 交易信号
        if context.rl_signals:
            parts.append("\n## 📈 交易信号\n")
            parts.append("| 股票 | 操作 | 置信度 | 理由 |\n|------|------|--------|------|\n")
            for s in context.rl_signals[:8]:
                parts.append(f"| {s['stock']} | {s['action']} | "
                             f"{s['confidence']:.0%} | {s.get('reason', '')} |\n")

        # 策略表现
        if context.strategy_results:
            parts.append("\n## 📊 策略绩效\n")
            perf = context.strategy_results.get("strategy_performance", {})
            if perf:
                parts.append("| 策略 | 收益率 | Sharpe | 最大回撤 |\n")
                parts.append("|------|--------|--------|---------|\n")
                for name, metrics in perf.items():
                    parts.append(f"| {name} | {metrics.get('total_return', 0):.2%} | "
                                 f"{metrics.get('sharpe_ratio', 0):.3f} | "
                                 f"{metrics.get('max_drawdown', 0):.2%} |\n")

            best_sharpe = context.strategy_results.get("best_sharpe_strategy", "")
            best_return = context.strategy_results.get("best_return_strategy", "")
            if best_sharpe:
                parts.append(f"\n🏆 **最佳 Sharpe 策略**: {best_sharpe}\n")
            if best_return:
                parts.append(f"🏆 **最佳收益策略**: {best_return}\n")

        # 市场研判
        if context.market_judgement:
            mj = context.market_judgement
            parts.append("\n## 📊 市场研判\n")
            parts.append("| 维度 | 判断 |\n|------|------|\n")
            parts.append(f"| 市场阶段 | {mj.get('market_phase', '—')} |\n")
            parts.append(f"| 趋势方向 | {mj.get('trend_direction', '—')} |\n")
            parts.append(f"| 政策预期 | {mj.get('policy_outlook', '—')} |\n")
            parts.append(f"| 置信度 | {mj.get('confidence', '—')} |\n")

            d = mj.get("details", {})
            sb = d.get("stock_breadth", {})
            sec = d.get("sector_breadth", {})
            idx = d.get("index_trend", {})

            parts.append(
                f"\n**市场宽度**: {sb.get('pct_above_ma50', 0)}% 股票站上MA50 "
                f"| **板块活跃度**: {sec.get('breadth', 'unknown')} "
                f"({sec.get('total_sectors', 0)}个热门板块)\n"
            )
            parts.append(
                f"**指数**: 价格vsMA200={idx.get('price_vs_ma200_pct', 0)}% "
                f"| 均线排列={idx.get('ma_alignment', 'mixed')} "
                f"| RSI={idx.get('rsi_14', 50)}\n"
            )

            prediction = mj.get("next_trend", "")
            if prediction:
                parts.append(f"\n**走势预判**: {prediction}\n")

            summary = mj.get("summary", "")
            if summary:
                parts.append(f"\n{summary}\n")

        # 风控
        if context.risk_metrics:
            dd = context.risk_metrics.get("drawdown", {})
            parts.append(f"\n## ⚠️ 风险控制\n")
            parts.append(f"- 当前回撤: {dd.get('dd_pct', 0):.2%}\n")
            parts.append(f"- 状态: {dd.get('level', 'normal')}\n")
            parts.append(f"- 建议: {dd.get('message', '正常交易')}\n")

        # 可视化
        if context.viz_path:
            parts.append(f"\n📊 **可视化报告**: {context.viz_path}\n")

        context.report_text = "".join(parts)
        return context
