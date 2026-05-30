"""Portfolio Manager agent - synthesises risk debate into final PortfolioDecision.

Uses the deep-thinking LLM to evaluate all three risk perspectives and produce
a final position decision with executive summary and investment thesis.

Inspired by TradingAgents' Portfolio Manager which makes the final call
after hearing from all risk debaters.
"""
import json
from ..agents.base import AgentContext, BaseAgent
from ..llm.client import LLMClient
from ..agents.schemas import PortfolioDecision, PortfolioRating


_PORTFOLIO_MANAGER_PROMPT = (
    "You are the Portfolio Manager making the FINAL decision for {stock}.\n\n"
    "Research Recommendation: {recommendation} (confidence: {confidence})\n\n"
    "=== RISK DEBATE ===\n"
    "Aggressive view: max_pos={agg_pos}, stop={agg_stop}, verdict: {agg_verdict}\n"
    "Conservative view: max_pos={con_pos}, stop={con_stop}, verdict: {con_verdict}\n"
    "Neutral view: max_pos={neu_pos}, stop={neu_stop}, verdict: {neu_verdict}\n\n"
    "Market Regime: {regime}\n"
    "Technical Signals: {signals}\n\n"
    "A-share considerations:\n"
    "- 涨跌停板 limits daily P&L to 10% (20% for 科创板/创业板)\n"
    "- T+1 settlement means no intraday exit\n"
    "- Consider market cap and liquidity for position sizing\n"
    "- 北向资金 flows may affect sentiment\n\n"
    "Output as JSON with keys: rating, executive_summary, investment_thesis, "
    "price_target, time_horizon, position_pct\n"
    "Rating must be one of: Buy, Overweight, Hold, Underweight, Sell"
)


class PortfolioManagerAgent(BaseAgent):
    """Portfolio Manager - final decision after risk debate."""

    name = "portfolio_manager"
    description = "Makes final position decision after risk debate"

    def __init__(self, llm_client: LLMClient = None):
        self.llm = llm_client or LLMClient.from_config()

    def execute(self, context: AgentContext) -> AgentContext:
        stock = context.current_debate_stock
        if not stock:
            return context

        plan = context.research_plan
        risk = context.risk_assessments
        signals = self._summarize_signals(context, stock)
        regime = context.regime or "unknown"

        prompt = _PORTFOLIO_MANAGER_PROMPT.format(
            stock=stock,
            recommendation=plan.recommendation.value if plan else "Hold",
            confidence=plan.confidence if plan else 0.5,
            agg_pos=risk.get("aggressive").max_position_pct if risk.get("aggressive") else 0.3,
            agg_stop=risk.get("aggressive").stop_loss_pct if risk.get("aggressive") else 0.08,
            agg_verdict=risk.get("aggressive").verdict if risk.get("aggressive") else "N/A",
            con_pos=risk.get("conservative").max_position_pct if risk.get("conservative") else 0.1,
            con_stop=risk.get("conservative").stop_loss_pct if risk.get("conservative") else 0.05,
            con_verdict=risk.get("conservative").verdict if risk.get("conservative") else "N/A",
            neu_pos=risk.get("neutral").max_position_pct if risk.get("neutral") else 0.2,
            neu_stop=risk.get("neutral").stop_loss_pct if risk.get("neutral") else 0.06,
            neu_verdict=risk.get("neutral").verdict if risk.get("neutral") else "N/A",
            regime=regime, signals=signals,
        )

        response = self.llm.deep_chat([
            {"role": "system", "content": "You are a Portfolio Manager. Output JSON only."},
            {"role": "user", "content": prompt},
        ])

        decision = self._parse_response(response)
        if decision:
            context.portfolio_decision = decision
            # Update rl_signals with the final decision
            self._apply_decision(context, stock, decision)
            context.warnings.append(
                f"[PortfolioManager] {stock}: {decision.rating.value} "
                f"(pos={decision.position_pct:.0%})"
            )
        else:
            # Fallback to research plan
            self._apply_research_fallback(context, stock)
        return context

    def _parse_response(self, response: str) -> PortfolioDecision | None:
        try:
            if "{" in response:
                json_str = response[response.index("{"):response.rindex("}")+1]
                data = json.loads(json_str)
                return PortfolioDecision(
                    rating=data.get("rating", "Hold"),
                    executive_summary=data.get("executive_summary", ""),
                    investment_thesis=data.get("investment_thesis", ""),
                    price_target=data.get("price_target"),
                    time_horizon=data.get("time_horizon"),
                    position_pct=float(data.get("position_pct", 0.0)),
                )
        except Exception:
            pass
        return None

    def _apply_decision(self, ctx, stock, decision):
        """Convert PortfolioDecision to rl_signals format."""
        if decision.rating in (PortfolioRating.BUY, PortfolioRating.OVERWEIGHT):
            action = "buy"
        elif decision.rating in (PortfolioRating.SELL, PortfolioRating.UNDERWEIGHT):
            action = "sell"
        else:
            action = "hold"

        if action != "hold":
            ind = ctx.indicators.get(stock, {})
            ctx.rl_signals.append({
                "stock": stock,
                "action": action,
                "strategy": "llm_debate",
                "confidence": decision.position_pct,
                "price": ind.get("close", 0),
                "position_pct": decision.position_pct,
                "reason": decision.executive_summary[:200],
            })

    def _apply_research_fallback(self, ctx, stock):
        """Fallback using research plan directly."""
        plan = ctx.research_plan
        if plan and plan.recommendation in (PortfolioRating.BUY, PortfolioRating.OVERWEIGHT):
            action = "buy"
            conf = plan.confidence
        elif plan and plan.recommendation in (PortfolioRating.SELL, PortfolioRating.UNDERWEIGHT):
            action = "sell"
            conf = plan.confidence
        else:
            return

        ind = ctx.indicators.get(stock, {})
        ctx.rl_signals.append({
            "stock": stock,
            "action": action,
            "strategy": "llm_debate",
            "confidence": conf,
            "price": ind.get("close", 0),
            "position_pct": conf * 0.2,
            "reason": plan.rationale[:200] if plan else "",
        })

    def _summarize_signals(self, ctx, stock):
        hits = [s for s in ctx.ts_signals if s.get("stock") == stock]
        return "; ".join(f"{s['type']}(conf={s['confidence']})" for s in hits[:5]) or "None"
