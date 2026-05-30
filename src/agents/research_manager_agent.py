"""Research Manager agent - synthesises Bull/Bear debate into a structured ResearchPlan.

Uses the deep-thinking LLM (more powerful) to evaluate debate arguments and
produce a clear investment recommendation with supporting rationale.

Inspired by TradingAgents' Research Manager role which bridges the analyst
research phase and the trader execution phase.
"""
import json
from ..agents.base import AgentContext, BaseAgent
from ..llm.client import LLMClient
from ..agents.schemas import ResearchPlan


_RESEARCH_MANAGER_PROMPT = (
    "You are the Research Manager overseeing an investment debate for {stock}. "
    "Your job is to synthesise the Bull and Bear arguments into a clear recommendation.\n\n"
    "Technical Signals: {signals}\n"
    "Market Indicators: {indicators}\n"
    "Sector: {sector}\n"
    "Market Regime: {regime}\n\n"
    "=== BULL ARGUMENTS ===\n"
    "{bull_args}\n\n"
    "=== BEAR ARGUMENTS ===\n"
    "{bear_args}\n\n"
    "Provide your recommendation considering:\n"
    "1. Which side has stronger evidence given the current market regime?\n"
    "2. Are there risk factors that override the technical signals?\n"
    "3. What is the appropriate position sizing?\n\n"
    "Recommendation must be one of: Buy, Overweight, Hold, Underweight, Sell.\n"
    "Output as JSON with keys: recommendation, rationale, strategic_actions, confidence"
)


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to readable markdown."""
    return (
        f"**Recommendation**: {plan.recommendation.value}\n\n"
        f"**Rationale**: {plan.rationale}\n\n"
        f"**Strategic Actions**: {plan.strategic_actions}\n\n"
        f"**Confidence**: {plan.confidence:.2f}"
    )


class ResearchManagerAgent(BaseAgent):
    """Research Manager - synthesises debate into structured plan."""

    name = "research_manager"
    description = "Synthesises bull/bear debate into investment recommendation"

    def __init__(self, llm_client: LLMClient = None):
        self.llm = llm_client or LLMClient.from_config()

    def execute(self, context: AgentContext) -> AgentContext:
        debate = context.invest_debate
        stock = context.current_debate_stock
        if debate is None or not debate.is_concluded:
            # No debate happened, use heuristic fallback
            self._heuristic_fallback(context)
            return context

        signals = self._summarize_signals(context, stock)
        indicators = self._get_indicators(context, stock)
        sector = self._get_sector(context, stock)
        regime = context.regime or "unknown"

        bull_args = "\n".join(f"- {a}" for a in debate.bull_history) or "No bull arguments."
        bear_args = "\n".join(f"- {a}" for a in debate.bear_history) or "No bear arguments."

        prompt = _RESEARCH_MANAGER_PROMPT.format(
            stock=stock, signals=signals, indicators=indicators,
            sector=sector, regime=regime,
            bull_args=bull_args, bear_args=bear_args,
        )

        response = self.llm.deep_chat([
            {"role": "system", "content": (
                "You are a seasoned investment Research Manager. Evaluate debate arguments "
                "and output a JSON with keys: recommendation, rationale, strategic_actions, confidence."
            )},
            {"role": "user", "content": prompt},
        ])

        plan = self._parse_response(response, stock, context)
        if plan:
            context.research_plan = plan
            context.warnings.append(
                f"[ResearchManager] {stock}: {plan.recommendation.value} "
                f"(confidence={plan.confidence:.2f})"
            )
        else:
            context.warnings.append(f"[ResearchManager] {stock}: parse failed, using fallback")
        return context

    def _parse_response(self, response: str, stock: str, ctx: AgentContext) -> ResearchPlan | None:
        """Parse LLM response into ResearchPlan, with fallback."""
        try:
            # Try to extract JSON from response
            if "{" in response:
                json_str = response[response.index("{"):response.rindex("}")+1]
                data = json.loads(json_str)
                return ResearchPlan(
                    recommendation=data.get("recommendation", "Hold"),
                    rationale=data.get("rationale", ""),
                    strategic_actions=data.get("strategic_actions", ""),
                    confidence=float(data.get("confidence", 0.5)),
                )
        except Exception:
            pass
        return self._heuristic_fallback(ctx)

    def _heuristic_fallback(self, ctx: AgentContext) -> ResearchPlan:
        """Heuristic fallback when LLM is unavailable. Returns ResearchPlan."""
        stock = ctx.current_debate_stock
        if not stock:
            ctx.research_plan = ResearchPlan(
                recommendation="Hold", rationale="No stock to analyze",
                strategic_actions="Wait for signals.", confidence=0.0,
            )
            return ctx.research_plan

        ind = ctx.indicators.get(stock, {})
        rsi = ind.get("rsi_14", 50)
        price_pos = ind.get("price_position", 0.5)
        signals = [s for s in ctx.ts_signals if s.get("stock") == stock]

        buy_signals = sum(1 for s in signals if s["type"] in ("valley", "up_trend_start", "lower_breakout"))
        sell_signals = sum(1 for s in signals if s["type"] in ("peak", "down_trend_start", "upper_breakout"))

        if rsi < 35 and buy_signals >= 1:
            rec = "Buy"
            conf = 0.7
        elif rsi > 70 and sell_signals >= 1:
            rec = "Sell"
            conf = 0.7
        elif buy_signals > sell_signals:
            rec = "Overweight" if buy_signals >= 2 else "Hold"
            conf = 0.5 + 0.1 * buy_signals
        elif sell_signals > buy_signals:
            rec = "Underweight" if sell_signals >= 2 else "Hold"
            conf = 0.5 + 0.1 * sell_signals
        else:
            rec = "Hold"
            conf = 0.5

        plan = ResearchPlan(
            recommendation=rec,
            rationale=f"Heuristic: RSI={rsi}, price_pos={price_pos:.2f}, buy_sig={buy_signals}, sell_sig={sell_signals}",
            strategic_actions=f"Position based on {rec} signal. Monitor RSI and volume.",
            confidence=conf,
        )
        ctx.research_plan = plan
        return plan

    def _summarize_signals(self, ctx: AgentContext, stock: str) -> str:
        hits = [s for s in ctx.ts_signals if s.get("stock") == stock]
        if not hits:
            return "No specific technical signals detected."
        return "; ".join(f"{s['type']}(conf={s['confidence']})" for s in hits[:5])

    def _get_indicators(self, ctx: AgentContext, stock: str) -> str:
        ind = ctx.indicators.get(stock, {})
        if not ind:
            return "No indicators available."
        return json.dumps({k: round(v, 3) if isinstance(v, float) else v
                          for k, v in ind.items()}, ensure_ascii=False)

    def _get_sector(self, ctx: AgentContext, stock: str) -> str:
        for hs in ctx.hot_sectors:
            if stock in hs.get("stocks", []):
                return hs.get("sector", "unknown")
        return "unknown"
