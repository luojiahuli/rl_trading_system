"""Bull/Bear researcher debate agents.

Inspired by TradingAgents: two agents with opposing mandates debate the
investment merits of each stock, then a Research Manager synthesises the
results into a structured ResearchPlan.

The debate improves decision quality by forcing both sides to be considered
before a recommendation is made, reducing confirmation bias.
"""
import json
from ..agents.base import AgentContext, BaseAgent
from ..llm.client import LLMClient


_BULL_PROMPT = (
    "You are a bullish researcher analyzing A-share stocks. Your job is to find "
    "reasons to be bullish on {stock} based on the technical signals and market data below.\n\n"
    "Technical Signals: {signals}\n"
    "Market Indicators: {indicators}\n"
    "Sector: {sector}\n"
    "Market Regime: {regime}\n\n"
    "{opponent_context}\n"
    "Your bullish argument (2-4 sentences, be specific with numbers):"
)

_BEAR_PROMPT = (
    "You are a bearish researcher analyzing A-share stocks. Your job is to find "
    "reasons to be bearish on {stock} based on the technical signals and market data below.\n\n"
    "Technical Signals: {signals}\n"
    "Market Indicators: {indicators}\n"
    "Sector: {sector}\n"
    "Market Regime: {regime}\n\n"
    "{opponent_context}\n"
    "Your bearish argument (2-4 sentences, be specific with numbers):"
)


class BullResearcherAgent(BaseAgent):
    """Bullish researcher - argues for the long side in the debate."""

    name = "bull_researcher"
    description = "Bullish researcher - argues buy/long case based on signals"

    def __init__(self, llm_client: LLMClient = None):
        self.llm = llm_client or LLMClient.from_config()

    def execute(self, context: AgentContext) -> AgentContext:
        debate = context.invest_debate
        if debate is None or debate.is_concluded:
            return context

        stock = context.current_debate_stock
        signals = self._summarize_signals(context, stock)
        indicators = self._get_indicators(context, stock)
        sector = self._get_sector(context, stock)
        regime = context.regime or "unknown"

        # Build opponent context - what the bear said previously
        bear_args = debate.bear_history
        opponent_ctx = ""
        if bear_args:
            opponent_ctx = f"The bearish researcher previously argued:\n{bear_args[-1]}\n\nRespond to their concerns."

        prompt = _BULL_PROMPT.format(
            stock=stock, signals=signals, indicators=indicators,
            sector=sector, regime=regime, opponent_context=opponent_ctx,
        )
        argument = self.llm.quick_chat([
            {"role": "system", "content": "You are a bullish A-share stock analyst. Be concise and specific."},
            {"role": "user", "content": prompt},
        ])
        debate.add_bull_argument(argument)
        context.warnings.append(f"[Bull] {stock}: {argument[:80]}...")
        return context

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


class BearResearcherAgent(BaseAgent):
    """Bearish researcher - argues against the long side in the debate."""

    name = "bear_researcher"
    description = "Bearish researcher - argues sell/short case based on signals"

    def __init__(self, llm_client: LLMClient = None):
        self.llm = llm_client or LLMClient.from_config()

    def execute(self, context: AgentContext) -> AgentContext:
        debate = context.invest_debate
        if debate is None or debate.is_concluded:
            return context

        stock = context.current_debate_stock
        signals = self._summarize_signals(context, stock)
        indicators = self._get_indicators(context, stock)
        sector = self._get_sector(context, stock)
        regime = context.regime or "unknown"

        bull_args = debate.bull_history
        opponent_ctx = ""
        if bull_args:
            opponent_ctx = f"The bullish researcher argued:\n{bull_args[-1]}\n\nRespond to their points."

        prompt = _BEAR_PROMPT.format(
            stock=stock, signals=signals, indicators=indicators,
            sector=sector, regime=regime, opponent_context=opponent_ctx,
        )
        argument = self.llm.quick_chat([
            {"role": "system", "content": "You are a bearish A-share stock analyst. Be concise and specific."},
            {"role": "user", "content": prompt},
        ])
        debate.add_bear_argument(argument)
        context.warnings.append(f"[Bear] {stock}: {argument[:80]}...")
        return context

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
