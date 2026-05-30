"""Three-perspective risk debate agents: Aggressive, Conservative, Neutral.

Inspired by TradingAgents' risk management debate cycle. Each agent represents
a distinct risk perspective and argues for their position sizing approach.
The Portfolio Manager then synthesises the final decision.

A-share specific: considers 涨跌停板 limits, T+1 settlement, market cap liquidity.
"""
import json
from ..agents.base import AgentContext, BaseAgent
from ..llm.client import LLMClient
from ..agents.schemas import RiskAssessment


# ── System prompts for each risk perspective ──
# NOTE: {{ and }} are escaped for Python str.format()

_AGGRESSIVE_PROMPT = (
    "You are an AGGRESSIVE risk analyst for A-share trading. "
    "You believe the current opportunity justifies above-average risk exposure.\n\n"
    "Stock: {stock}\n"
    "Research Recommendation: {recommendation}\n"
    "Confidence: {confidence}\n"
    "Technical Signals: {signals}\n"
    "Market Regime: {regime}\n"
    "Current Indicators: {indicators}\n\n"
    "{opponent_context}\n"
    "Argue for a larger position size and wider stop-loss. "
    "Consider A-share specific factors: 10%/20% 涨跌停板, T+1 settlement, 沪深 liquidity.\n"
    "What is your maximum position size recommendation (0-100% of allocation)? "
    "What stop-loss level is appropriate?\n"
    'Output as JSON: {{"max_position_pct": float, "stop_loss_pct": float, "verdict": str, "score": float}}'
)

_CONSERVATIVE_PROMPT = (
    "You are a CONSERVATIVE risk analyst for A-share trading. "
    "You prioritise capital preservation and caution.\n\n"
    "Stock: {stock}\n"
    "Research Recommendation: {recommendation}\n"
    "Confidence: {confidence}\n"
    "Technical Signals: {signals}\n"
    "Market Regime: {regime}\n"
    "Current Indicators: {indicators}\n\n"
    "{opponent_context}\n"
    "Argue for a smaller position size and tighter stop-loss. "
    "Consider A-share specific risks: T+1 means you cannot exit intraday, "
    "涨跌停板 may trap positions, 市场情绪 shifts rapidly.\n"
    "What is your maximum position size recommendation (0-100% of allocation)? "
    "What stop-loss level is appropriate?\n"
    'Output as JSON: {{"max_position_pct": float, "stop_loss_pct": float, "verdict": str, "score": float}}'
)

_NEUTRAL_PROMPT = (
    "You are a NEUTRAL risk analyst for A-share trading. "
    "You provide balanced risk assessment based purely on data.\n\n"
    "Stock: {stock}\n"
    "Research Recommendation: {recommendation}\n"
    "Confidence: {confidence}\n"
    "Technical Signals: {signals}\n"
    "Market Regime: {regime}\n"
    "Current Indicators: {indicators}\n\n"
    "{opponent_context}\n"
    "Provide a balanced position size recommendation. "
    "Consider A-share factors: historical volatility, volume liquidity, sector rotation.\n"
    "What is your maximum position size recommendation (0-100% of allocation)? "
    "What stop-loss level is appropriate?\n"
    'Output as JSON: {{"max_position_pct": float, "stop_loss_pct": float, "verdict": str, "score": float}}'
)


class AggressiveRiskAgent(BaseAgent):
    """Aggressive risk perspective - argues for higher position sizing."""

    name = "risk_aggressive"
    description = "Aggressive risk analyst - argues for larger positions"

    def __init__(self, llm_client: LLMClient = None):
        self.llm = llm_client or LLMClient.from_config()

    def execute(self, context: AgentContext) -> AgentContext:
        debate = context.risk_debate
        if debate is None or debate.is_concluded:
            return context
        return self._argue(context, debate, "aggressive", _AGGRESSIVE_PROMPT)

    def _argue(self, ctx, debate, perspective, prompt_template):
        stock = ctx.current_debate_stock or "unknown"
        plan = ctx.research_plan
        signals = self._summarize_signals(ctx, stock)
        indicators = self._get_indicators(ctx, stock)
        regime = ctx.regime or "unknown"

        # Build opponent context
        other_args = debate.conservative_history + debate.neutral_history
        opponent_ctx = ""
        if other_args:
            opponent_ctx = "Previous arguments from other analysts:\n" + "\n".join(
                f"- {a}" for a in other_args[-2:]
            )

        prompt = prompt_template.format(
            stock=stock,
            recommendation=plan.recommendation.value if plan else "Hold",
            confidence=plan.confidence if plan else 0.5,
            signals=signals, indicators=indicators,
            regime=regime, opponent_context=opponent_ctx,
        )

        response = self.llm.quick_chat([
            {"role": "system", "content": f"You are a {perspective} A-share risk analyst."},
            {"role": "user", "content": prompt},
        ])

        assessment = self._parse_response(response, perspective)
        if assessment:
            debate.add_argument(perspective, assessment.verdict)
            ctx.risk_assessments[perspective] = assessment
            ctx.warnings.append(
                f"[{perspective.capitalize()}Risk] {stock}: "
                f"max_pos={assessment.max_position_pct:.0%}, "
                f"stop={assessment.stop_loss_pct:.0%}"
            )
        return ctx

    def _parse_response(self, response: str, perspective: str = "aggressive") -> RiskAssessment | None:
        try:
            if "{" in response:
                json_str = response[response.index("{"):response.rindex("}")+1]
                data = json.loads(json_str)
                defaults = {"aggressive": (0.3, 0.08, 0.0),
                            "conservative": (0.1, 0.05, 0.0),
                            "neutral": (0.2, 0.06, 0.0)}
                max_pos, stop, score = defaults.get(perspective, (0.2, 0.06, 0.0))
                return RiskAssessment(
                    perspective=perspective,
                    max_position_pct=float(data.get("max_position_pct", max_pos)),
                    stop_loss_pct=float(data.get("stop_loss_pct", stop)),
                    verdict=data.get("verdict", ""),
                    score=float(data.get("score", score)),
                )
        except Exception:
            pass
        return None

    def _summarize_signals(self, ctx, stock):
        hits = [s for s in ctx.ts_signals if s.get("stock") == stock]
        return "; ".join(f"{s['type']}(conf={s['confidence']})" for s in hits[:5]) or "None"

    def _get_indicators(self, ctx, stock):
        ind = ctx.indicators.get(stock, {})
        return json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in ind.items()},
                         ensure_ascii=False) if ind else "None"


class ConservativeRiskAgent(BaseAgent):
    """Conservative risk perspective - argues for smaller positions."""

    name = "risk_conservative"
    description = "Conservative risk analyst - argues for smaller positions"

    def __init__(self, llm_client: LLMClient = None):
        self.llm = llm_client or LLMClient.from_config()

    def execute(self, context: AgentContext) -> AgentContext:
        debate = context.risk_debate
        if debate is None or debate.is_concluded:
            return context
        return AggressiveRiskAgent._argue(self, context, debate, "conservative", _CONSERVATIVE_PROMPT)


class NeutralRiskAgent(BaseAgent):
    """Neutral risk perspective - provides balanced assessment."""

    name = "risk_neutral"
    description = "Neutral risk analyst - balanced risk assessment"

    def __init__(self, llm_client: LLMClient = None):
        self.llm = llm_client or LLMClient.from_config()

    def execute(self, context: AgentContext) -> AgentContext:
        debate = context.risk_debate
        if debate is None or debate.is_concluded:
            return context
        return AggressiveRiskAgent._argue(self, context, debate, "neutral", _NEUTRAL_PROMPT)
