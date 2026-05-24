#!/usr/bin/env python3
"""本地 LLM 问答 Agent - Qwen2.5-1.5B"""
import json
from ..agents.base import AgentContext, BaseAgent
from config import LLM_MODEL_PATH, OLLAMA_MODEL


class QAAgent(BaseAgent):
    name = "qa"
    description = "本地 Qwen2.5-1.5B 回答用户关于市场机会的问题"

    def __init__(self):
        self._llm = None

    def execute(self, context: AgentContext) -> AgentContext:
        # 构造上下文给 LLM
        prompt = self._build_prompt(context)
        context.report_text = prompt

        try:
            answer = self._query_llm(prompt)
            context.report_text = f"## 💡 智能问答\n\n{answer}"
        except Exception as e:
            context.warnings.append(f"LLM 问答失败（回退到模板回答）: {e}")
            context.report_text = self._fallback_answer(context)

        return context

    def _build_prompt(self, ctx: AgentContext) -> str:
        sectors = ", ".join([s["sector"] for s in ctx.hot_sectors[:5]])
        signals_summary = ", ".join([
            f"{s['stock']}({s['action']})" for s in ctx.rl_signals[:5]
        ])
        regime = ctx.regime or "未知"

        return f"""你是量化交易分析师。基于以下今日分析结果，回答用户问题。

## 今日市场数据
- 日期: {ctx.date}
- 市场状态: {regime}
- 热门板块: {sectors}
- 交易信号: {signals_summary or "无"}

## 策略表现
{json.dumps(ctx.strategy_results.get("strategy_performance", {}), indent=2, ensure_ascii=False) if ctx.strategy_results else "暂无"}

## 风控指标
{json.dumps(ctx.risk_metrics.get("drawdown", {}), indent=2, ensure_ascii=False) if ctx.risk_metrics else "暂无"}

请回答：今天哪些板块有机会？推荐什么操作策略？有什么风险要注意？
"""

    def _query_llm(self, prompt: str) -> str:
        """调用本地 LLM"""
        if LLM_MODEL_PATH:
            return self._query_llama_cpp(prompt)
        else:
            return self._query_ollama(prompt)

    def _query_ollama(self, prompt: str) -> str:
        import requests
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json().get("response", "")
        raise ConnectionError(f"Ollama 返回 {resp.status_code}")

    def _query_llama_cpp(self, prompt: str) -> str:
        from llama_cpp import Llama
        if self._llm is None:
            self._llm = Llama(
                model_path=LLM_MODEL_PATH,
                n_ctx=2048,
                n_threads=4,
            )
        output = self._llm(prompt, max_tokens=512, temperature=0.7, stop=["<|im_end|>"])
        return output["choices"][0]["text"].strip() if output.get("choices") else ""

    def _fallback_answer(self, ctx: AgentContext) -> str:
        """无 LLM 时的模板回答"""
        lines = [f"## 📊 {ctx.date} 市场机会分析\n"]
        if ctx.hot_sectors:
            lines.append("### 🔥 热门板块\n")
            for s in ctx.hot_sectors[:5]:
                lines.append(f"- **{s['sector']}** (热度: {s['heat_score']})")
        if ctx.rl_signals:
            lines.append("\n### 📈 交易信号\n")
            for s in ctx.rl_signals[:5]:
                lines.append(f"- {s['stock']}: **{s['action']}** "
                             f"(置信度: {s['confidence']:.0%}, {s.get('reason', '')})")
        if ctx.risk_metrics:
            dd = ctx.risk_metrics.get("drawdown", {})
            lines.append(f"\n### ⚠️ 风控\n- 回撤水平: {dd.get('dd_pct', 0):.2%}")
            lines.append(f"- 建议: {dd.get('message', '正常')}")
        return "\n".join(lines)
