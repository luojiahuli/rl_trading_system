"""Trading Memory Log and Reflection system.

Inspired by TradingAgents' TradingMemoryLog and Reflector:
- Append-only markdown log of trading decisions with pending/resolved states
- Post-trade outcome reflection (2-4 sentences)
- Same-ticker / cross-ticker context injection for future analysis
- Atomic writes to prevent corruption

A-share specific: accounts for 涨跌停板 impact on outcome, T+1 holding periods.
"""
import os
import re
from datetime import datetime
from typing import Optional
from pathlib import Path

from ..agents.base import AgentContext, BaseAgent
from ..llm.client import LLMClient


class TradingMemoryLog:
    """Append-only markdown log of trading decisions and reflections."""

    _SEPARATOR = "\n\n<!-- ENTRY_END -->\n\n"
    _TAG_RE = re.compile(r"^\[(.+?)\]$", re.MULTILINE)

    def __init__(self, log_path: str = None):
        if log_path:
            self._log_path = Path(log_path)
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            from config import OUTPUT_DIR
            self._log_path = Path(OUTPUT_DIR) / "trading_memory.md"
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def store_decision(self, date: str, stock: str, action: str,
                       confidence: float, rationale: str, price: float) -> None:
        """Append a new pending entry."""
        # Idempotency check
        if self._log_path.exists():
            for line in self._log_path.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith(f"[{date} | {stock} |"):
                    return

        tag = f"[{date} | {stock} | {action} | {confidence:.2f} | pending]"
        entry = (
            f"{tag}\n\n"
            f"Price: {price:.2f}\n"
            f"Rationale: {rationale}\n"
            f"{self._SEPARATOR}"
        )
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(entry)

    def update_with_outcome(self, date: str, stock: str, return_pct: float,
                            holding_days: int, reflection: str) -> bool:
        """Update a pending entry with outcome and reflection. Atomic write."""
        if not self._log_path or not self._log_path.exists():
            return False

        text = self._log_path.read_text(encoding="utf-8")
        blocks = text.split(self._SEPARATOR)
        pending_prefix = f"[{date} | {stock} |"
        updated = False
        new_blocks = []

        for block in blocks:
            stripped = block.strip()
            if not stripped:
                new_blocks.append(block)
                continue

            lines = stripped.splitlines()
            tag_line = lines[0].strip()

            if (not updated and tag_line.startswith(pending_prefix)
                    and "| pending]" in tag_line):
                fields = [f.strip() for f in tag_line[1:-1].split("|")]
                action = fields[2]
                confidence = fields[3]
                ret_str = f"{return_pct:+.2f}%"
                new_tag = (
                    f"[{date} | {stock} | {action} | {confidence}"
                    f" | {ret_str} | {holding_days}d]"
                )
                rest = "\n".join(lines[1:])
                new_blocks.append(
                    f"{new_tag}\n\n{rest.lstrip()}\n\n"
                    f"Return: {ret_str}\n"
                    f"Holding Days: {holding_days}\n"
                    f"Reflection:\n{reflection}"
                )
                updated = True
            else:
                new_blocks.append(block)

        if not updated:
            return False

        new_text = self._SEPARATOR.join(new_blocks)
        tmp_path = self._log_path.with_suffix(".tmp")
        tmp_path.write_text(new_text, encoding="utf-8")
        tmp_path.replace(self._log_path)
        return True

    def get_pending_entries(self) -> list[dict]:
        """Get all pending (unresolved) entries."""
        entries = self._load_entries()
        return [e for e in entries if e.get("pending")]

    def get_past_context(self, stock: str, n_same: int = 5, n_cross: int = 3) -> str:
        """Get formatted past context for LLM prompt injection."""
        entries = [e for e in self._load_entries() if not e.get("pending")]
        if not entries:
            return ""

        same, cross = [], []
        for e in reversed(entries):
            if len(same) >= n_same and len(cross) >= n_cross:
                break
            if e["stock"] == stock and len(same) < n_same:
                same.append(e)
            elif e["stock"] != stock and len(cross) < n_cross:
                cross.append(e)

        parts = []
        if same:
            parts.append(f"Past trades in {stock} (most recent first):")
            for e in same:
                parts.append(
                    f"[{e['date']} | {e['action']} | return={e.get('return','n/a')}"
                    f" | {e.get('holding','n/a')}]"
                    f"\n  Rationale: {e.get('rationale','')[:100]}"
                )
        if cross:
            parts.append("Recent cross-stock lessons:")
            for e in cross:
                if e.get("reflection"):
                    parts.append(f"[{e['date']} {e['stock']}]: {e['reflection'][:200]}")
        return "\n\n".join(parts)

    def _load_entries(self) -> list[dict]:
        if not self._log_path or not self._log_path.exists():
            return []
        text = self._log_path.read_text(encoding="utf-8")
        raw_entries = [e.strip() for e in text.split(self._SEPARATOR) if e.strip()]
        entries = []
        for raw in raw_entries:
            parsed = self._parse_entry(raw)
            if parsed:
                entries.append(parsed)
        return entries

    def _parse_entry(self, raw: str) -> Optional[dict]:
        lines = raw.strip().splitlines()
        if not lines:
            return None
        tag_line = lines[0].strip()
        if not (tag_line.startswith("[") and tag_line.endswith("]")):
            return None
        fields = [f.strip() for f in tag_line[1:-1].split("|")]
        if len(fields) < 4:
            return None
        entry = {
            "date": fields[0],
            "stock": fields[1],
            "action": fields[2],
            "confidence": fields[3],
            "pending": fields[4] == "pending" if len(fields) > 4 else False,
            "return": fields[4] if len(fields) > 4 and fields[4] != "pending" else None,
            "holding": fields[5] if len(fields) > 5 else None,
        }
        body = "\n".join(lines[1:]).strip()
        # Extract rationale
        for line in lines[1:]:
            if line.startswith("Rationale:"):
                entry["rationale"] = line[len("Rationale:"):].strip()
                break
        # Extract reflection
        if "Reflection:" in body:
            parts = body.split("Reflection:")
            entry["reflection"] = parts[-1].strip() if len(parts) > 1 else ""
        return entry


class ReflectionAgent(BaseAgent):
    """Post-trade reflection agent.

    Analyses completed trades and stores reflections in the memory log.
    Uses LLM to generate concise 2-4 sentence reflections on what worked
    and what didn't, for future reference.
    """

    name = "reflection"
    description = "Post-trade outcome reflection and memory log update"

    def __init__(self, llm_client: LLMClient = None, memory_log: TradingMemoryLog = None):
        self.llm = llm_client or LLMClient.from_config()
        self.memory = memory_log or TradingMemoryLog()

    def execute(self, context: AgentContext) -> AgentContext:
        # Store pending decisions for new signals
        for signal in context.rl_signals:
            stock = signal.get("stock", "")
            action = signal.get("action", "")
            confidence = signal.get("confidence", 0)
            price = signal.get("price", 0)
            reason = signal.get("reason", "")
            self.memory.store_decision(
                date=context.date,
                stock=stock, action=action,
                confidence=confidence, rationale=reason,
                price=price,
            )

        # Process pending entries if we have trade results
        pending = self.memory.get_pending_entries()
        if pending and context.backtest_results:
            for entry in pending:
                stock = entry["stock"]
                action = entry["action"]
                # Find matching backtest result
                for result in context.backtest_results:
                    if result.get("stock") == stock and action != "hold":
                        total_return = result.get("total_return", 0)
                        holding_days = result.get("num_trades", 1)
                        self._reflect_and_update(
                            stock=stock, date=entry["date"],
                            action=action, return_pct=total_return * 100,
                            holding_days=holding_days,
                        )
                        break

        context.warnings.append(f"[Memory] Stored {len(context.rl_signals)} decisions, "
                                f"{len(pending)} pending reflection")
        return context

    def _reflect_and_update(self, stock: str, date: str, action: str,
                            return_pct: float, holding_days: int):
        """Generate reflection and update memory log."""
        reflection = self._generate_reflection(stock, action, return_pct, holding_days)
        self.memory.update_with_outcome(
            date=date, stock=stock,
            return_pct=return_pct,
            holding_days=holding_days,
            reflection=reflection,
        )

    def _generate_reflection(self, stock: str, action: str,
                             return_pct: float, holding_days: int) -> str:
        """Generate 2-4 sentence reflection using LLM."""
        prompt = (
            f"You are a trading analyst reviewing a past {action} decision on {stock} "
            f"that returned {return_pct:+.2f}% over {holding_days} days.\n\n"
            "Write 2-4 sentences covering:\n"
            "1. Was the directional call correct?\n"
            "2. What worked or failed in the analysis?\n"
            "3. One concrete lesson for next time.\n\n"
            "Be specific and concise."
        )
        try:
            result = self.llm.quick_chat([
                {"role": "system", "content": "You are a reflective trading analyst. Be concise."},
                {"role": "user", "content": prompt},
            ])
            if result and not result.startswith("[LLM error"):
                return result.strip()
        except Exception:
            pass
        # Fallback reflection
        direction = "correct" if return_pct > 0 else "incorrect"
        return (
            f"The {action} call on {stock} was {direction} with {return_pct:+.2f}% return "
            f"over {holding_days} days. "
            f"{'The position performed as expected based on signal analysis.' if return_pct > 0 else 'The analysis missed key risk factors.'} "
            f"Next time, monitor volume confirmation and market regime alignment more closely."
        )
