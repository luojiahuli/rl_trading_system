"""Two-tier LLM client: quick_thinking_llm for analysts, deep_thinking_llm for managers.

Supports Ollama (local) and OpenAI-compatible APIs with automatic fallback.
A-share specific: prompts are in Chinese-influenced English for technical analysis.
"""
import os
import json
import urllib.request
from typing import Optional


def _ollama_chat(model: str, messages: list, temperature: float = 0.7,
                 base_url: str = "http://localhost:11434") -> str:
    """Call Ollama API directly."""
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
    }).encode()
    url = f"{base_url.rstrip('/')}/api/chat"
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=10).read())
        return resp.get("message", {}).get("content", "")
    except Exception as e:
        return f"[LLM error: {e}]"


def _openai_chat(
    model: str,
    messages: list,
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    temperature: float = 0.7,
) -> str:
    """Call OpenAI-compatible API."""
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=10).read())
        return resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        return f"[LLM error: {e}]"


class LLMClient:
    """Two-tier LLM client.

    - quick_thinking_llm: cheaper/faster model for analysts, researchers, trader, risk debaters
    - deep_thinking_llm: more powerful model for managers (Research Manager, Portfolio Manager)
    """

    def __init__(self, config: dict = None):
        cfg = config or {}

        # Quick thinking (analysts, researchers, trader, risk)
        self.quick_provider = cfg.get("quick_provider",
                                       os.getenv("QUICK_LLM_PROVIDER", "ollama"))
        self.quick_model = cfg.get("quick_model",
                                    os.getenv("QUICK_LLM_MODEL", "qwen2.5:1.5b"))
        self.quick_api_key = cfg.get("quick_api_key",
                                      os.getenv("QUICK_LLM_API_KEY", ""))
        self.quick_base_url = cfg.get("quick_base_url",
                                       os.getenv("QUICK_LLM_BASE_URL", "http://localhost:11434"))

        # Deep thinking (managers)
        self.deep_provider = cfg.get("deep_provider",
                                      os.getenv("DEEP_LLM_PROVIDER", "ollama"))
        self.deep_model = cfg.get("deep_model",
                                   os.getenv("DEEP_LLM_MODEL", "qwen2.5:1.5b"))
        self.deep_api_key = cfg.get("deep_api_key",
                                     os.getenv("DEEP_LLM_API_KEY", ""))
        self.deep_base_url = cfg.get("deep_base_url",
                                      os.getenv("DEEP_LLM_BASE_URL", "http://localhost:11434"))

    def _call(self, provider: str, model: str, messages: list,
              api_key: str = "", base_url: str = "", temperature: float = 0.7) -> str:
        """Route to the correct provider."""
        if provider in ("openai", "openai-compatible"):
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                return "[LLM error: no API key for OpenAI provider]"
            return _openai_chat(model, messages, api_key,
                                base_url or "https://api.openai.com/v1", temperature)
        # Default: ollama
        return _ollama_chat(model, messages, temperature, base_url)

    def quick_chat(self, messages: list, temperature: float = 0.7) -> str:
        """Call the quick-thinking LLM (for analysts, researchers, trader, risk)."""
        return self._call(
            self.quick_provider, self.quick_model, messages,
            self.quick_api_key, self.quick_base_url, temperature,
        )

    def deep_chat(self, messages: list, temperature: float = 0.3) -> str:
        """Call the deep-thinking LLM (for managers). Lower temperature for more deterministic output."""
        return self._call(
            self.deep_provider, self.deep_model, messages,
            self.deep_api_key, self.deep_base_url, temperature,
        )

    @staticmethod
    def from_config() -> "LLMClient":
        """Create client from rl_trading_system config."""
        try:
            import config as cfg
            return LLMClient({
                "quick_provider": cfg.QUICK_LLM_PROVIDER,
                "quick_model": cfg.QUICK_LLM_MODEL,
                "quick_api_key": cfg.QUICK_LLM_API_KEY,
                "quick_base_url": cfg.QUICK_LLM_BASE_URL,
                "deep_provider": cfg.DEEP_LLM_PROVIDER,
                "deep_model": cfg.DEEP_LLM_MODEL,
                "deep_api_key": cfg.DEEP_LLM_API_KEY,
                "deep_base_url": cfg.DEEP_LLM_BASE_URL,
            })
        except (ImportError, AttributeError):
            return LLMClient()
