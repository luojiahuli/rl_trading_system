#!/usr/bin/env python3
"""热门板块挖掘 Agent — A 股版（ECC 架构：COLLECT → ENRICH → STORE）"""
from ..agents.base import AgentContext, BaseAgent
from scrapling_utils import (
    fetch_all_parallel,
    classify_sectors,
    classify_by_keywords,
    get_resolved_config,
    ContentHashCache,
)

_CACHE = ContentHashCache(ttl_minutes=30)


class HotSectorMiningAgent(BaseAgent):
    name = "hot_sector_mining"
    description = "从财经媒体挖掘 A 股热门板块"

    def execute(self, context: AgentContext) -> AgentContext:
        cfg = get_resolved_config()

        # ── Phase 1: COLLECT — 并行抓取 ──────────────────
        news_items = fetch_all_parallel(
            market="cn",
            max_workers=cfg.get("global", {}).get("max_workers", 5),
            max_per_source=cfg.get("markets", {}).get("cn", {}).get("max_news_per_source", 10),
        )
        context.news_data = [n.to_dict() for n in news_items]
        context.warnings.append(f"并行抓取: {len(news_items)} 条新闻")

        # ── Phase 2: ENRICH — 板块分类 ────────────────────
        if news_items:
            texts = [n.title + " " + (n.content or "") for n in news_items]
            use_llm = cfg.get("ai", {}).get("enabled", False)
            api_key = cfg.get("ai", {}).get("api_key", "")
            sectors = classify_sectors(texts, market="cn", api_key=api_key, use_llm=use_llm)
        else:
            sectors = None

        if sectors:
            hot_sectors = self._enrich_with_stocks(sectors)
        else:
            hot_sectors = self._preset_sectors()
            context.warnings.append("无新闻数据，使用预设板块")

        # ── Phase 3: STORE — 输出 ─────────────────────────
        context.hot_sectors = hot_sectors
        context.warnings.append(f"A 股热门板块: {len(hot_sectors)} 个")

        pool = []
        for s in hot_sectors:
            for code in s.get("stocks", []):
                if code not in pool:
                    pool.append(code)
        context.stock_pool = pool
        return context

    def _enrich_with_stocks(self, classified: list[dict]) -> list[dict]:
        """为板块结果补充股票代码"""
        from ..data.sector_map import get_sector_stock_codes

        results = []
        for item in classified[:8]:
            codes = get_sector_stock_codes(item["sector"])
            results.append({
                "sector": item["sector"],
                "heat_score": item["heat_score"],
                "summary": f"热度{item['heat_score']}({item.get('source', 'keyword')})",
                "stocks": codes[:6],
            })
        return results

    def _preset_sectors(self) -> list[dict]:
        from ..data.sector_map import SECTOR_KEYWORDS
        results = []
        heat = 80
        for sector in SECTOR_KEYWORDS:
            results.append({
                "sector": sector,
                "heat_score": heat,
                "summary": f"热度{heat}(预设)",
                "stocks": [],
            })
            heat -= 5
        return results[:8]
