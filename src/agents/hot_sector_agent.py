#!/usr/bin/env python3
"""热门板块挖掘 Agent — 基于 scrapling_utils"""
from ..agents.base import AgentContext, BaseAgent
from ..data.sector_map import extract_hot_sectors_from_news
from scrapling_utils import SmartFetcher
from scrapling_utils.news_sources import (
    SinaFinanceNews, CailiansheNews, EastMoneySectorNews
)

_fetcher = SmartFetcher()
_sina = SinaFinanceNews()
_sina.fetcher = _fetcher
_cls = CailiansheNews()
_cls.fetcher = _fetcher
_em = EastMoneySectorNews()
_em.fetcher = _fetcher


class HotSectorMiningAgent(BaseAgent):
    name = "hot_sector_mining"
    description = "从新闻/板块行情中挖掘热门板块"

    def execute(self, context: AgentContext) -> AgentContext:
        news_items = []

        # 1. 新浪财经
        sina_items = _sina.fetch(lid="2516")
        news_items = [n.to_dict() for n in sina_items]

        # 2. 财联社
        if not news_items:
            cls_items = _cls.fetch()
            news_items = [n.to_dict() for n in cls_items]

        # 3. 东方财富板块热度
        if not news_items:
            sectors = _em.fetch()
            if sectors:
                context.hot_sectors = sectors
                context.news_data = []
                context.warnings.append(f"发现 {len(sectors)} 个热门板块(东方财富)")
                return context

        # 4. 新闻 → 板块提取
        if news_items:
            hot_sectors_raw = extract_hot_sectors_from_news(news_items)
            hot_sectors = []
            for sector, score in hot_sectors_raw[:8]:
                hot_sectors.append({
                    "sector": sector,
                    "heat_score": score * 10,
                    "summary": "新闻热点",
                    "stocks": [],
                })
            context.hot_sectors = hot_sectors
            context.news_data = news_items
            context.warnings.append(f"从 {len(news_items)} 条新闻发现 {len(hot_sectors)} 个热门板块")
            return context

        # 5. AKShare 兜底
        try:
            import akshare as ak
            boards = ak.stock_board_concept_name_em()
            if not boards.empty and "名称" in boards.columns:
                hot_sectors = []
                for _, row in boards.head(10).iterrows():
                    name = row.get("名称", "")
                    hot_sectors.append({
                        "sector": str(name),
                        "heat_score": round(float(row.get("关注度", 0)), 2),
                        "summary": "概念板块热度排名",
                        "stocks": [],
                    })
                context.hot_sectors = hot_sectors
                context.news_data = news_items
                return context
        except Exception as e2:
            context.warnings.append(f"AKShare 板块失败: {e2}")

        context.news_data = news_items
        return context
