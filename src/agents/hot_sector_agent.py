#!/usr/bin/env python3
"""热门板块挖掘 Agent"""
from ..agents.base import AgentContext, BaseAgent
from ..data.sector_map import extract_hot_sectors_from_news
import akshare as ak


class HotSectorMiningAgent(BaseAgent):
    name = "hot_sector_mining"
    description = "从央视新闻联播、政策新闻中挖掘热门板块"

    def execute(self, context: AgentContext) -> AgentContext:
        news_items = []

        # 1. 央视新闻联播
        try:
            cctv = ak.news_cctv(date=context.date)
            if not cctv.empty:
                for _, row in cctv.iterrows():
                    news_items.append({
                        "source": "cctv",
                        "title": str(row.get("title", "")),
                        "content": str(row.get("content", "")),
                    })
        except Exception as e:
            context.warnings.append(f"央视新闻获取失败: {e}")

        # 2. 如果没获取到新闻，用本地板块热度数据
        if not news_items:
            try:
                boards = ak.stock_board_concept_name_em()
                if not boards.empty and "名称" in boards.columns:
                    hot_sectors = []
                    for _, row in boards.head(10).iterrows():
                        name = row.get("名称", "")
                        hot_sectors.append({
                            "sector": str(name),
                            "heat_score": round(float(row.get("关注度", 0)), 2),
                            "summary": f"概念板块热度排名",
                            "stocks": [],
                        })
                    context.hot_sectors = hot_sectors
                    context.news_data = news_items
                    return context
            except Exception as e2:
                context.warnings.append(f"板块热度获取失败: {e2}")

        # 3. 提取热门板块
        hot_sectors_raw = extract_hot_sectors_from_news(news_items)

        # 4. 获取对应股票代码
        hot_sectors = []
        for sector, score in hot_sectors_raw[:8]:
            try:
                df = ak.stock_board_industry_cons_em(symbol=sector)
                stocks = df["代码"].tolist()[:10] if "代码" in df.columns else []
            except Exception:
                stocks = []
            hot_sectors.append({
                "sector": sector,
                "heat_score": score,
                "summary": f"新闻热点板块",
                "stocks": stocks,
            })

        context.hot_sectors = hot_sectors
        context.news_data = news_items
        context.warnings.append(f"发现 {len(hot_sectors)} 个热门板块")
        return context
