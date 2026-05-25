#!/usr/bin/env python3
"""热门板块挖掘 Agent"""
from ..agents.base import AgentContext, BaseAgent
from ..data.sector_map import extract_hot_sectors_from_news
import requests
import json
import os

# 自动检测系统代理（VPN 开启后自动生效）
_PROXY = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or os.getenv("ALL_PROXY")
_SESSION = requests.Session()
if _PROXY:
    _SESSION.proxies.update({"http": _PROXY, "https": _PROXY})
_SESSION.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})


def _fetch_sina_news() -> list:
    """从新浪财经获取最新新闻"""
    try:
        r = _SESSION.get(
            "https://feed.mix.sina.com.cn/api/roll/get",
            params={"pageid": 153, "lid": 2516, "k": "", "num": 10, "page": 1},
            timeout=10,
        )
        data = r.json()
        items = []
        for item in data.get("result", {}).get("data", []):
            items.append({
                "source": "sina",
                "title": item.get("title", ""),
                "content": item.get("intro", ""),
            })
        return items
    except Exception:
        return []


def _fetch_cls_news() -> list:
    """从财联社获取热点新闻（外网可访问）"""
    try:
        r = _SESSION.get(
            "https://www.cls.cn/api/telegraph",
            params={"category": "1", "limit": 10},
            timeout=10,
            headers={"Referer": "https://www.cls.cn/"},
        )
        data = r.json()
        items = []
        for item in data.get("data", {}).get("roll_data", []):
            items.append({
                "source": "cls",
                "title": item.get("title", ""),
                "content": item.get("content", "")[:200],
            })
        return items
    except Exception:
        return []


def _fetch_eastmoney_sectors() -> list:
    """从东方财富获取概念板块热度排行"""
    try:
        r = _SESSION.get(
            "https://push2.eastmoney.com/api/qt/clist/get",
            params={
                "pn": 1, "pz": 20, "po": 1, "np": 1,
                "fltt": 2, "invt": 2, "fid": "f3",
                "fs": "m:90+t:2",
                "fields": "f12,f14,f3",
            },
            timeout=10,
        )
        data = r.json()
        sectors = []
        for item in data.get("data", {}).get("diff", []):
            code = item.get("f12", "")
            name = item.get("f14", "")
            change = item.get("f3", 0)
            if name and change is not None:
                sectors.append({
                    "sector": name,
                    "heat_score": round(float(change) * 10 + 60, 1),
                    "summary": f"今日涨幅{change}%",
                    "stocks": [],
                })
        sectors.sort(key=lambda x: -x["heat_score"])
        return sectors[:10]
    except Exception:
        return []


class HotSectorMiningAgent(BaseAgent):
    name = "hot_sector_mining"
    description = "从新闻/板块行情中挖掘热门板块"

    def execute(self, context: AgentContext) -> AgentContext:
        news_items = []

        # 1. 尝试新浪财经新闻
        news_items = _fetch_sina_news()

        # 2. 财联社新闻后备（外网友好）
        if not news_items:
            news_items = _fetch_cls_news()

        # 3. 尝试东方财富板块数据
        if not news_items:
            hot_sectors = _fetch_eastmoney_sectors()
            if hot_sectors:
                context.hot_sectors = hot_sectors
                context.news_data = news_items
                context.warnings.append(f"发现 {len(hot_sectors)} 个热门板块(东方财富)")
                return context

        # 4. 有新闻则用新闻提取板块
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

        # 5. 最后尝试 AKShare 概念板块（需 VPN/代理）
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
