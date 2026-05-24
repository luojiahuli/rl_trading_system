#!/usr/bin/env python3
"""数据获取 Agent - 按热门板块拉取股票数据"""
import pandas as pd
from ..agents.base import AgentContext, BaseAgent
from ..data.fetcher import fetch_stock_daily
from ..data.indicators import compute_indicators
from config import START_DATE, END_DATE, EXCLUDED_SECTORS


class DataFetchAgent(BaseAgent):
    name = "data_fetch"
    description = "按热门板块获取 A 股股票数据和指标"

    def execute(self, context: AgentContext) -> AgentContext:
        stock_pool = []
        market_data = {}
        indicators = {}

        for hs in context.hot_sectors:
            sector = hs["sector"]
            # 排除金融券商板块
            if any(ex in sector for ex in EXCLUDED_SECTORS):
                continue

            stocks = hs.get("stocks", [])
            for code in stocks[:5]:
                try:
                    df = fetch_stock_daily(code, START_DATE, END_DATE)
                    if df is None or len(df) < 30:
                        continue

                    # 计算指标
                    df = compute_indicators(df)
                    market_data[code] = df
                    stock_pool.append(code)

                    # 最新指标
                    if len(df) > 0:
                        last = df.iloc[-1]
                        indicators[code] = {
                            "close": last.get("close", 0),
                            "pct_chg": last.get("pct_chg", 0),
                            "volume_ratio": last.get("volume_ratio", 1),
                            "rsi_14": last.get("rsi_14", 50),
                            "price_position": last.get("price_position", 0.5),
                            "atr": last.get("atr", 0),
                        }
                except Exception as e:
                    context.warnings.append(f"{code} 数据获取失败: {e}")
                    continue

        context.stock_pool = stock_pool
        context.market_data = market_data
        context.indicators = indicators
        context.warnings.append(f"获取 {len(stock_pool)} 只股票数据")
        return context
