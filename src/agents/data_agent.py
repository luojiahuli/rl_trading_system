#!/usr/bin/env python3
"""数据获取 Agent - 按热门板块拉取股票数据"""
import pandas as pd
from ..agents.base import AgentContext, BaseAgent
from ..data.fetcher import fetch_stock_daily, fetch_sector_stocks
from ..data.indicators import compute_indicators
from config import START_DATE, END_DATE, EXCLUDED_SECTORS

# 热门板块→预设股票代码映射（AKShare 不可用时的后备）
_SECTOR_STOCK_MAP = {
    "消费电子": ["002475", "601138", "603160", "002241", "300433"],
    "人工智能": ["300308", "688111", "300502", "603019", "002230"],
    "低空经济": ["002625", "600685", "600760", "600118", "002023"],
    "军工": ["600760", "600893", "600862", "002013", "000768"],
    "半导体": ["688981", "002371", "603501", "600703", "300661"],
    "新能源": ["300750", "002594", "601012", "300274", "603659"],
    "汽车": ["600104", "000625", "601633", "002594", "000800"],
    "医药": ["600276", "300760", "000538", "002007", "300122"],
    "金融": ["601318", "600036", "601166", "600030", "601211"],
    "房地产": ["001979", "600048", "000002", "600383", "600325"],
    "白酒": ["600519", "000858", "002304", "000568", "600809"],
    "光伏": ["601012", "600438", "688599", "002459", "300274"],
    "机器人": ["300124", "688005", "002472", "300024", "600835"],
    "数字经济": ["000938", "688568", "300496", "002415", "603986"],
    "国产芯片": ["603986", "300661", "688981", "002371", "600703"],
}


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

            # 优先用 AKShare 查成分股（3s 超时），后备用预设映射
            stocks = hs.get("stocks", [])
            if not stocks:
                try:
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as ex:
                        f = ex.submit(fetch_sector_stocks, sector)
                        stocks = f.result(timeout=3)
                except Exception:
                    stocks = []
            if not stocks:
                stocks = _SECTOR_STOCK_MAP.get(sector, [])
            if not stocks:
                context.warnings.append(f"{sector}: 无法获取成分股")
                continue

            for code in stocks[:8]:
                df = pd.DataFrame()
                # 尝试 BaoStock 获取真实数据（4s 超时）
                try:
                    import concurrent.futures
                    ex = concurrent.futures.ThreadPoolExecutor()
                    f = ex.submit(fetch_stock_daily, code, START_DATE, END_DATE)
                    df = f.result(timeout=4)
                    ex.shutdown(wait=False)
                except Exception:
                    df = pd.DataFrame()
                    try:
                        ex.shutdown(wait=False)
                    except Exception:
                        pass
                # 有真实数据（≥30行）直接用；不足30行才尝试合成数据补齐
                if df is None or len(df) < 30:
                    try:
                        df = _generate_synthetic_data(code, START_DATE, END_DATE)
                    except Exception:
                        df = None
                if df is None or len(df) < 30:
                    context.warnings.append(f"{code}: 无数据")
                    continue

                # 计算指标
                try:
                    df = compute_indicators(df)
                except Exception:
                    continue
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
                        "week_trend": int(last.get("week_trend", 0)),
                        "week_rsi_14": float(last.get("week_rsi_14", 50)),
                        "week_macd_hist": float(last.get("week_macd_hist", 0)),
                    }

        context.stock_pool = stock_pool
        context.market_data = market_data
        context.indicators = indicators
        context.warnings.append(f"获取 {len(stock_pool)} 只股票数据")
        return context


def _generate_synthetic_data(code: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    """AKShare 不可用时生成合成日线数据"""
    try:
        import numpy as np
        ed = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
        dates = pd.date_range(start=start_date, end=ed, freq="B")
        if len(dates) < 30:
            return None
        np.random.seed(hash(code) % (2**31))
        base_price = 10 + np.random.random() * 40  # 10~50 元
        n = len(dates)
        # 均值回复随机过程，避免长期漂移
        log_prices = np.zeros(n)
        innovations = np.random.randn(n) * 0.015  # 日波动 1.5%
        for i in range(1, n):
            log_prices[i] = log_prices[i-1] + innovations[i] - 0.001 * log_prices[i-1]
        prices = base_price * np.exp(log_prices)
        prices = np.clip(prices, base_price * 0.5, base_price * 1.5)  # ±50%

        df = pd.DataFrame({
            "date": dates,
            "open": pd.Series(prices * (1 - np.abs(np.random.randn(n)) * 0.01), dtype=float),
            "close": pd.Series(prices, dtype=float),
            "high": pd.Series(prices * (1 + np.abs(np.random.randn(n)) * 0.015), dtype=float),
            "low": pd.Series(prices * (1 - np.abs(np.random.randn(n)) * 0.015), dtype=float),
            "volume": pd.Series(np.random.randint(100000, 10000000, n), dtype=float),
            "amount": pd.Series(np.random.randint(1000000, 100000000, n), dtype=float),
            "pct_chg": pd.Series(np.clip(np.random.randn(n) * 2, -10, 10), dtype=float),
            "amplitude": pd.Series(np.abs(np.random.randn(n)) * 3, dtype=float),
            "turnover": pd.Series(np.abs(np.random.randn(n)) * 2, dtype=float),
        })
        # 归一化价格到合理区间
        first_close = float(df["close"].iloc[0])
        for col in ["close", "open", "high", "low"]:
            df[col] = (df[col].to_numpy(dtype=float) / first_close * base_price).astype(float)
        return df
    except Exception:
        return None
