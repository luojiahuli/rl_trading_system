#!/usr/bin/env python3
"""数据获取模块 - AKShare A 股数据"""
import pandas as pd
import numpy as np


def fetch_stock_daily(symbol: str, start_date: str = "2024-01-01",
                      end_date: str = None) -> pd.DataFrame:
    """获取个股日线数据"""
    import akshare as ak
    df = ak.stock_zh_a_hist(
        symbol=symbol,
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", "") if end_date else "",
        adjust="hfq",
    )
    if df.empty:
        return df
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns={
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume",
        "成交额": "amount", "振幅": "amplitude",
        "涨跌幅": "pct_chg", "涨跌额": "change",
        "换手率": "turnover",
    }, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def fetch_sector_stocks(sector_name: str) -> list:
    """获取某板块下的成分股列表"""
    import akshare as ak
    try:
        df = ak.stock_board_industry_cons_em(symbol=sector_name)
        codes = df["代码"].tolist() if "代码" in df.columns else []
        return codes[:20]  # 最多取前 20
    except Exception:
        return []


def fetch_all_sectors() -> pd.DataFrame:
    """获取所有板块列表"""
    import akshare as ak
    df = ak.stock_board_industry_name_em()
    return df


def fetch_sector_daily(sector_name: str, start_date: str = "2024-01-01",
                       end_date: str = None) -> pd.DataFrame:
    """获取板块指数日线"""
    import akshare as ak
    df = ak.stock_board_industry_hist_em(
        symbol=sector_name,
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", "") if end_date else "",
        adjust="",
    )
    if df.empty:
        return df
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns={
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume",
        "成交额": "amount", "振幅": "amplitude",
        "涨跌幅": "pct_chg", "涨跌额": "change", "换手率": "turnover",
    }, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def fetch_concept_boards() -> pd.DataFrame:
    """获取概念板块热度"""
    import akshare as ak
    df = ak.stock_board_concept_name_em()
    return df
