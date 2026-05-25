#!/usr/bin/env python3
"""数据获取模块 - AKShare (主) + BaoStock (备), 支持外网代理"""
import pandas as pd
import numpy as np
import os

# 代理配置：VPN 开启时走系统代理
_PROXY = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or os.getenv("ALL_PROXY")


def _set_akshare_proxy():
    """为 AKShare 配置代理（requests 层面）"""
    if _PROXY:
        import akshare as ak
        # AKShare 底层使用 requests，配置全局代理
        import requests
        requests.session().proxies.update({
            "http": _PROXY,
            "https": _PROXY,
        })


def fetch_stock_daily(symbol: str, start_date: str = "2024-01-01",
                      end_date: str = None) -> pd.DataFrame:
    """获取个股日线数据: AKShare → BaoStock → 空"""
    _set_akshare_proxy()

    # 1) AKShare（主力数据源，有 VPN 时稳定）
    try:
        import akshare as ak
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", "") if end_date else "",
            adjust="hfq",
        )
        if not df.empty:
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
    except Exception:
        pass

    # 2) BaoStock 后备（外网直连，无需 VPN）
    try:
        return _fetch_stock_daily_baostock(symbol, start_date, end_date)
    except Exception:
        pass

    return pd.DataFrame()


def _fetch_stock_daily_baostock(symbol: str, start_date: str,
                                 end_date: str = None) -> pd.DataFrame:
    """BaoStock 获取日线数据（外网可用，无需代理）"""
    import baostock as bs
    bs.login()
    try:
        prefix = "sh" if symbol.startswith("6") else "sz"
        rs = bs.query_history_k_data_plus(
            f"{prefix}.{symbol}",
            "date,open,high,low,close,volume,amount,pctChg,turn",
            start_date=start_date,
            end_date=end_date or pd.Timestamp.today().strftime("%Y-%m-%d"),
            frequency="d",
            adjustflag="2",  # 前复权
        )
        rows = []
        while rs.next():
            row = rs.get_row_data()
            if row[0]:
                rows.append(row)
        bs.logout()

        if len(rows) < 30:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=[
            "date", "open", "high", "low", "close", "volume",
            "amount", "pct_chg", "turnover",
        ])
        for col in ["open", "high", "low", "close", "volume",
                     "amount", "pct_chg", "turnover"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception:
        bs.logout()
        return pd.DataFrame()


def fetch_sector_stocks(sector_name: str) -> list:
    """获取某板块下的成分股列表"""
    _set_akshare_proxy()
    import akshare as ak
    try:
        df = ak.stock_board_industry_cons_em(symbol=sector_name)
        codes = df["代码"].tolist() if "代码" in df.columns else []
        return codes[:20]  # 最多取前 20
    except Exception:
        return []


def fetch_all_sectors() -> pd.DataFrame:
    """获取所有板块列表"""
    _set_akshare_proxy()
    import akshare as ak
    df = ak.stock_board_industry_name_em()
    return df


def fetch_sector_daily(sector_name: str, start_date: str = "2024-01-01",
                       end_date: str = None) -> pd.DataFrame:
    """获取板块指数日线"""
    _set_akshare_proxy()
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
    _set_akshare_proxy()
    import akshare as ak
    df = ak.stock_board_concept_name_em()
    return df
