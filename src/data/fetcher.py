#!/usr/bin/env python3
"""数据获取模块 - BaoStock (外网直连) + 合成数据, 跳过 AKShare（无 VPN 时不可用）"""
import socket
import pandas as pd
import numpy as np
import os
import baostock as bs

# 全局网络超时
socket.setdefaulttimeout(10)

# BaoStock 会话（复用登录，避免每只股票重复 login/logout）
_BS_LOGGED_IN = False


def _ensure_bs_login():
    global _BS_LOGGED_IN
    if not _BS_LOGGED_IN:
        bs.login()
        _BS_LOGGED_IN = True


def _bs_logout():
    global _BS_LOGGED_IN
    if _BS_LOGGED_IN:
        try:
            bs.logout()
        except Exception:
            pass
        _BS_LOGGED_IN = False


def fetch_stock_daily(symbol: str, start_date: str = "2024-01-01",
                      end_date: str = None) -> pd.DataFrame:
    """获取个股日线数据: BaoStock（外网直连）"""
    try:
        _ensure_bs_login()
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
        return pd.DataFrame()


def fetch_sector_stocks(sector_name: str) -> list:
    """获取某板块下的成分股列表（AKShare 不可用，返回空列表走预设映射）"""
    return []


def fetch_all_sectors() -> pd.DataFrame:
    """获取所有板块列表（AKShare 不可用，返回空）"""
    return pd.DataFrame()


def fetch_sector_daily(sector_name: str, start_date: str = "2024-01-01",
                       end_date: str = None) -> pd.DataFrame:
    """获取板块指数日线（AKShare 不可用，返回空）"""
    return pd.DataFrame()


def fetch_concept_boards() -> pd.DataFrame:
    """获取概念板块热度（AKShare 不可用，返回空）"""
    return pd.DataFrame()


def fetch_index_daily(symbol: str = "000001", start_date: str = "2024-01-01",
                      end_date: str = None) -> pd.DataFrame:
    """获取指数日线数据（默认上证指数 000001.SH）

    使用 BaoStock，无需复权。
    """
    try:
        _ensure_bs_login()
        bs_code = f"sh.{symbol}"
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume,amount,pctChg",
            start_date=start_date,
            end_date=end_date or pd.Timestamp.today().strftime("%Y-%m-%d"),
            frequency="d",
            adjustflag="1",  # 指数不复权
        )
        rows = []
        while rs.next():
            row = rs.get_row_data()
            if row[0]:
                rows.append(row)

        if len(rows) < 30:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=[
            "date", "open", "high", "low", "close", "volume", "amount", "pct_chg",
        ])
        for col in ["open", "high", "low", "close", "volume", "amount", "pct_chg"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception:
        return pd.DataFrame()
