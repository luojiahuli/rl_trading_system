"""
SQLite 数据库管理器 — 持久化 Agent 运行结果、交易信号、模型标签等

6 张业务表:
  - agent_logs       Agent 执行日志
  - hot_sectors      热门板块快照
  - trading_signals  交易信号（含时间窗口关联）
  - backtest_results 回测结果
  - model_labels     模型有效标签（核心）
  - market_cache     市场数据缓存
"""
import os
import json
import sqlite3
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class DatabaseManager:
    """SQLite 数据库管理器"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn: sqlite3.Connection | None = None

    # ── 连接 / 初始化 ─────────────────────────────────────

    def connect(self):
        """建立连接并建表（自动创建目录）"""
        if self.conn is not None:
            return self
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_tables()
        logger.info(f"[DB] 已连接: {self.db_path}")
        return self

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def _init_tables(self):
        cur = self.conn.cursor()

        # 1. Agent 执行日志
        cur.execute("""
            CREATE TABLE IF NOT EXISTS agent_logs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name      TEXT NOT NULL,
                date            TEXT NOT NULL,
                status          TEXT NOT NULL DEFAULT 'ok',
                input_summary   TEXT,
                output_summary  TEXT,
                execution_time_ms INTEGER,
                error           TEXT,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 2. 热门板块快照
        cur.execute("""
            CREATE TABLE IF NOT EXISTS hot_sectors (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                date        TEXT NOT NULL,
                sector      TEXT NOT NULL,
                heat_score  REAL,
                source      TEXT,
                stocks_json TEXT,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 3. 交易信号
        cur.execute("""
            CREATE TABLE IF NOT EXISTS trading_signals (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                date              TEXT NOT NULL,
                stock             TEXT NOT NULL,
                signal_type       TEXT DEFAULT 'rl',
                action            TEXT NOT NULL,
                confidence        REAL,
                reason            TEXT,
                ts_signal_window  TEXT,
                created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 4. 回测结果
        cur.execute("""
            CREATE TABLE IF NOT EXISTS backtest_results (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                date            TEXT NOT NULL,
                strategy_name   TEXT NOT NULL,
                total_return    REAL,
                sharpe_ratio    REAL,
                max_drawdown    REAL,
                num_trades      INTEGER,
                regime          TEXT,
                params_json     TEXT,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 5. 模型有效标签（核心 — 沉淀模型计算的有效标签）
        cur.execute("""
            CREATE TABLE IF NOT EXISTS model_labels (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                date            TEXT NOT NULL,
                model_name      TEXT NOT NULL,
                label_type      TEXT NOT NULL,
                label_value     TEXT NOT NULL,
                confidence      REAL,
                features_json   TEXT,
                is_effective    INTEGER DEFAULT 1,
                verified_by     TEXT,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 6. 市场数据缓存
        cur.execute("""
            CREATE TABLE IF NOT EXISTS market_cache (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_code      TEXT NOT NULL,
                date            TEXT NOT NULL,
                data_type       TEXT NOT NULL DEFAULT 'daily',
                data_json       TEXT,
                expires_at      TIMESTAMP,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(stock_code, date, data_type)
            )
        """)

        # 索引
        cur.execute("CREATE INDEX IF NOT EXISTS idx_agent_logs_date ON agent_logs(date)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_hot_sectors_date ON hot_sectors(date)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_trading_signals_date ON trading_signals(date)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_backtest_date ON backtest_results(date)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_model_labels_date ON model_labels(date)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_model_labels_type ON model_labels(label_type)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_market_cache_lookup ON market_cache(stock_code, date, data_type)")

        self.conn.commit()

    # ── Agent 日志 ────────────────────────────────────────

    def save_agent_log(self, agent_name: str, date: str, status: str = "ok",
                       input_summary: str = "", output_summary: str = "",
                       execution_time_ms: int = 0, error: str = ""):
        self.conn.execute(
            "INSERT INTO agent_logs(agent_name,date,status,input_summary,output_summary,execution_time_ms,error) "
            "VALUES(?,?,?,?,?,?,?)",
            (agent_name, date, status, input_summary[:500], output_summary[:500],
             execution_time_ms, error),
        )
        self.conn.commit()

    # ── 热门板块 ──────────────────────────────────────────

    def save_hot_sectors(self, date: str, sectors: list[dict]):
        for s in sectors:
            stocks = s.get("stocks", [])
            self.conn.execute(
                "INSERT INTO hot_sectors(date,sector,heat_score,source,stocks_json) VALUES(?,?,?,?,?)",
                (date, s["sector"], s.get("heat_score", 0),
                 s.get("summary", ""), json.dumps(stocks, ensure_ascii=False)),
            )
        self.conn.commit()

    def get_hot_sectors(self, date: str, limit: int = 10) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM hot_sectors WHERE date=? ORDER BY heat_score DESC LIMIT ?",
            (date, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── 交易信号 ──────────────────────────────────────────

    def save_trading_signals(self, date: str, signals: list[dict]):
        for s in signals:
            self.conn.execute(
                "INSERT INTO trading_signals(date,stock,signal_type,action,confidence,reason,ts_signal_window) "
                "VALUES(?,?,?,?,?,?,?)",
                (date, s.get("stock", ""), s.get("signal_type", "rl"),
                 s.get("action", ""), s.get("confidence", 0),
                 s.get("reason", ""), json.dumps(s.get("ts_signal_window", {}))),
            )
        self.conn.commit()

    def get_trading_signals(self, date: str, action: str = None) -> list[dict]:
        if action:
            rows = self.conn.execute(
                "SELECT * FROM trading_signals WHERE date=? AND action=? ORDER BY confidence DESC",
                (date, action),
            )
        else:
            rows = self.conn.execute(
                "SELECT * FROM trading_signals WHERE date=? ORDER BY confidence DESC", (date,))
        return [dict(r) for r in rows.fetchall()]

    # ── 回测结果 ──────────────────────────────────────────

    def save_backtest_results(self, date: str, results: list[dict], regime: str = ""):
        for r in results:
            self.conn.execute(
                "INSERT INTO backtest_results(date,strategy_name,total_return,sharpe_ratio,"
                "max_drawdown,num_trades,regime,params_json) VALUES(?,?,?,?,?,?,?,?)",
                (date, r.get("strategy", ""), r.get("total_return", 0),
                 r.get("sharpe_ratio", 0), r.get("max_drawdown", 0),
                 r.get("num_trades", 0), regime,
                 json.dumps({k: v for k, v in r.items()
                            if k in ("total_return", "sharpe_ratio", "max_drawdown", "num_trades")})),
            )
        self.conn.commit()

    def get_backtest_results(self, date: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM backtest_results WHERE date=? ORDER BY sharpe_ratio DESC", (date,))
        return [dict(r) for r in rows.fetchall()]

    # ── 模型标签（核心）────────────────────────────────────

    def save_model_label(self, date: str, model_name: str, label_type: str,
                         label_value: str, confidence: float = 0,
                         features: dict = None, is_effective: bool = True,
                         verified_by: str = ""):
        self.conn.execute(
            "INSERT INTO model_labels(date,model_name,label_type,label_value,confidence,"
            "features_json,is_effective,verified_by) VALUES(?,?,?,?,?,?,?,?)",
            (date, model_name, label_type, label_value, confidence,
             json.dumps(features or {}, ensure_ascii=False),
             1 if is_effective else 0, verified_by),
        )
        self.conn.commit()

    def save_model_labels_batch(self, date: str, labels: list[dict]):
        for lb in labels:
            self.save_model_label(
                date=date,
                model_name=lb.get("model_name", ""),
                label_type=lb.get("label_type", ""),
                label_value=lb.get("label_value", ""),
                confidence=lb.get("confidence", 0),
                features=lb.get("features"),
                is_effective=lb.get("is_effective", True),
                verified_by=lb.get("verified_by", ""),
            )

    def get_effective_labels(self, date: str = None, label_type: str = None) -> list[dict]:
        sql = "SELECT * FROM model_labels WHERE is_effective=1"
        params = []
        if date:
            sql += " AND date=?"
            params.append(date)
        if label_type:
            sql += " AND label_type=?"
            params.append(label_type)
        sql += " ORDER BY confidence DESC"
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def get_label_summary(self, date: str) -> list[dict]:
        """获取某日各类型标签统计"""
        rows = self.conn.execute("""
            SELECT label_type, COUNT(*) as cnt, AVG(confidence) as avg_conf,
                   SUM(is_effective) as effective_cnt
            FROM model_labels WHERE date=?
            GROUP BY label_type
        """, (date,))
        return [dict(r) for r in rows.fetchall()]

    # ── 市场缓存 ──────────────────────────────────────────

    def save_market_cache(self, stock_code: str, date: str, data_type: str,
                          data: dict, ttl_hours: int = 24):
        from datetime import timedelta
        expires = datetime.now() + timedelta(hours=ttl_hours)
        self.conn.execute(
            "INSERT OR REPLACE INTO market_cache(stock_code,date,data_type,data_json,expires_at) "
            "VALUES(?,?,?,?,?)",
            (stock_code, date, data_type, json.dumps(data, ensure_ascii=False),
             expires.isoformat()),
        )
        self.conn.commit()

    def get_market_cache(self, stock_code: str, date: str, data_type: str = "daily") -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM market_cache WHERE stock_code=? AND date=? AND data_type=? AND "
            "expires_at > datetime('now')", (stock_code, date, data_type),
        ).fetchone()
        return dict(row) if row else None

    # ── 统计 ──────────────────────────────────────────────

    def table_stats(self) -> dict[str, int]:
        """各表记录数"""
        stats = {}
        for table in ("agent_logs", "hot_sectors", "trading_signals",
                      "backtest_results", "model_labels", "market_cache"):
            row = self.conn.execute(f"SELECT COUNT(*) as cnt FROM {table}").fetchone()
            stats[table] = row["cnt"]
        return stats
