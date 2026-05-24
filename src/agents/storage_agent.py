#!/usr/bin/env python3
"""存储 Agent — 将各 Agent 输出持久化到 SQLite"""
import time
from ..agents.base import AgentContext, BaseAgent


class StorageAgent(BaseAgent):
    name = "storage"
    description = "将各 Agent 输出持久化到 SQLite 数据库，沉淀模型有效标签"

    def execute(self, context: AgentContext) -> AgentContext:
        db = context.db
        if db is None:
            context.warnings.append("数据库未初始化，跳过持久化")
            return context

        date = context.date
        t0 = time.time()

        # 1. 持久化热门板块
        if context.hot_sectors:
            db.save_hot_sectors(date, context.hot_sectors)
            context.warnings.append(f"已保存 {len(context.hot_sectors)} 个热门板块")

        # 2. 持久化交易信号（含标签）
        if context.rl_signals:
            db.save_trading_signals(date, context.rl_signals)
            self._label_signals(db, date, context.rl_signals)

        # 3. 持久化回测结果
        if context.backtest_results:
            db.save_backtest_results(date, context.backtest_results, context.regime)
            self._label_backtest(db, date, context.backtest_results, context.regime)

        # 4. 持久化市场状态标签
        if context.regime:
            db.save_model_label(
                date=date,
                model_name="market_regime_kmeans",
                label_type="market_regime",
                label_value=context.regime,
                confidence=0.7,
                features={"n_clusters": 3},
                is_effective=True,
            )

        # 5. 持久化风控标签
        if context.risk_metrics:
            dd = context.risk_metrics.get("drawdown", {})
            db.save_model_label(
                date=date,
                model_name="risk_manager",
                label_type="drawdown",
                label_value=dd.get("level", "normal"),
                confidence=max(0, 1 - abs(dd.get("dd_pct", 0))),
                features={"dd_pct": dd.get("dd_pct", 0)},
                is_effective=dd.get("level") != "critical",
            )
            if "var_95" in context.risk_metrics:
                db.save_model_label(
                    date=date,
                    model_name="risk_manager",
                    label_type="value_at_risk",
                    label_value=f"95% VaR",
                    confidence=0.95,
                    features={"var_95": context.risk_metrics["var_95"]},
                    is_effective=True,
                )

        # 6. 记录 agent 日志
        elapsed = int((time.time() - t0) * 1000)
        db.save_agent_log(self.name, date, status="ok",
                          output_summary=f"存储 {len(context.hot_sectors)} 板块, "
                                         f"{len(context.rl_signals)} 信号, "
                                         f"{len(context.backtest_results)} 回测",
                          execution_time_ms=elapsed)

        # 打印统计
        stats = db.table_stats()
        context.warnings.append(
            f"数据库统计: agent_logs={stats.get('agent_logs',0)} "
            f"hot_sectors={stats.get('hot_sectors',0)} "
            f"signals={stats.get('trading_signals',0)} "
            f"backtest={stats.get('backtest_results',0)} "
            f"labels={stats.get('model_labels',0)}"
        )

        return context

    # ── 内部打标辅助 ──────────────────────────────────────

    def _label_signals(self, db, date: str, signals: list[dict]):
        """为交易信号生成模型标签"""
        buy_signals = [s for s in signals if s.get("action") == "buy"]
        sell_signals = [s for s in signals if s.get("action") == "sell"]

        labels = []
        if buy_signals:
            avg_conf = sum(s.get("confidence", 0) for s in buy_signals) / len(buy_signals)
            labels.append({
                "model_name": "rl_trading_heuristic",
                "label_type": "buy_signal",
                "label_value": f"{len(buy_signals)}个买入信号",
                "confidence": avg_conf,
                "features": {"count": len(buy_signals), "avg_confidence": avg_conf},
                "is_effective": avg_conf > 0.5,
            })
        if sell_signals:
            avg_conf = sum(s.get("confidence", 0) for s in sell_signals) / len(sell_signals)
            labels.append({
                "model_name": "rl_trading_heuristic",
                "label_type": "sell_signal",
                "label_value": f"{len(sell_signals)}个卖出信号",
                "confidence": avg_conf,
                "features": {"count": len(sell_signals), "avg_confidence": avg_conf},
                "is_effective": avg_conf > 0.5,
            })
        if labels:
            db.save_model_labels_batch(date, labels)

    def _label_backtest(self, db, date: str, results: list[dict], regime: str):
        """为回测结果生成模型标签"""
        labels = []
        for r in results:
            sharpe = r.get("sharpe_ratio", 0)
            ret = r.get("total_return", 0)
            labels.append({
                "model_name": f"strategy_{r.get('strategy', 'unknown')}",
                "label_type": "strategy_performance",
                "label_value": f"Sharpe={sharpe:.3f} Return={ret*100:+.2f}%",
                "confidence": max(0, min(1, (sharpe + 1) / 3)),
                "features": {"total_return": ret, "sharpe_ratio": sharpe,
                             "max_drawdown": r.get("max_drawdown", 0)},
                "is_effective": sharpe > 0.5,
            })
        if labels:
            db.save_model_labels_batch(date, labels)
