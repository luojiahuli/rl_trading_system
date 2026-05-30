#!/bin/bash
# 每日回测复盘 + 自动推送 GitHub
# 运行: bash run_daily.sh
# 建议 cron: 30 15 * * 1-5 (交易日 15:30)

set -e

cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"
LOG_FILE="${PROJECT_DIR}/output/logs/daily_run.log"

mkdir -p "${PROJECT_DIR}/output/logs"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ====== 开始每日回测 ======" | tee -a "$LOG_FILE"

# 1. 运行回测
/usr/local/bin/python3 run_backtest_review.py >> "$LOG_FILE" 2>&1
RET=$?
if [ $RET -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ 回测失败 (exit=$RET)" | tee -a "$LOG_FILE"
    exit $RET
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ 回测完成" | tee -a "$LOG_FILE"

# 2. 检查是否有变更
cd "$PROJECT_DIR"

# 检测 Git 配置
if ! git config user.name > /dev/null 2>&1; then
    git config user.name "luojiahuli"
    git config user.email "luojiahuli@users.noreply.github.com"
fi

# 暂存变更文件
git add run_backtest_review.py run_daily.sh src/backtest/ README.md output/reports/backtest_review_*.json output/reports/equity_curves_*.csv

if git diff --cached --quiet; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ℹ️  无变更，跳过推送" | tee -a "$LOG_FILE"
else
    DATE_TAG=$(date '+%Y-%m-%d')
    git commit -m "daily backtest review ${DATE_TAG}"
    git push origin main 2>&1 | tee -a "$LOG_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ 已推送 GitHub" | tee -a "$LOG_FILE"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ====== 完成 ======" | tee -a "$LOG_FILE"
