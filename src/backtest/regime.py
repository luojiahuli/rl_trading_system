#!/usr/bin/env python3
"""市场状态分类"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class MarketRegimeClassifier:
    """市场状态分类器"""

    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.model = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)

    def fit(self, df: pd.DataFrame):
        """基于特征聚类识别市场状态"""
        features = self._extract_features(df)
        if len(features) < self.n_regimes:
            return self
        self.model.fit(features)
        self._labels = self.model.labels_
        self._feature_names = ["return_5d", "volatility_20d", "volume_trend", "rsi_avg"]
        return self

    def predict(self, df: pd.DataFrame) -> int:
        """预测当前市场状态"""
        features = self._extract_features(df)
        if len(features) == 0:
            return 0
        return int(self.model.predict(features[-1:])[0])

    def get_regime_name(self, label: int) -> str:
        """获取状态名称"""
        names = {0: "震荡市", 1: "牛市", 2: "熊市"}
        return names.get(label, f"状态{label}")

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """提取市场特征"""
        close = df["close"].values.astype(float)
        volume = df["volume"].values.astype(float)
        features = []
        for i in range(20, len(df)):
            ret_5d = close[i] / close[i-5] - 1
            vol_20d = np.std(np.diff(close[i-20:i]) / close[i-20:i-1]) * np.sqrt(252)
            vol_trend = volume[i] / np.mean(volume[max(0, i-20):i])
            rsi_vals = df.get("rsi_14", pd.Series([50] * len(df))).values
            rsi_avg = np.mean(rsi_vals[max(0, i-5):i])
            features.append([ret_5d, vol_20d, vol_trend, rsi_avg])
        return np.array(features)
