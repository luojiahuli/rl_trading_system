"""
进程内消息总线 — 基于 queue.Queue 的发布/订阅

智能体之间通过 topic 解耦通信，防止直接函数调用带来的阻塞链。
"""
import json
import logging
from queue import Queue, Empty
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MessageBus:
    """轻量级进程内消息总线"""

    def __init__(self):
        self._queues: dict[str, Queue] = {}
        self._topics: set[str] = set()

    # ── 发布 ──────────────────────────────────────────────

    def publish(self, topic: str, message: Any):
        """发布消息到主题，所有订阅者会收到"""
        if topic not in self._queues:
            self._queues[topic] = Queue()
        self._queues[topic].put(message)
        self._topics.add(topic)
        logger.debug(f"[Bus] publish → {topic}")

    # ── 订阅 / 消费 ───────────────────────────────────────

    def subscribe(self, topic: str) -> Queue:
        """订阅主题，返回消息队列"""
        if topic not in self._queues:
            self._queues[topic] = Queue()
        self._topics.add(topic)
        return self._queues[topic]

    def consume(self, topic: str, timeout: Optional[float] = None) -> Any:
        """消费一个消息（阻塞直到有新消息或超时）"""
        queue = self.subscribe(topic)
        try:
            msg = queue.get(timeout=timeout)
            logger.debug(f"[Bus] consume ← {topic}")
            return msg
        except Empty:
            return None

    # ── 状态查询 ──────────────────────────────────────────

    @property
    def topics(self) -> list[str]:
        return sorted(self._topics)

    def message_count(self, topic: str) -> int:
        return self._queues[topic].qsize() if topic in self._queues else 0

    # ── 批量 ──────────────────────────────────────────────

    def consume_all(self, topic: str, timeout: Optional[float] = 0.1) -> list:
        """消费主题下当前所有消息"""
        results = []
        while True:
            msg = self.consume(topic, timeout=timeout)
            if msg is None:
                break
            results.append(msg)
        return results

    def __repr__(self):
        return f"<MessageBus topics={self.topics}>"
