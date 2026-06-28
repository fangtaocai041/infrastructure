"""
holland/tagging.py — 标签原理 (Tagging)
=========================================
Holland 原理②: 标签是促进选择性交互的标识机制。
"Tags are a pervasive mechanism for facilitating selective
interaction in CAS. They allow agents to distinguish between
entities that would otherwise be indistinguishable."

工程实现:
  - 多维标签图: 物种标签 × 理论标签 × 方法标签
  - 标签介导的选择性路由: 论文通过标签匹配找到相关理论
  - 标签涌现: 新标签自发出现 (如新研究方向的术语)

Holland 原文映射:
  "Tags have a powerful role in aggregation — they form the
   basis for the filtering and specialisation that leads to
   boundaries and emergent hierarchical organization."
   → 工程: 标签 = 维度过滤器 = 涌现层级的边界条件
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaggedNode:
    """带标签的图节点。"""
    id: str
    tags: set[str] = field(default_factory=set)
    weight: float = 1.0

    def match(self, query_tags: set[str]) -> float:
        """标签匹配度: Jaccard 相似度"""
        if not query_tags or not self.tags:
            return 0.0
        return len(self.tags & query_tags) / len(self.tags | query_tags)


class TagDynamics:
    """标签动力学: 检测标签涌现与演变。

    Holland: "Tags facilitate the formation of aggregates
              by enabling selective recognition."
    """

    def __init__(self):
        self._tag_timeline: list[dict[str, int]] = []  # 时间序列: {tag: count}
        self._emerged_tags: dict[str, float] = {}      # 新涌现标签: {tag: score}

    def record(self, tags_per_item: list[set[str]]):
        """记录一个时间步的标签分布。"""
        counter: dict[str, int] = {}
        for tags in tags_per_item:
            for t in tags:
                counter[t] = counter.get(t, 0) + 1
        self._tag_timeline.append(counter)
        self._detect_emerged_tags()

    def _detect_emerged_tags(self):
        """检测新涌现的标签: 近期频率/历史频率 > 阈值"""
        if len(self._tag_timeline) < 3:
            return

        recent = self._tag_timeline[-1]
        historical: dict[str, float] = defaultdict(float)
        for snap in self._tag_timeline[:-1]:
            for tag, count in snap.items():
                historical[tag] += count
        n = len(self._tag_timeline) - 1
        for tag, cnt in recent.items():
            hist_avg = historical.get(tag, 0.0) / max(n, 1)
            if hist_avg > 0 and cnt / max(hist_avg, 1e-9) > 2.0:
                self._emerged_tags[tag] = min(
                    1.0, (cnt / max(hist_avg, 1e-9)) / 5.0
                )

    @property
    def active_tags(self) -> list[tuple[str, float]]:
        """当前激活的涌现标签及其分数"""
        return sorted(
            self._emerged_tags.items(),
            key=lambda x: x[1], reverse=True
        )


class TaggedGraph:
    """标签图: 节点通过标签选择性连接。

    Holland: "Tags define the interactions that matter."
    """

    def __init__(self):
        self.nodes: dict[str, TaggedNode] = {}
        self.edges: dict[tuple[str, str], float] = {}
        self._tag_index: dict[str, set[str]] = defaultdict(set)  # tag → node_ids

    def add_node(self, node_id: str, tags: set[str], weight: float = 1.0):
        node = TaggedNode(id=node_id, tags=tags, weight=weight)
        self.nodes[node_id] = node
        for tag in tags:
            self._tag_index[tag].add(node_id)

    def route_by_tags(self, query_tags: set[str], top_k: int = 5
                      ) -> list[tuple[str, float]]:
        """标签选择性路由: 返回 top-k 匹配节点

        Holland: "Tags allow agents to selectively ignore
                  agents that do not carry matching tags."
        """
        scores: dict[str, float] = {}
        for tag in query_tags:
            for node_id in self._tag_index.get(tag, set()):
                node = self.nodes[node_id]
                scores[node_id] = node.match(query_tags)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def emerge_new_tag(self, candidate: str, threshold: float = 0.3) -> bool:
        """检测新标签是否可涌现为有效筛选器。

        判据: 新标签需匹配足够多的已有标签节点，形成"语义桥接"。
        """
        words = set(candidate.lower().split("_"))
        matches = 0
        for tag, nodes in self._tag_index.items():
            tag_words = set(tag.lower().split("_"))
            if len(words & tag_words) / max(len(words | tag_words), 1) > 0.3:
                matches += len(nodes)
        return matches / max(len(self.nodes), 1) >= threshold
