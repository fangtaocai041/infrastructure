"""
holland/flows.py — 流原理 (Flows)
===================================
Holland 原理④: 资源、能量、信息通过网络流动, 产生乘数效应。
"Flows of resources through a network of nodes — with the
multiplier effect and recycling — are a central feature of CAS."

工程实现:
  - 信息流网络: 论文引用图 → 知识流动路径
  - 乘数效应检测: 输入变化在网络中的放大倍数
  - 循环效果: 知识再循环增强

Holland 原文映射:
  "The multiplier effect: a resource injected at one node
   can be multiplied as it passes through the network."
   → 工程: 引用链乘数 = Π(节点权重) 沿路径累乘
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FlowPath:
    """知识流路径。"""
    nodes: list[str]           # 途经节点
    multiplier: float = 1.0    # 累乘效应
    cycles: int = 0            # 循环次数
    emergence_score: float = 0.0


class FlowNetwork:
    """Holland 流网络: 有向图 + 边权重 + 乘数效应。

    Holland: "The network structure determines the dynamics
              of flows — which nodes amplify, which attenuate."
    """

    def __init__(self):
        self.adj: dict[str, dict[str, float]] = defaultdict(dict)
        self.node_weights: dict[str, float] = {}

    def add_edge(self, source: str, target: str, weight: float = 1.0):
        self.adj[source][target] = weight
        if source not in self.node_weights:
            self.node_weights[source] = 1.0
        if target not in self.node_weights:
            self.node_weights[target] = 1.0

    def set_node_weight(self, node: str, weight: float):
        self.node_weights[node] = weight

    def trace_flow(self, start: str, max_depth: int = 5) -> list[FlowPath]:
        """追踪从起点出发的信息流。

        Holland: "Following the flows reveals the structure of
                  interactions in the system."
        """
        paths: list[FlowPath] = []
        queue: deque[tuple[str, list[str], float, int]] = deque(
            [(start, [start], 1.0, 0)]
        )

        while queue:
            node, path, multiplier, depth = queue.popleft()
            if depth >= max_depth:
                continue

            for neighbor, weight in self.adj.get(node, {}).items():
                new_mult = multiplier * weight * self.node_weights.get(
                    neighbor, 1.0
                )
                cycles = path.count(neighbor)  # Holland: 循环 = 再循环
                new_path = path + [neighbor]

                paths.append(FlowPath(
                    nodes=list(new_path),
                    multiplier=round(new_mult, 4),
                    cycles=cycles,
                    emergence_score=self._calc_flow_emergence(
                        new_mult, cycles, depth + 1
                    ),
                ))

                if neighbor not in path or cycles < 2:
                    queue.append((neighbor, new_path, new_mult, depth + 1))

        paths.sort(key=lambda p: p.emergence_score, reverse=True)
        return paths

    def multiplier_effect(self, source: str, target: str, depth: int = 5
                          ) -> float:
        """计算源到目标的乘数效应。

        Holland: "The multiplier is the ratio of total effect to
                  initial injection."
        """
        paths = self.trace_flow(source, depth)
        total = 1.0
        for p in paths:
            if p.nodes[-1] == target:
                total += p.multiplier
        return round(total, 4)

    def _calc_flow_emergence(self, multiplier: float, cycles: int,
                             depth: int) -> float:
        """流涌现度: 乘数↑ + 循环↑ + 深度↑ → 流涌现"""
        return min(1.0, (
            0.4 * min(multiplier / 10.0, 1.0) +
            0.3 * min(cycles / 5.0, 1.0) +
            0.3 * min(depth / 10.0, 1.0)
        ))


class FlowAnalyzer:
    """流分析器: 检测知识在网络中的涌现流。

    将论文引用图建模为 Holland 流网络, 检测:
      - 关键放大节点 (通过它流量倍增)
      - 循环路径 (知识自增强)
      - 涌现流 (高乘数 + 多循环)
    """

    def __init__(self):
        self.network = FlowNetwork()

    def build_from_citations(self, papers: list[dict]):
        """从论文列表构建引用流网络。

        Args:
            papers: list of {id, title, references: [id, ...],
                             citations: int}
        """
        for p in papers:
            pid = p.get("id", p.get("title", ""))
            weight = min(p.get("citation_count", 1) / 100.0 + 0.5, 5.0)
            self.network.set_node_weight(pid, weight)

            for ref in p.get("references", []):
                self.network.add_edge(pid, ref, weight=0.5)

        # 反向边: 被引论文 → 引用论文 (知识流出方向)
        for p in papers:
            pid = p.get("id", p.get("title", ""))
            for ref in p.get("references", []):
                if ref in self.network.node_weights:
                    weight = self.network.node_weights.get(ref, 1.0)
                    self.network.add_edge(ref, pid, weight=weight * 0.3)

    def find_emergence_nodes(self, top_k: int = 5
                             ) -> list[tuple[str, float]]:
        """找涌现节点: 出度 × 权重 最大的节点。

        Holland: "Nodes with high multiplier effect are where
                  emergence is most likely to manifest."
        """
        scores: dict[str, float] = {}
        for source in self.network.adj:
            paths = self.network.trace_flow(source, max_depth=3)
            scores[source] = sum(p.emergence_score for p in paths)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
