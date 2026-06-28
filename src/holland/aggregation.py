"""
holland/aggregation.py — 聚集原理 (Aggregation)
=================================================
Holland 原理①: 简单主体通过局部规则聚集成复杂元主体。
"Emergence occurs when relatively simple elements organize into
complex wholes, with properties not present in the parts."

工程实现:
  - 检测数据中的聚集模式 (cluster emergence)
  - 构建层级 MetaAgent 树
  - 度量聚集度: 信息熵降低 → 新层级涌现

Holland 原文映射:
  "Aggregation is the emergence of complex large-scale behaviors
   from the aggregate interactions of less complex agents."
   → 工程: 数据点 → 类别 → 主题 → 理论域 (四层聚集)
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetaAgent:
    """Holland 意义上的元主体: 简单主体的聚合产物。

    Holland: "Meta-agents are themselves agents,
              capable of aggregating at higher levels."
    """
    name: str
    level: int = 0                  # 聚集层级 (0=原子, 1=类别, 2=主题, 3=理论域)
    members: list[str] = field(default_factory=list)
    children: list["MetaAgent"] = field(default_factory=list)
    emergence_score: float = 0.0    # 聚集涌现度 (0~1)
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.members) + sum(c.size for c in self.children)

    def describe(self) -> str:
        return (f"MetaAgent({self.name}, level={self.level}, "
                f"size={self.size}, emergence={self.emergence_score:.2f})")


class AggregationDetector:
    """检测数据中的聚集涌现。

    Holland: "We can think of aggregation as a general process
              in which agents combine to form meta-agents."

    算法: 自适应层次聚类 + 信息增益检测
      - Level 0: 原始数据点 (原子主体)
      - Level 1: 类别聚集 (如论文按关键词聚类)
      - Level 2: 主题聚集 (如类别按理论域聚类)
      - Level 3: 涌现域 (一个新主题自发形成)

    涌现判据: 当 Level N+1 的信息熵显著低于 Level N 的期望值时,
              认为发生了一次聚集涌现。
    """

    def __init__(self, emergence_threshold: float = 0.5):
        self.threshold = emergence_threshold
        self._history: list[list[MetaAgent]] = []

    def detect(self, items: list[dict], key_field: str = "keywords"
               ) -> list[MetaAgent]:
        """检测聚集涌现。

        Args:
            items: 数据点列表 (论文/数据记录/观测值)
            key_field: 用于聚集的关键字段

        Returns:
            层级 MetaAgent 树
        """
        # Level 1: 按共同属性聚集
        groups: dict[str, list[str]] = defaultdict(list)
        for i, item in enumerate(items):
            keys = self._extract_keys(item, key_field)
            for k in keys:
                groups[k].append(f"item_{i}")

        level1 = [
            MetaAgent(
                name=key,
                level=1,
                members=members,
                emergence_score=self._calc_emergence(members, len(items)),
            )
            for key, members in groups.items()
            if len(members) >= 2  # Holland: 至少2个主体才能聚集
        ]

        # Level 2: 元主体间再次聚集 (主题级)
        level2 = self._aggregate_meta(level1, level=2)

        # 涌现检测: 从 Level 1 → Level 2 的信息增益
        for ma in level2:
            child_entropy = sum(
                c.emergence_score * math.log(max(c.emergence_score, 1e-9))
                for c in ma.children
            ) / max(len(ma.children), 1)
            ma.emergence_score = 1.0 - abs(child_entropy) / math.log(2)

        self._history.append(level1 + level2)
        return [a for a in level1 + level2 if a.emergence_score >= self.threshold]

    def _extract_keys(self, item: dict, field: str) -> list[str]:
        val = item.get(field, "")
        if isinstance(val, list):
            return [str(v).lower().strip() for v in val]
        if isinstance(val, str):
            return [t.strip().lower() for t in val.split(",") if t.strip()]
        return [str(val)]

    def _calc_emergence(self, members: list, total: int) -> float:
        """涌现度 = 1 - (成员熵 / 最大熵)"""
        if total == 0:
            return 0.0
        p = len(members) / total
        return 1.0 + p * math.log(max(p, 1e-9))

    def _aggregate_meta(self, agents: list[MetaAgent], level: int
                        ) -> list[MetaAgent]:
        """高阶聚集: 将 level N 的元主体聚集成 level N+1 的主主体"""
        # 按成员重叠度聚集
        meta: list[MetaAgent] = []
        used = set()
        for i, a1 in enumerate(agents):
            if i in used:
                continue
            children = [a1]
            for j, a2 in enumerate(agents[i + 1 :], i + 1):
                if j in used:
                    continue
                overlap = len(set(a1.members) & set(a2.members))
                if overlap > 0:
                    children.append(a2)
                    used.add(j)
            if len(children) > 1:
                all_members = list(set(
                    m for c in children for m in c.members
                ))
                meta.append(MetaAgent(
                    name=f"theme_{len(meta)}",
                    level=level,
                    members=all_members,
                    children=children,
                    emergence_score=0.0,
                ))
            used.add(i)

        return meta
