"""
holland/diversity.py — 多样性原理 (Diversity)
===============================================
Holland 原理⑤: 异质主体的多样性是 CAS 适应的基础。
"Diversity is a dynamic pattern in CAS. It arises from the
ongoing adaptation of agents to niches and each other."

工程实现:
  - 多样性指数: Shannon-Wiener / Simpson / Berger-Parker
  - 生态位多样性: 物种在研究空间中的分布均匀度
  - 涌现检测: 多样性突然增加 = 新生态位涌现

Holland 原文映射:
  "Diversity is not just a static ensemble — it is the
   generator of further diversity through recombination."
   → 工程: 多样性 = 适应度景观的覆盖面积
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field


@dataclass
class DiversitySnapshot:
    """多样性快照。"""
    shannon: float = 0.0
    simpson: float = 0.0
    richness: int = 0           # 物种数 (类比: 标签/类别数)
    evenness: float = 0.0       # Pielou 均匀度
    dominant_share: float = 0.0 # Berger-Parker 优势度


class DiversityIndex:
    """Holland 多样性量度: 多指标综合。

    Holland: "Diversity begets diversity — in a CAS,
              the presence of diverse agents creates new niches
              which can be filled by yet more diverse agents."
    """

    def __init__(self):
        self._history: list[DiversitySnapshot] = []
        self._emergence_events: list[dict] = []

    def compute(self, distribution: dict[str, int]
                ) -> DiversitySnapshot:
        """计算多样性指数。

        Shannon:   H' = -Σ (p_i × ln p_i)     (信息多样性)
        Simpson:   D  = 1 - Σ p_i²             (均匀度偏重)
        Evenness:  J  = H' / ln(S)             (Pielou)
        Dominance: max(p_i)                    (Berger-Parker)
        """
        total = sum(distribution.values())
        if total == 0:
            snap = DiversitySnapshot()
            self._history.append(snap)
            return snap

        proportions = {k: v / total for k, v in distribution.items()}
        richness = len(distribution)

        shannon = -sum(p * math.log(max(p, 1e-9)) for p in proportions.values())
        simpson = 1.0 - sum(p * p for p in proportions.values())
        evenness = shannon / max(math.log(richness), 1e-9) if richness > 1 else 0.0
        dominant = max(proportions.values()) if proportions else 0.0

        snap = DiversitySnapshot(
            shannon=round(shannon, 4),
            simpson=round(simpson, 4),
            richness=richness,
            evenness=round(evenness, 4),
            dominant_share=round(dominant, 4),
        )
        self._history.append(snap)
        self._check_diversity_emergence(snap)
        return snap

    def _check_diversity_emergence(self, snap: DiversitySnapshot):
        """检测多样性涌现。

        Holland 判据: 多样性 (Shannon) 突然增加 + 均匀度上升
        → 新生态位涌现 (旧主体未消失, 新主体加入)
        """
        if len(self._history) < 3:
            return

        recent = [h.shannon for h in self._history[-3:]]
        trend = recent[-1] - recent[-3]

        if trend > 0.5 and snap.evenness > self._history[-3].evenness:
            self._emergence_events.append({
                "type": "diversity_emergence",
                "shannon_gain": round(trend, 4),
                "new_richness": snap.richness,
                "evenness": snap.evenness,
            })

    @property
    def emergence_ratio(self) -> float:
        """涌现频率: 涌现事件数 / 观测数"""
        return (len(self._emergence_events) /
                max(len(self._history), 1))


class DiversityTracker:
    """多样性追踪器: 时间序列 + 涌现检测。

    Holland: "Tracking diversity over time reveals the
              adaptive dynamics of the system."
    """

    def __init__(self):
        self.index = DiversityIndex()
        self._timeseries: dict[str, list[int]] = defaultdict(list)

    def record(self, timeline: list[dict[str, int]]):
        """记录一个时间序列的多样性快照。

        Args:
            timeline: [{tag: count}, {tag: count}, ...]
        """
        for t, snapshot in enumerate(timeline):
            self.index.compute(snapshot)
            for tag, count in snapshot.items():
                self._timeseries[tag].append(count)

    def niche_emergence(self, min_frequency: int = 3
                        ) -> list[tuple[str, float]]:
        """检测新生态位: 近期出现且持续增长的标签。

        Holland: "New niches emerge when diverse interactions
                  create opportunities for specialisation."
        """
        new_niches: list[tuple[str, float]] = []
        for tag, series in self._timeseries.items():
            if len(series) < 3:
                continue
            # 前半段 = 0, 后半段 > 0 → 新生态位
            mid = len(series) // 2
            early = sum(series[:mid])
            late = sum(series[mid:])
            if early == 0 and late >= min_frequency:
                growth_rate = late / max(len(series[mid:]), 1)
                new_niches.append((tag, round(growth_rate, 4)))

        return sorted(new_niches, key=lambda x: x[1], reverse=True)
