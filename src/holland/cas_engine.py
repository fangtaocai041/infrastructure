"""
holland/cas_engine.py — CAS 统一涌现引擎
==========================================
将 Holland 七条原理汇聚为统一的涌现检测流程。

Holland: "Emergence is the product of interactions among
the seven basics — aggregation, tagging, nonlinearity, flows,
diversity, internal models, and building blocks — acting
simultaneously in a complex adaptive system."

工程架构:
  输入数据 → 七条原理并行检测 → HollandScore 加权融合 → 涌现判定

Holland 涌现判据:
  当至少 3 个维度的得分同时超过阈值时, 认为系统发生了涌现。
  "Emergence is a multi-dimensional phenomenon — no single
   metric captures it, but convergent signals across dimensions
   provide strong evidence."
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from .aggregation import AggregationDetector, MetaAgent
from .tagging import TaggedGraph, TagDynamics
from .nonlinear import NonlinearProbe, SensitivityAnalyzer
from .flows import FlowNetwork, FlowAnalyzer
from .diversity import DiversityIndex, DiversityTracker
from .internal_models import InternalModel, ModelCompetition
from .blocks import BlockDecomposer, BuildingBlock


@dataclass
class HollandScore:
    """Holland 涌现综合评分。

    Holland: "No single metric captures emergence — but
              the convergence of multiple metrics is itself
              a signal of emergent order."
    """
    aggregation: float = 0.0     # ① 聚集度
    tagging: float = 0.0         # ② 标签涌现度
    nonlinear: float = 0.0       # ③ 非线性敏感度
    flows: float = 0.0           # ④ 流涌现度
    diversity: float = 0.0       # ⑤ 多样性变化
    internal_models: float = 0.0 # ⑥ 模型竞争涌现
    blocks: float = 0.0          # ⑦ 积木块创新度

    # 复合指标
    holland_index: float = 0.0   # Holland 涌现指数 (0~1)
    dimensions_active: int = 0   # 有多少维度超过阈值
    is_emergent: bool = False    # 是否判定为涌现

    def __post_init__(self):
        self._compute_composite()

    def _compute_composite(self):
        scores = [
            self.aggregation, self.tagging, self.nonlinear,
            self.flows, self.diversity, self.internal_models, self.blocks,
        ]
        self.holland_index = round(
            sum(scores) / len(scores), 4
        )
        self.dimensions_active = sum(
            1 for s in scores if s > 0.3
        )
        # Holland 涌现判据: ≥3 维度同时超过 0.3
        self.is_emergent = self.dimensions_active >= 3

    def describe(self) -> str:
        dims = []
        for name, val in [
            ("聚集", self.aggregation), ("标签", self.tagging),
            ("非线性", self.nonlinear), ("流", self.flows),
            ("多样性", self.diversity), ("内部模型", self.internal_models),
            ("积木块", self.blocks),
        ]:
            bar = "#" * int(val * 10) + "·" * (10 - int(val * 10))
            dims.append(f"  {name}: {bar} {val:.2f}")
        return (
            f"HollandScore(index={self.holland_index:.2f}, "
            f"emergent={'YES' if self.is_emergent else 'no'}, "
            f"active_dims={self.dimensions_active})\n"
            + "\n".join(dims)
        )


class CASCognitiveEngine:
    """CAS 认知涌现引擎 — Holland 七条原理的统一执行器。

    Usage:
        cas = CASCognitiveEngine()
        score = cas.scan(papers, species="Coilia nasus", data={"biomass": [...]})
        print(score.describe())
        if score.is_emergent:
            hypotheses = cas.generate_hypotheses(score)
    """

    def __init__(self, emergence_threshold: float = 0.3):
        self.threshold = emergence_threshold
        # 七条原理的检测器
        self.aggregation = AggregationDetector()
        self.tag_dynamics = TagDynamics()
        self.tag_graph = TaggedGraph()
        self.nonlinear_probe = NonlinearProbe()
        self.flow_analyzer = FlowAnalyzer()
        self.diversity_tracker = DiversityTracker()
        self.model_competition = ModelCompetition()
        self.block_decomposer = BlockDecomposer()

        self._history: list[HollandScore] = []

    def scan(self,
             papers: list[dict] | None = None,
             species: str = "",
             data: dict[str, list] | None = None,
             ) -> HollandScore:
        """执行七条原理的并行检测。

        Args:
            papers: 论文列表 (文献搜索结果)
            species: 目标物种
            data: 时间序列数据 {"years": [...], "metric": [...]}

        Returns:
            HollandScore with all 7 dimensions scored
        """
        papers = papers or []
        score = HollandScore()

        # ① 聚集: 检测论文如何在主题上聚集
        if papers:
            meta_agents = self.aggregation.detect(papers, key_field="keywords")
            score.aggregation = min(1.0, sum(
                a.emergence_score for a in meta_agents
            ) / max(len(meta_agents), 1))

        # ② 标签: 标签涌现 + 选择性路由
        if papers:
            tags_per_item = [
                set(p.get("keywords", "").lower().split(","))
                for p in papers
            ]
            self.tag_dynamics.record(tags_per_item)
            emerged = self.tag_dynamics.active_tags
            score.tagging = min(1.0, sum(
                s for _, s in emerged[:5]
            ) / 5.0)

        # ③ 非线性: 对数据施加 ε 扰动, 测量灵敏度
        if data:
            def system_fn(inp: dict) -> dict:
                return {k: sum(v) / max(len(v), 1) for k, v in inp.items()
                        if isinstance(v, list)}

            results = self.nonlinear_probe.probe(
                system_fn,
                {k: sum(v) / max(len(v), 1)
                 for k, v in data.items() if isinstance(v, list)}
            )
            emergent_vars = self.nonlinear_probe.analyzer.emergent_variables()
            score.nonlinear = min(1.0, len(emergent_vars) / max(
                len(results), 1
            ) * 3)

        # ④ 流: 构建引用网络, 检测流涌现
        if papers:
            self.flow_analyzer.build_from_citations(papers)
            emergence_nodes = self.flow_analyzer.find_emergence_nodes()
            score.flows = min(1.0, sum(
                s for _, s in emergence_nodes[:5]
            ) / 5.0)

        # ⑤ 多样性: 标签分布均匀度
        if papers:
            keyword_counts: dict[str, int] = {}
            for p in papers:
                for kw in p.get("keywords", "").lower().split(","):
                    kw = kw.strip()
                    if kw:
                        keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
            snap = self.diversity_tracker.index.compute(keyword_counts)
            score.diversity = snap.evenness

        # ⑥ 内部模型: 如果有预测函数竞争
        if self.model_competition.models:
            emergences = self.model_competition.check_cognitive_emergence()
            score.internal_models = min(1.0, len(emergences) * 0.33)

        # ⑦ 积木块: 跨域重组创新检测
        if papers and len(self.block_decomposer.blocks) > 0:
            innovations = self.block_decomposer.suggest_innovations(
                "fisheries", top_k=5
            )
            score.blocks = min(1.0, sum(
                s for _, s in innovations
            ) / 5.0)

        self._history.append(score)
        return score

    def generate_hypotheses(self, score: HollandScore,
                            species: str = "") -> list[str]:
        """基于 Holland 涌现评分生成假说。

        Holland: "When emergence is detected, the system
                  should propose testable hypotheses about
                  the underlying generating mechanisms."
        """
        if not score.is_emergent:
            return []

        hypotheses = []

        if score.aggregation > 0.5:
            hypotheses.append(
                f"[聚集假说] {species}研究正在从分散方向向核心主题聚集, "
                "暗示一个新兴研究范式的形成"
            )
        if score.tagging > 0.4:
            hypotheses.append(
                f"[标签假说] 新术语/标签出现, 标记了{species}研究中 "
                "以往未认识到的维度, 可能定义了新研究方向"
            )
        if score.nonlinear > 0.4:
            hypotheses.append(
                f"[非线性假说] {species}生态系统中检测到非线性敏感变量, "
                "小扰动可能产生大效应 — 建议进行敏感度实验验证"
            )
        if score.flows > 0.4:
            hypotheses.append(
                f"[流假说] 知识在{species}引用网络中产生了乘数效应, "
                "某篇/某几篇论文成为关键放大节点 — 建议追踪其影响路径"
            )
        if score.diversity > 0.5:
            hypotheses.append(
                f"[多样性假说] {species}研究视角正在多样化, "
                "新方法/新范式的引入可能开辟新的研究生态位"
            )
        if score.internal_models > 0.3:
            hypotheses.append(
                f"[内部模型假说] 竞争中的预测模型出现了优胜者, "
                "该模型对{species}的行为/生态有更强的解释力"
            )
        if score.blocks > 0.3:
            hypotheses.append(
                f"[积木块假说] 来自其他域的积木块可重组应用到{species}研究, "
                "跨学科移植是创新的主要来源 — 建议探索跨域组合"
            )

        return hypotheses

    def reset(self):
        """重置所有检测器状态。"""
        self.aggregation = AggregationDetector()
        self.tag_dynamics = TagDynamics()
        self.nonlinear_probe = NonlinearProbe()
        self.flow_analyzer = FlowAnalyzer()
        self.diversity_tracker = DiversityTracker()
        self.model_competition = ModelCompetition()
        self.block_decomposer = BlockDecomposer()
        self._history.clear()
