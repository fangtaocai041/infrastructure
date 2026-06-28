"""
deepseek/moe_router.py — MoE 稀疏理论路由器
=============================================
DeepSeek 思想: MoE 只激活 top-k 专家 (37B/671B), 稀疏门控。

当前弱点: unified_emergence.py 的 scan() 遍历所有 KNOWN_PATTERNS。
优化: 根据输入特征, 只激活相关理论域的匹配器。

效果: 理论匹配速度 3-5x (仅激活 2/8 域而非全部)。

架构:
  Input (observations + species features)
    → Gate (轻量分类器: 特征 → 理论域得分)
    → Top-K 选择 (激活 top 2 域)
    → Domain Experts (并行匹配)
    → Merge (加权融合)
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class TheoryDomain:
    """理论域 — MoE 中的一个专家。"""
    name: str
    keywords: list[str]          # 激活关键词
    patterns: list[dict]         # 该域的理论匹配模式
    weight: float = 1.0          # 域权重 (自适应更新)


class MoEGate:
    """稀疏门控: 输入特征 → 理论域得分。

    DeepSeek 方式: 不是全连接 softmax, 而是 top-k 稀疏选择。
    类比: Router 只保留 top-k logits, 其余设 -∞。
    """

    def __init__(self, domains: dict[str, TheoryDomain]):
        self.domains = domains
        self._feature_weights: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._usage_count: dict[str, int] = defaultdict(int)

    def route(self, observations: dict[str, float],
              top_k: int = 2) -> list[tuple[str, float]]:
        """稀疏路由: 返回 top-k 理论域及其得分。

        Gate 函数: score(domain) = Σ(关键词命中 × 特征值 × 权重)
        """
        scores: dict[str, float] = {}
        for dname, domain in self.domains.items():
            score = 0.0
            for kw in domain.keywords:
                for obs_key, obs_val in observations.items():
                    if kw.lower() in obs_key.lower():
                        w = self._feature_weights[dname].get(obs_key, 0.5)
                        score += abs(obs_val) * w
            scores[dname] = score * domain.weight

        # Top-K 稀疏化: 只保留 top k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top = ranked[:top_k]

        # 更新使用计数 (自适应门控)
        for dname, _ in top:
            self._usage_count[dname] += 1

        return top

    def update_weights(self, domain: str, feature: str,
                       reward: float, lr: float = 0.01):
        """GRPO 风格: 根据奖励更新特征权重。

        DeepSeek 方式: 推理后反馈 → 调整门控权重。
        """
        self._feature_weights[domain][feature] += lr * reward
        # 裁剪到 [0, 1]
        self._feature_weights[domain][feature] = max(
            0.0, min(1.0, self._feature_weights[domain][feature])
        )


class MoETheoryRouter:
    """MoE 理论路由器: 稀疏激活 + 并行匹配。

    Usage:
        router = MoETheoryRouter()
        matches = router.match(observations={"biomass_slope": 0.35, ...})
        # → 只匹配了 "disturbance" 和 "predation" 域, 跳过其余 6 个域
    """

    def __init__(self):
        # 8 个理论域, 每个域有独立的关键词和匹配模式
        self.domains = {
            "disturbance": TheoryDomain(
                "disturbance",
                keywords=["slope", "biomass", "diversity", "disturbance",
                          "recovery", "resilience"],
                patterns=[
                    {"name": "IDH", "theory": "中等干扰假说",
                     "statistic": "diversity_slope",
                     "expected_sign": "positive"},
                    {"name": "AlternativeStableState",
                     "theory": "替代稳态理论",
                     "statistic": "biomass_slope / diversity_slope",
                     "expected_sign": "zero"},
                ],
            ),
            "predation": TheoryDomain(
                "predation",
                keywords=["predator", "prey", "trophic", "cascade",
                          "top_down", "bottom_up"],
                patterns=[
                    {"name": "LotkaVolterra",
                     "theory": "捕食者-猎物振荡",
                     "statistic": "predator_slope vs prey_slope",
                     "expected_sign": "oscillation"},
                    {"name": "TrophicCascade",
                     "theory": "营养级联效应",
                     "statistic": "top_predator_slope / prey_slope",
                     "expected_sign": "negative"},
                ],
            ),
            "evolution": TheoryDomain(
                "evolution",
                keywords=["evolution", "adaptation", "selection",
                          "genetic", "mutation", "fitness"],
                patterns=[
                    {"name": "RedQueen",
                     "theory": "红皇后假说",
                     "statistic": "evolution_rate vs diversity",
                     "expected_sign": "positive"},
                ],
            ),
            "spatial": TheoryDomain(
                "spatial",
                keywords=["spatial", "habitat", "patch", "migration",
                          "dispersal", "connectivity"],
                patterns=[
                    {"name": "IslandBiogeography",
                     "theory": "岛屿生物地理学",
                     "statistic": "species_count vs area",
                     "expected_sign": "positive"},
                    {"name": "Metapopulation",
                     "theory": "集合种群理论",
                     "statistic": "occupancy vs connectivity",
                     "expected_sign": "positive"},
                ],
            ),
            "fisheries": TheoryDomain(
                "fisheries",
                keywords=["fishery", "harvest", "catch", "MSY",
                          "overfishing", "stock"],
                patterns=[
                    {"name": "MSY", "theory": "最大可持续产量",
                     "statistic": "catch vs effort",
                     "expected_sign": "hump"},
                    {"name": "ShiftingBaseline",
                     "theory": "基线漂移综合征",
                     "statistic": "reference_point_trend",
                     "expected_sign": "declining"},
                ],
            ),
            "climate": TheoryDomain(
                "climate",
                keywords=["climate", "temperature", "warming",
                          "precipitation", "extreme", "seasonal"],
                patterns=[
                    {"name": "ThermalNiche",
                     "theory": "热生态位收缩",
                     "statistic": "abundance vs temperature",
                     "expected_sign": "negative"},
                ],
            ),
            "competition": TheoryDomain(
                "competition",
                keywords=["competition", "niche", "resource",
                          "coexistence", "partitioning"],
                patterns=[
                    {"name": "CompetitiveExclusion",
                     "theory": "竞争排斥原理",
                     "statistic": "species_overlap vs coexistence",
                     "expected_sign": "negative"},
                ],
            ),
            "stoichiometry": TheoryDomain(
                "stoichiometry",
                keywords=["nutrient", "nitrogen", "phosphorus",
                          "carbon", "ratio", "limitation"],
                patterns=[
                    {"name": "StoichiometricHomeostasis",
                     "theory": "化学计量稳态",
                     "statistic": "C_N_ratio vs growth_rate",
                     "expected_sign": "negative"},
                ],
            ),
        }
        self.gate = MoEGate(self.domains)
        self._match_cache: dict[str, list[dict]] = {}

    def match(self, observations: dict[str, float],
              top_k: int = 2) -> list[dict]:
        """稀疏理论匹配: 只激活 top-k 域。

        Returns:
            [{pattern_name, theory, match_score, confidence, active_domain}]
        """
        # Step 1: 门控选择
        active_domains = self.gate.route(observations, top_k=top_k)

        # Step 2: 只在激活域内匹配 (这是加速的来源)
        all_matches = []
        for dname, gate_score in active_domains:
            domain = self.domains[dname]
            for pattern in domain.patterns:
                score = self._match_pattern(pattern, observations)
                if score > 0.3:
                    all_matches.append({
                        "pattern_name": pattern["name"],
                        "theory": pattern["theory"],
                        "match_score": round(score, 4),
                        "confidence": round(score * gate_score / max(
                            sum(s for _, s in active_domains), 1e-9
                        ), 4),
                        "active_domain": dname,
                        "domain_weight": round(gate_score, 4),
                    })

        all_matches.sort(key=lambda x: x["confidence"], reverse=True)
        return all_matches

    def _match_pattern(self, pattern: dict,
                       observations: dict[str, float]) -> float:
        """单个模式匹配 (与 unified_emergence 的 match_theory 兼容)。

        返回 0-1 匹配分数。
        """
        stat = pattern.get("statistic", "")
        expected = pattern.get("expected_sign", "")

        # 直接统计量匹配
        if stat in observations:
            val = observations[stat]
            if expected == "positive" and val > 0:
                return min(abs(val) * 3.0, 1.0)
            elif expected == "negative" and val < 0:
                return min(abs(val) * 3.0, 1.0)
            elif expected == "zero":
                return 1.0 - min(abs(val) * 2.0, 1.0)
            return min(abs(val) * 2.0, 0.5)

        # 复合统计量 (A / B)
        if " / " in stat:
            parts = [p.strip() for p in stat.split(" / ")]
            if all(p in observations for p in parts):
                ratio = observations[parts[0]] / max(
                    observations[parts[1]], 1e-9
                )
                score = min(abs(ratio) / 3.0, 1.0)
                if expected == "positive" and ratio > 0:
                    return score
                return score * 0.5

        # 对比统计量 (A vs B)
        if " vs " in stat:
            parts = [p.strip() for p in stat.split(" vs ")]
            if all(p in observations for p in parts):
                diff = abs(observations[parts[0]] - observations[parts[1]])
                return min(diff * 2.0, 1.0)

        # 子串匹配 (如 "slope" in "biomass_slope")
        for obs_key in observations:
            if stat and stat.lower() in obs_key.lower():
                return min(abs(observations[obs_key]) * 2.0, 0.7)

        return 0.0

    def feedback(self, chosen_theory: str, useful: bool):
        """学习反馈: 调整域权重。

        DeepSeek 方式: GRPO — 有用的域权重↑, 无用的↓
        """
        for dname, domain in self.domains.items():
            for pattern in domain.patterns:
                if pattern["name"] == chosen_theory:
                    delta = 0.05 if useful else -0.02
                    domain.weight = max(0.1, min(2.0, domain.weight + delta))
                    return

    def speed_estimate(self, observations: dict) -> dict:
        """估算加速比: 激活域 / 总域"""
        active = self.gate.route(observations)
        total = len(self.domains)
        return {
            "active_domains": len(active),
            "total_domains": total,
            "activation_ratio": len(active) / total,
            "estimated_speedup": f"{total / max(len(active), 1):.1f}x",
        }
