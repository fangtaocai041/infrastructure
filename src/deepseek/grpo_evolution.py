"""
deepseek/grpo_evolution.py — GRPO 自进化反馈环
================================================
DeepSeek 思想: GRPO (Group Relative Policy Optimization) —
RL 驱动的推理增强, 组内相对优势更新策略。

工程实现:
  - 假说质量 → 奖励信号 (论文引用/专家验证)
  - 组内相对优势 (baseline) → 策略梯度
  - MoE 门控权重自适应更新

Holland + DeepSeek 融合:
  Holland 的"内部模型竞争" + DeepSeek 的 GRPO 策略优化
  → 涌现引擎的自进化能力
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GRPOStep:
    """一次 GRPO 学习步。"""
    hypothesis: str
    reward: float                # -1 ~ +1
    group_baseline: float        # 组平均奖励
    advantage: float             # 相对优势
    action: str                  # 调整了哪个参数
    delta: float                 # 调整幅度


class GRPOOptimizer:
    """GRPO 策略优化器。

    GRPO 公式:
      advantage_i = reward_i - baseline (组内相对优势)
      Δθ = lr × advantage × ∇log π(a|s)

    简化为:
      weight += lr × advantage (正优势 → 增强, 负优势 → 削弱)
    """

    def __init__(self, learning_rate: float = 0.01,
                 baseline_window: int = 10):
        self.lr = learning_rate
        self.baseline_window = baseline_window
        self._reward_buffer: list[float] = []
        self._history: list[GRPOStep] = []
        self._weights: dict[str, float] = defaultdict(lambda: 0.5)

    def step(self, hypothesis: str, reward: float,
             action: str) -> GRPOStep:
        """执行一步 GRPO 更新。

        Args:
            hypothesis: 假说标识
            reward: 奖励信号 (-1=完全错误, 0=无关, +1=完全正确)
            action: 被调参数 (domain, theory, feature...)
        """
        self._reward_buffer.append(reward)
        if len(self._reward_buffer) > self.baseline_window * 2:
            self._reward_buffer = self._reward_buffer[-self.baseline_window:]

        # 组内 baseline: 最近 N 步的平均奖励
        recent = (self._reward_buffer[-self.baseline_window:]
                  if len(self._reward_buffer) >= self.baseline_window
                  else self._reward_buffer)
        baseline = sum(recent) / len(recent) if recent else 0.0

        # 相对优势
        advantage = reward - baseline

        # 策略更新
        delta = self.lr * advantage
        self._weights[action] = max(0.05, min(0.95,
            self._weights[action] + delta
        ))

        step = GRPOStep(
            hypothesis=hypothesis,
            reward=round(reward, 4),
            group_baseline=round(baseline, 4),
            advantage=round(advantage, 4),
            action=action,
            delta=round(delta, 4),
        )
        self._history.append(step)
        return step

    def get_weight(self, action: str) -> float:
        """获取行动的当前权重。"""
        return self._weights.get(action, 0.5)

    def top_actions(self, top_k: int = 5) -> list[tuple[str, float]]:
        """返回权重最高的行动。"""
        return sorted(self._weights.items(),
                      key=lambda x: x[1], reverse=True)[:top_k]

    def convergence_metric(self) -> float:
        """收敛度量: 最近 N 步的 advantage 方差。

        方差 → 0 表示策略趋于稳定。
        """
        if len(self._history) < 5:
            return 1.0
        recent = [s.advantage for s in self._history[-10:]]
        mean = sum(recent) / len(recent)
        var = sum((a - mean) ** 2 for a in recent) / len(recent)
        return round(max(0.0, 1.0 - math.sqrt(var)), 4)


class EmergenceBridge:
    """Holland ↔ Unified Emergence 桥接器。

    将 Holland 涌现评分反馈到 unified_emergence.py 的主管线,
    通过 GRPO 优化理论路由的权重。

    Usage:
        bridge = EmergenceBridge()
        bridge.feed_holland_score(holland_score, matching_theories)
        # GRPO 更新后, MoE 路由器下次会更准确
    """

    def __init__(self):
        self.grpo = GRPOOptimizer()
        self._holland_to_theory: dict[str, str] = {
            "aggregation": "disturbance",   # 聚集 → 干扰理论
            "nonlinear": "predation",       # 非线性 → 捕食理论
            "flows": "fisheries",           # 流 → 渔业理论
            "diversity": "evolution",       # 多样性 → 进化理论
            "blocks": "spatial",            # 积木块 → 空间理论
        }

    def feed_holland_score(self, holland_index: float,
                           active_dimensions: list[str],
                           matched_theories: list[dict]):
        """输入 Holland 涌现评分, GRPO 更新理论域权重。

        Args:
            holland_index: Holland 涌现指数 (0~1)
            active_dimensions: 激活的 Holland 维度
            matched_theories: 匹配的理论列表
        """
        # 每个激活的 Holland 维度映射到对应的理论域
        for dim in active_dimensions:
            domain = self._holland_to_theory.get(dim, dim)
            # 奖励 = Holland 指数 × 是否有匹配理论
            has_match = any(
                t.get("active_domain") == domain
                for t in matched_theories
            )
            reward = holland_index * (1.0 if has_match else -0.2)
            self.grpo.step(
                hypothesis=f"holland_{dim}",
                reward=reward,
                action=domain,
            )

    def get_domain_weights(self) -> dict[str, float]:
        """获取 GRPO 优化后的域权重。"""
        return {k: self.grpo.get_weight(v)
                for k, v in self._holland_to_theory.items()}

    def optimize_moe_router(self, router) -> int:
        """将 GRPO 优化后的权重同步到 MoE 路由器。

        Returns:
            更新的域数
        """
        updated = 0
        weights = self.get_domain_weights()
        for dname, domain in router.domains.items():
            if dname in weights:
                # GRPO 权重 × 原始权重 → 自适应调整
                domain.weight = weights[dname] * 2.0  # scale to [0.1, 1.9]
                updated += 1
        return updated
