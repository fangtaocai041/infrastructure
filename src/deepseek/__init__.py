"""
deepseek/__init__.py — DeepSeek 工程思想优化模块
==================================================
将 DeepSeek 的五个核心工程思想落地到 infrastructure 涌现引擎:

  P0 ① MoE 稀疏路由   → moe_router.py    (理论匹配 3-5x 加速)
  P0 ② 投机假说       → speculative.py    (假说质量 ↑30%)
  P1 ⑤ 多假说并行     → multi_hypothesis.py (发现率 ↑2x)
  P1 ④ MLA 图谱压缩   → mla_graph.py       (路由 10x 加速)
  P2 ③ GRPO 自进化    → grpo_evolution.py  (长期自适应)

Usage:
    from src.deepseek import (
        MoETheoryRouter, SpeculativeEngine,
        MultiHypothesisGenerator, MLAGraphCompressor,
        GRPOOptimizer, EmergenceBridge
    )
"""

from .moe_router import MoETheoryRouter, MoEGate, TheoryDomain
from .speculative import SpeculativeEngine, Hypothesis
from .multi_hypothesis import MultiHypothesisGenerator, HypothesisPath
from .mla_graph import MLAGraphCompressor, CompressedNode
from .grpo_evolution import GRPOOptimizer, GRPOStep, EmergenceBridge

__all__ = [
    "MoETheoryRouter", "MoEGate", "TheoryDomain",
    "SpeculativeEngine", "Hypothesis",
    "MultiHypothesisGenerator", "HypothesisPath",
    "MLAGraphCompressor", "CompressedNode",
    "GRPOOptimizer", "GRPOStep", "EmergenceBridge",
]
