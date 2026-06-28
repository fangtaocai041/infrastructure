"""
holland/__init__.py — Holland 涌现理论工程化实现
==================================================
将 John Holland 的七条涌现原理 (Emergence / Hidden Order) 转化为
可运行的 Python 工程原语，集成到 infrastructure 涌现检测管线。

七条原理:
  ① 聚集 (Aggregation):   简单主体 → 元主体的层次涌现
  ② 标签 (Tagging):       选择性交互的标识机制
  ③ 非线性 (Nonlinearity): 小扰动产生大效应
  ④ 流 (Flows):           资源/信息在网络中的乘数效应
  ⑤ 多样性 (Diversity):   异质主体维持系统适应力
  ⑥ 内部模型 (Internal):  预测模型的竞争与进化
  ⑦ 积木块 (Blocks):      复杂系统的可分解性与复用

Usage:
    from src.holland import CASCognitiveEngine

    cas = CASCognitiveEngine()
    result = cas.scan_for_emergence(data, species="Coilia nasus")
    # → HollandScore: aggregation=0.7, nonlinear=0.9, diversity=0.4, flows=0.6...
"""

from .aggregation import AggregationDetector, MetaAgent
from .tagging import TaggedGraph, TagDynamics
from .nonlinear import NonlinearProbe, SensitivityAnalyzer
from .flows import FlowNetwork, FlowAnalyzer
from .diversity import DiversityIndex, DiversityTracker
from .internal_models import InternalModel, ModelCompetition
from .blocks import BlockDecomposer, BuildingBlock
from .cas_engine import CASCognitiveEngine, HollandScore

__all__ = [
    # Aggregation
    "AggregationDetector", "MetaAgent",
    # Tagging
    "TaggedGraph", "TagDynamics",
    # Nonlinearity
    "NonlinearProbe", "SensitivityAnalyzer",
    # Flows
    "FlowNetwork", "FlowAnalyzer",
    # Diversity
    "DiversityIndex", "DiversityTracker",
    # Internal Models
    "InternalModel", "ModelCompetition",
    # Building Blocks
    "BlockDecomposer", "BuildingBlock",
    # Engine
    "CASCognitiveEngine", "HollandScore",
]
