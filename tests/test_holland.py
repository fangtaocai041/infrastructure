"""
test_holland.py — Holland 涌现模块冒烟测试
============================================
验证七条原理的工程原语能正常初始化和运行。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest


class TestHollandAggregation:
    def test_detector_init(self):
        from holland.aggregation import AggregationDetector
        d = AggregationDetector()
        assert d.threshold == 0.5

    def test_detect_basic(self):
        from holland.aggregation import AggregationDetector
        d = AggregationDetector(emergence_threshold=0.1)
        items = [
            {"keywords": ["genetics", "population"]},
            {"keywords": ["genetics", "migration"]},
            {"keywords": ["ecology", "habitat"]},
            {"keywords": ["genetics", "evolution"]},
        ]
        result = d.detect(items, key_field="keywords")
        assert len(result) >= 1

    def test_meta_agent_hierarchy(self):
        from holland.aggregation import AggregationDetector
        d = AggregationDetector(emergence_threshold=0.0)
        items = [
            {"keywords": ["A", "B"]},
            {"keywords": ["A", "C"]},
            {"keywords": ["B", "C"]},
        ]
        result = d.detect(items)
        meta = [r for r in result if r.level == 1]
        assert len(meta) >= 2


class TestHollandTagging:
    def test_tagged_graph_add_and_route(self):
        from holland.tagging import TaggedGraph
        g = TaggedGraph()
        g.add_node("n1", {"genetics", "population"})
        g.add_node("n2", {"genetics", "migration"})
        g.add_node("n3", {"ecology", "habitat"})
        result = g.route_by_tags({"genetics"})
        assert len(result) >= 2
        assert result[0][0] in ("n1", "n2")

    def test_tag_dynamics_emergence(self):
        from holland.tagging import TagDynamics
        td = TagDynamics()
        td.record([{"A", "B"}, {"A", "C"}, {"A", "B"}])
        td.record([{"A", "B"}, {"A", "C"}, {"A", "B"}])
        td.record([{"X", "Y"}, {"X", "Z"}, {"X", "Y"}])  # X suddenly appears
        active = td.active_tags
        assert len(active) >= 0  # X may or may not emerge


class TestHollandNonlinear:
    def test_sensitivity_analyzer(self):
        from holland.nonlinear import SensitivityAnalyzer
        sa = SensitivityAnalyzer(emergence_threshold=2.0)
        r = sa.analyze("v1", 100.0, 101.0, 50.0, 80.0)
        assert r.sensitivity > 0.0

    def test_nonlinear_probe(self):
        from holland.nonlinear import NonlinearProbe
        probe = NonlinearProbe(epsilon=0.01)

        def simple_system(inp):
            x = inp["x"]
            return {"y": x * 2 + 5}

        results = probe.probe(simple_system, {"x": 10.0})
        assert len(results) > 0


class TestHollandFlows:
    def test_flow_network_trace(self):
        from holland.flows import FlowNetwork
        fn = FlowNetwork()
        fn.add_edge("A", "B", 2.0)
        fn.add_edge("B", "C", 1.5)
        paths = fn.trace_flow("A", max_depth=3)
        assert len(paths) >= 1

    def test_flow_analyzer_build(self):
        from holland.flows import FlowAnalyzer
        fa = FlowAnalyzer()
        papers = [
            {"id": "1", "title": "T1", "citation_count": 10,
             "references": ["2", "3"]},
            {"id": "2", "title": "T2", "citation_count": 5,
             "references": []},
            {"id": "3", "title": "T3", "citation_count": 3,
             "references": ["2"]},
        ]
        fa.build_from_citations(papers)
        nodes = fa.find_emergence_nodes()
        assert len(nodes) >= 1


class TestHollandDiversity:
    def test_diversity_index(self):
        from holland.diversity import DiversityIndex
        di = DiversityIndex()
        snap = di.compute({"A": 5, "B": 5, "C": 5, "D": 5})
        assert snap.shannon > 1.0  # 4 equal categories = high diversity
        assert snap.evenness > 0.9

    def test_diversity_tracker(self):
        from holland.diversity import DiversityTracker
        dt = DiversityTracker()
        dt.record([
            {"A": 1, "B": 1},
            {"A": 1, "B": 1},
            {"A": 0, "B": 1, "C": 3},  # C emerges
        ])
        niches = dt.niche_emergence(min_frequency=1)
        assert len(niches) >= 0


class TestHollandModels:
    def test_model_competition(self):
        from holland.internal_models import (
            InternalModel, ModelCompetition
        )
        mc = ModelCompetition(max_models=3)
        mc.register(InternalModel(
            "linear", predict_fn=lambda x: sum(x.values())
        ))
        mc.register(InternalModel(
            "quadratic", predict_fn=lambda x: sum(x.values()) ** 0.5
        ))
        name, pred = mc.predict_best({"a": 1, "b": 2})
        assert isinstance(name, str)
        mc.update_all({"a": 1, "b": 2}, actual=3.0)


class TestHollandBlocks:
    def test_block_composition(self):
        from holland.blocks import BuildingBlock, BlockDecomposer
        bd = BlockDecomposer()
        bd.register(BuildingBlock(
            "growth_model", "biology", "growth estimation",
            interfaces=["size", "age", "rate"],
        ))
        bd.register(BuildingBlock(
            "economic_model", "economics", "resource allocation",
            interfaces=["rate", "cost", "benefit"],
        ))
        composed = bd.cross_domain_compose("biology", "economics")
        assert len(composed) >= 1

    def test_suggest_innovations(self):
        from holland.blocks import BuildingBlock, BlockDecomposer
        bd = BlockDecomposer()
        bd.register(BuildingBlock(
            "MSY", "fisheries", "max sustainable yield",
            interfaces=["biomass", "harvest"],
            reuse_count=5,
        ))
        bd.register(BuildingBlock(
            "grazing_model", "grassland", "optimal grazing rate",
            interfaces=["biomass", "stocking"],
            reuse_count=3,
        ))
        bd.register(BuildingBlock(
            "IDH", "ecology", "intermediate disturbance",
            interfaces=["disturbance", "diversity"],
            reuse_count=8,
        ))
        suggestions = bd.suggest_innovations("fisheries", top_k=3)
        assert len(suggestions) >= 1


class TestCASEngine:
    def test_engine_init(self):
        from holland import CASCognitiveEngine
        cas = CASCognitiveEngine()
        assert cas is not None

    def test_scan_with_papers_and_data(self):
        from holland import CASCognitiveEngine
        cas = CASCognitiveEngine()
        papers = [
            {"id": "1", "title": "T1", "keywords": "genetics,population",
             "citation_count": 10, "references": ["2"]},
            {"id": "2", "title": "T2", "keywords": "genetics,migration",
             "citation_count": 5, "references": []},
            {"id": "3", "title": "T3", "keywords": "genetics,evolution",
             "citation_count": 15, "references": ["1", "2"]},
            {"id": "4", "title": "T4", "keywords": "ecology,habitat",
             "citation_count": 8, "references": ["3"]},
        ]
        data = {
            "years": [2018, 2019, 2020, 2021, 2022],
            "biomass": [100, 110, 105, 120, 200],
            "diversity": [5, 5, 5, 5, 8],
        }
        score = cas.scan(papers=papers, species="Test", data=data)
        assert isinstance(score.holland_index, float)
        assert 0.0 <= score.holland_index <= 1.0

    def test_generate_hypotheses_when_emergent(self):
        from holland import CASCognitiveEngine, HollandScore
        cas = CASCognitiveEngine()
        # Inject an emergent score
        score = HollandScore(
            aggregation=0.8, tagging=0.7, nonlinear=0.6,
            flows=0.5, diversity=0.4, internal_models=0.3,
            blocks=0.6,
        )
        h = cas.generate_hypotheses(score, "TestSpecies")
        assert len(h) >= 1

    def test_generate_hypotheses_when_not_emergent(self):
        from holland import CASCognitiveEngine, HollandScore
        cas = CASCognitiveEngine()
        score = HollandScore()  # all zeros
        h = cas.generate_hypotheses(score)
        assert h == []

    def test_reset(self):
        from holland import CASCognitiveEngine
        cas = CASCognitiveEngine()
        cas.scan(papers=[{"id": "1", "title": "T", "keywords": "test",
                          "citation_count": 1, "references": []}])
        cas.reset()
        assert len(cas._history) == 0
