"""
test_deepseek.py — DeepSeek 优化模块冒烟测试
==============================================
验证 5 个优化模块能正常初始化和运行。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest


class TestMoERouter:
    def test_router_init(self):
        from deepseek.moe_router import MoETheoryRouter
        router = MoETheoryRouter()
        assert len(router.domains) == 8

    def test_router_match(self):
        from deepseek.moe_router import MoETheoryRouter
        router = MoETheoryRouter()
        obs = {"biomass_slope": 0.95, "diversity_slope": 0.82,
               "predator_slope": -0.7, "prey_slope": 0.6,
               "catch_slope": -0.5, "habitat_slope": 0.4}
        matches = router.match(obs, top_k=2)
        assert len(matches) >= 1

    def test_router_speed_estimate(self):
        from deepseek.moe_router import MoETheoryRouter
        router = MoETheoryRouter()
        stats = router.speed_estimate({"biomass_slope": 0.1})
        assert stats["estimated_speedup"].endswith("x")
        assert stats["active_domains"] <= 2

    def test_router_feedback(self):
        from deepseek.moe_router import MoETheoryRouter
        router = MoETheoryRouter()
        old_w = router.domains["disturbance"].weight
        router.feedback("IDH", useful=True)
        assert router.domains["disturbance"].weight > old_w


class TestSpeculative:
    def test_engine_init(self):
        from deepseek.speculative import SpeculativeEngine
        se = SpeculativeEngine(n_drafts=3, top_k=2)
        assert se.n_drafts == 3

    def test_engine_run(self):
        from deepseek.speculative import SpeculativeEngine
        se = SpeculativeEngine(n_drafts=3, top_k=2, min_confidence=0.3)
        data = {"biomass_slope": 0.8, "diversity_slope": 0.3,
                "n_points": 8}
        results = se.run(data)
        assert isinstance(results, list)

    def test_quality_report(self):
        from deepseek.speculative import SpeculativeEngine
        se = SpeculativeEngine()
        se.run({"biomass_slope": 0.5, "n_points": 5})
        report = se.quality_report()
        assert "total_runs" in report


class TestMultiHypothesis:
    def test_generator(self):
        from deepseek.multi_hypothesis import MultiHypothesisGenerator
        mg = MultiHypothesisGenerator()
        results = mg.generate(
            species="TestSpecies",
            observations={"biomass_slope": 0.7, "diversity_slope": -0.3},
            available_theories=[
                {"name": "IDH", "domain": "ecology",
                 "theory": "中等干扰假说", "match_score": 0.8},
                {"name": "MSY", "domain": "fisheries",
                 "theory": "最大可持续产量", "match_score": 0.6},
            ],
        )
        assert len(results) >= 1

    def test_hypothesis_network(self):
        from deepseek.multi_hypothesis import MultiHypothesisGenerator
        mg = MultiHypothesisGenerator()
        mg.generate(
            species="Test",
            observations={"biomass_slope": 0.9},
            available_theories=[
                {"name": "T1", "domain": "A",
                 "theory": "理论1", "match_score": 0.9},
                {"name": "T2", "domain": "A",
                 "theory": "理论2", "match_score": 0.7},
            ],
        )
        network = mg.hypothesis_network()
        assert "nodes" in network
        assert "edges" in network


class TestMLAGraph:
    def test_compressor(self):
        from deepseek.mla_graph import MLAGraphCompressor
        compressor = MLAGraphCompressor(latent_dim=64)
        graph = {
            "nodes": [
                {"id": "n1", "domain": "ecology"},
                {"id": "n2", "domain": "fisheries"},
                {"id": "n3", "domain": "evolution"},
            ],
            "edges": [
                {"source": "n1", "target": "n2", "weight": 0.8},
                {"source": "n2", "target": "n3", "weight": 0.5},
            ],
        }
        n = compressor.compress_graph(graph)
        assert n == 3

    def test_route(self):
        from deepseek.mla_graph import MLAGraphCompressor
        compressor = MLAGraphCompressor(latent_dim=32)
        graph = {
            "nodes": [
                {"id": "disturbance_theory", "domain": "ecology"},
                {"id": "fishery_theory", "domain": "fisheries"},
                {"id": "evolution_theory", "domain": "evolution"},
            ],
            "edges": [
                {"source": "disturbance_theory",
                 "target": "fishery_theory", "weight": 1.0},
            ],
        }
        compressor.compress_graph(graph)
        results = compressor.route(["ecology"], top_k=2)
        assert len(results) >= 1

    def test_compression_stats(self):
        from deepseek.mla_graph import MLAGraphCompressor
        compressor = MLAGraphCompressor(latent_dim=32)
        graph = {
            "nodes": [{"id": f"n{i}", "domain": "test"} for i in range(5)],
            "edges": [],
        }
        compressor.compress_graph(graph)
        stats = compressor.compression_stats()
        assert stats["nodes"] == 5
        assert stats["latent_dim"] == 32


class TestGRPO:
    def test_grpo_step(self):
        from deepseek.grpo_evolution import GRPOOptimizer
        grpo = GRPOOptimizer(learning_rate=0.01)
        # 先填充 baseline buffer
        for i in range(5):
            grpo.step(f"warmup_{i}", reward=0.5, action="test")
        # 现在一个高奖励的步应该产生正优势
        step = grpo.step("test_hypothesis", reward=0.8, action="disturbance")
        assert step.advantage > 0  # 高于 baseline

    def test_grpo_convergence(self):
        from deepseek.grpo_evolution import GRPOOptimizer
        grpo = GRPOOptimizer()
        for i in range(20):
            grpo.step(f"h{i}", reward=0.7 + (i % 3) * 0.1, action="fisheries")
        conv = grpo.convergence_metric()
        assert 0.0 <= conv <= 1.0

    def test_emergence_bridge(self):
        from deepseek.grpo_evolution import EmergenceBridge
        bridge = EmergenceBridge()
        bridge.feed_holland_score(
            holland_index=0.7,
            active_dimensions=["aggregation", "nonlinear"],
            matched_theories=[
                {"active_domain": "disturbance"},
                {"active_domain": "predation"},
            ],
        )
        weights = bridge.get_domain_weights()
        assert len(weights) >= 2
