"""
test_unified_emergence.py — 统一涌现检测引擎测试 (30项)

涵盖:
  - 基础类型 (EmergenceType, DimensionalLevel, dataclasses)
  - EmergenceMonitor 实时监控 (含 Welford)
  - DimensionalEmergenceMonitor 维度演进
  - EmergenceEngine 三层分析 (异常/突变/理论)
  - CUSUM + PELT 突变点检测
  - p-value 字段
  - 向后兼容
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from unified_emergence import (
    EmergenceType,
    DimensionalLevel,
    EmergenceSignal,
    DetectionResult,
    MetricTracker,
    EmergenceMonitor,
    DimensionalEmergenceMonitor,
    EmergenceEngine,
    KNOWN_PATTERNS,
    emerge_domains,
    record_search_result,
    _deduplicate_changes,
)


# ═══════════════════════════════════════════════════════════════
# Part 1: 基础类型
# ═══════════════════════════════════════════════════════════════

class TestEmergenceType:
    def test_emergence_type_values(self):
        assert EmergenceType.BENEFICIAL.value == "beneficial"
        assert EmergenceType.NEUTRAL.value == "neutral"
        assert EmergenceType.HARMFUL.value == "harmful"
        assert EmergenceType.PHASE_TRANSITION.value == "phase_transition"
        assert EmergenceType.ANOMALY.value == "anomaly"

    def test_dimensional_level_values(self):
        assert DimensionalLevel.D0.value == 0
        assert DimensionalLevel.D1.value == 1
        assert DimensionalLevel.D2.value == 2
        assert DimensionalLevel.D3.value == 3

    def test_emergence_signal_creation(self):
        sig = EmergenceSignal(
            id="EMG-0001",
            timestamp=1000.0,
            emergence_type=EmergenceType.BENEFICIAL,
            dimensional_level=DimensionalLevel.D1,
            sources=["src_a"],
            metrics={"recall": 0.9},
            deviation_sigma=3.5,
            description="test",
            confidence=0.8,
        )
        assert sig.id == "EMG-0001"
        assert not sig.resolved

    def test_detection_result_creation(self):
        dr = DetectionResult(
            detection_type="theory_match",
            species="Coilia nasus",
            description="非对称恢复",
            confidence=0.95,
            evidence={"slope": 2.5},
        )
        assert dr.detection_type == "theory_match"
        assert "isoformat" in dr.detected_at or len(dr.detected_at) > 10


# ═══════════════════════════════════════════════════════════════
# Part 2: MetricTracker (Welford)
# ═══════════════════════════════════════════════════════════════

class TestMetricTracker:
    def test_welford_basic(self):
        t = MetricTracker(name="test")
        for v in [10, 11, 9, 10, 12]:
            t.record(float(v))
        assert t.n == 5
        assert abs(t.mean - 10.4) < 0.1
        assert t.std > 0

    def test_welford_single_value(self):
        t = MetricTracker(name="test")
        t.record(42.0)
        assert t.n == 1
        assert t.mean == 42.0
        assert t.std == 1.0  # fallback

    def test_welford_large_sequence(self):
        """验证 Welford 无溢出 (原 _sum/_sum_sq bug 修复)。"""
        t = MetricTracker(name="test")
        for v in range(1000):
            t.record(float(v))
        assert t.n == 1000
        assert abs(t.mean - 499.5) < 1.0

    def test_deviation_sigma(self):
        t = MetricTracker(name="test")
        for v in [10, 10, 10, 10, 10]:
            t.record(float(v))
        deviation = t.deviation_sigma(20.0)
        assert deviation > 5.0  # 明显偏离

    def test_stats_output(self):
        t = MetricTracker(name="test")
        t.record(1.0)
        t.record(3.0)
        s = t.stats()
        assert s["name"] == "test"
        assert s["n"] == 2
        assert "mean" in s
        assert "std" in s
        assert "sigma" in s

    def test_deduplicate_changes(self):
        changes = [
            {"year": 2020, "magnitude": 0.5},
            {"year": 2021, "magnitude": 1.2},
            {"year": 2022, "magnitude": 0.3},
        ]
        deduped = _deduplicate_changes(changes, min_segment_length=3)
        assert len(deduped) >= 1  # 2020 和 2021 相邻, 保留效应量大的


# ═══════════════════════════════════════════════════════════════
# Part 3: EmergenceMonitor
# ═══════════════════════════════════════════════════════════════

class TestEmergenceMonitor:
    def test_monitor_init(self):
        mon = EmergenceMonitor()
        assert mon.threshold_sigma == 3.0
        assert mon.min_sources == 3
        assert len(mon.trackers) == 0

    def test_monitor_record(self):
        mon = EmergenceMonitor()
        mon.record("recall", 0.85, DimensionalLevel.D1)
        assert len(mon.trackers) == 1
        key = "D1:recall"
        assert key in mon.trackers
        assert mon.trackers[key].n == 1

    def test_no_signal_under_threshold(self):
        mon = EmergenceMonitor(emergence_threshold_sigma=3.0)
        # 记录 10 个接近的值 → 不会触发
        for _ in range(10):
            mon.record("recall", 0.5, DimensionalLevel.D1)
        signals = mon.check_emergence()
        assert len(signals) == 0

    def test_detects_anomaly(self):
        mon = EmergenceMonitor(emergence_threshold_sigma=1.5, min_sources=2)
        # 基线 — 3 个指标的稳定值
        for _ in range(8):
            mon.record("recall", 0.5, DimensionalLevel.D1)
            mon.record("precision", 0.5, DimensionalLevel.D1)
            mon.record("success_rate", 0.5, DimensionalLevel.D1)
        # 大幅偏离 (0.5→1.0 对于只有 8 个样本的集合 > 3σ)
        mon.record("recall", 1.0, DimensionalLevel.D1)
        mon.record("precision", 1.0, DimensionalLevel.D1)
        mon.record("success_rate", 1.0, DimensionalLevel.D1)
        signals = mon.check_emergence()
        assert len(signals) >= 1

    def test_health_report(self):
        mon = EmergenceMonitor()
        report = mon.health_report()
        assert "tracked_metrics" in report
        assert "total_signals" in report

    def test_beneficial_configurable(self):
        """验证 _is_beneficial 可通过 constructor 配置。"""
        mon = EmergenceMonitor(
            beneficial_metrics={"biomass", "catch_rate"},
            harmful_metrics={"pollution"},
        )
        assert "biomass" in mon._beneficial_metrics
        assert "pollution" in mon._harmful_metrics


# ═══════════════════════════════════════════════════════════════
# Part 4: DimensionalEmergenceMonitor
# ═══════════════════════════════════════════════════════════════

class TestDimensionalMonitor:
    def test_transition(self):
        mon = DimensionalEmergenceMonitor()
        mon.track_dimension_transition(
            DimensionalLevel.D1, DimensionalLevel.D2, 0.5
        )
        key = "D2:transition_cost_D1_to_D2"
        assert key in mon.trackers

    def test_no_false_phase_transition(self):
        """不应在没有足够 D2 信号时误报 D₃ 相变。"""
        mon = DimensionalEmergenceMonitor(min_sources=3)
        mon.track_dimension_transition(DimensionalLevel.D2, DimensionalLevel.D3, 0.5)
        # 只有 1 个信号, 不足 min_sources=3
        signal = mon.check_dimensional_emergence()
        assert signal is None


# ═══════════════════════════════════════════════════════════════
# Part 5: EmergenceEngine — 异常检测
# ═══════════════════════════════════════════════════════════════

class TestEngineAnomalyDetection:
    def test_zscore(self):
        data = [10, 11, 9, 10, 12, 8, 10, 200]
        dates = list(range(2018, 2026))
        result = EmergenceEngine.detect_anomalies(data, dates, method="zscore")
        anomalies = [r for r in result if r["is_anomaly"]]
        assert len(anomalies) >= 1
        assert "p_value" in result[0]

    def test_iqr(self):
        data = [10, 11, 9, 10, 12, 8, 10, 200]
        dates = list(range(2018, 2026))
        result = EmergenceEngine.detect_anomalies(data, dates, method="iqr")
        anomalies = [r for r in result if r["is_anomaly"]]
        assert len(anomalies) >= 1
        assert "p_value" in result[0]

    def test_zscore_no_false_positive(self):
        """平稳数据不应误报异常。"""
        data = [10, 11, 9, 10, 12, 11, 10, 9, 10, 11]
        dates = list(range(2015, 2025))
        result = EmergenceEngine.detect_anomalies(data, dates, method="zscore")
        anomalies = [r for r in result if r["is_anomaly"]]
        assert len(anomalies) == 0


# ═══════════════════════════════════════════════════════════════
# Part 6: EmergenceEngine — 突变点检测
# ═══════════════════════════════════════════════════════════════

class TestEngineChangePoint:
    def test_cusum_detects_step_change(self):
        data = [1, 1, 1, 1, 1, 100, 100, 100, 100, 100]
        dates = list(range(2015, 2025))
        result = EmergenceEngine.detect_change_points(
            data, dates, method="cusum", cusum_threshold=2.0
        )
        assert len(result) >= 1
        assert any(r["change_type"] == "up" for r in result)

    def test_cusum_no_false_on_flat(self):
        data = [10] * 10
        dates = list(range(2015, 2025))
        result = EmergenceEngine.detect_change_points(data, dates, method="cusum")
        assert len(result) == 0

    def test_pelt_detects_change(self):
        data = [5, 5, 5, 5, 5, 20, 20, 20, 20, 20]
        dates = list(range(2015, 2025))
        result = EmergenceEngine.detect_change_points(data, dates, method="pelt")
        assert len(result) >= 1

    def test_all_methods_return_method_field(self):
        data = [1, 1, 1, 1, 1, 10, 10, 10]
        dates = list(range(2015, 2023))
        for method in ["cusum", "pelt", "sliding", "diff"]:
            result = EmergenceEngine.detect_change_points(data, dates, method=method)
            if result:
                assert "method" in result[0]
                assert result[0]["method"] == method

    def test_cusum_detects_gradual_rise(self):
        """CUSUM 应能检测渐进上升 (滑动窗口可能漏掉)。"""
        data = [5, 5, 6, 7, 8, 10, 13, 17, 22, 28]
        dates = list(range(2015, 2025))
        result = EmergenceEngine.detect_change_points(
            data, dates, method="cusum", cusum_threshold=2.0
        )
        assert len(result) >= 1


# ═══════════════════════════════════════════════════════════════
# Part 7: EmergenceEngine — 理论匹配
# ═══════════════════════════════════════════════════════════════

class TestEngineTheoryMatch:
    def test_theory_match_with_compound(self):
        """验证复合统计量自动构造 + 理论匹配 (核心修复)。"""
        engine = EmergenceEngine()
        data = {
            "years": [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
            "body_size": [100, 95, 88, 105, 130, 175, 210, 260],
            "diversity": [35, 34, 33, 32, 33, 34, 35, 36],
        }
        results = engine.scan(data=data, species="test")
        tm = [r for r in results if r["detection_type"] == "theory_match"]
        assert len(tm) >= 1
        assert any("非对称恢复" in r.get("pattern_name", "") for r in tm)

    def test_theory_no_match(self):
        """随机波动数据不应误报理论匹配。"""
        engine = EmergenceEngine()
        data = {
            "years": [2018, 2019, 2020, 2021, 2022],
            "random_metric": [3, 4, 3, 4, 3],
        }
        results = engine.scan(data=data, species="test")
        tm = [r for r in results if r["detection_type"] == "theory_match"]
        assert len(tm) == 0


# ═══════════════════════════════════════════════════════════════
# Part 8: EmergenceEngine — 综合扫描
# ═══════════════════════════════════════════════════════════════

class TestEngineScan:
    def test_scan_with_real_data(self):
        """验证 scan() 返回结构完整。"""
        engine = EmergenceEngine()
        data = {
            "years": [2018, 2019, 2020, 2021, 2022],
            "biomass": [100, 120, 90, 130, 110],
        }
        results = engine.scan(data=data, species="Coilia nasus")
        assert isinstance(results, list)
        assert len(results) >= 0
        for r in results:
            assert "detection_type" in r

    def test_scan_without_data(self):
        results = EmergenceEngine().scan()
        assert len(results) == 1
        assert results[0]["detection_type"] == "status"


# ═══════════════════════════════════════════════════════════════
# Part 9: KNOWN_PATTERNS
# ═══════════════════════════════════════════════════════════════

class TestKnownPatterns:
    def test_known_patterns_structure(self):
        """每个模式必须包含关键字段。"""
        required = {"name", "theory", "test_statistic", "threshold", "priority"}
        for p in KNOWN_PATTERNS:
            for key in required:
                assert key in p, f"模式 {p.get('name', '?')} 缺少 {key}"

    def test_known_patterns_count(self):
        assert len(KNOWN_PATTERNS) == 6


# ═══════════════════════════════════════════════════════════════
# Part 10: 向后兼容
# ═══════════════════════════════════════════════════════════════

class TestBackwardCompat:
    def test_legacy_detection_result(self):
        from unified_emergence import DetectionResult
        dr = DetectionResult(
            detection_type="anomaly",
            species="test",
            description="test",
            confidence=0.5,
            evidence={},
        )
        assert dr.detection_type == "anomaly"

    def test_legacy_import(self):
        from unified_emergence import EmergenceMonitor, EmergenceEngine
        assert EmergenceMonitor is not None
        assert EmergenceEngine is not None
