"""
test_integration.py — 集成测试
===============================
验证 infrastructure 四个模块能被正确导入和使用:
  - unified_emergence  (EmergenceMonitor / EmergenceEngine / emerge_domains)
  - fish_classifier    (MODELS / 依赖检查)
  - chinese_nlp        (ECOLOGY_DICT / segment / ner / synonym_search)
  - fish_detector      (detect_image / process_video / 依赖检查)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# 确保 infrastructure 包在路径上
_INFRA_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_INFRA_DIR))


# ═══════════════════════════════════════════════════════════════
# 1. unified_emergence 模块导入
# ═══════════════════════════════════════════════════════════════

class TestUnifiedEmergenceImport:
    """验证 unified_emergence 的核心类型和类可正确导入。"""

    def test_import_types(self):
        from unified_emergence import (
            EmergenceType,
            DimensionalLevel,
            EmergenceSignal,
            DetectionResult,
        )
        assert EmergenceType.BENEFICIAL.value == "beneficial"
        assert DimensionalLevel.D3.value == 3

        sig = EmergenceSignal(
            id="test", timestamp=0.0,
            emergence_type=EmergenceType.ANOMALY,
            dimensional_level=DimensionalLevel.D1,
            sources=["s1"], metrics={"x": 1.0},
            deviation_sigma=2.0, description="test", confidence=0.5,
        )
        assert sig.id == "test"

    def test_import_monitor(self):
        from unified_emergence import EmergenceMonitor
        mon = EmergenceMonitor()
        assert mon.threshold_sigma == 3.0
        assert mon.min_sources == 3

    def test_import_engine(self):
        from unified_emergence import EmergenceEngine
        engine = EmergenceEngine()
        assert engine is not None

    def test_import_tools(self):
        from unified_emergence import (
            MetricTracker,
            KNOWN_PATTERNS,
            emerge_domains,
            record_search_result,
        )
        assert len(KNOWN_PATTERNS) == 6
        t = MetricTracker(name="test")
        t.record(1.0)
        assert t.n == 1


# ═══════════════════════════════════════════════════════════════
# 2. fish_classifier 模块导入
# ═══════════════════════════════════════════════════════════════

class TestFishClassifierImport:
    """验证 fish_classifier 模块可正确导入, MODELS 注册表结构正确。"""

    def test_import_module(self):
        import fish_classifier
        assert hasattr(fish_classifier, "MODELS")
        assert hasattr(fish_classifier, "classify_60fish")
        assert hasattr(fish_classifier, "extract_features_dinov2")
        assert hasattr(fish_classifier, "download_fishvista")
        assert hasattr(fish_classifier, "benchmark")

    def test_models_registry(self):
        from fish_classifier import MODELS
        assert "60fish" in MODELS
        assert "fishvista" in MODELS
        assert "dinov2" in MODELS
        # 每个模型至少包含这些键
        for name, info in MODELS.items():
            assert "hf_id" in info, f"{name} 缺少 hf_id"
            assert "type" in info, f"{name} 缺少 type"
            assert "description" in info, f"{name} 缺少 description"
            assert "install" in info, f"{name} 缺少 install"
            assert isinstance(info["install"], list)

    def test_species_count(self):
        from fish_classifier import MODELS
        assert MODELS["60fish"]["species"] == 60
        assert MODELS["fishvista"]["species"] == 1900


# ═══════════════════════════════════════════════════════════════
# 3. chinese_nlp 模块导入
# ═══════════════════════════════════════════════════════════════

class TestChineseNLPImport:
    """验证 chinese_nlp 模块可正确导入, 词典结构正确。"""

    def test_import_module(self):
        import chinese_nlp
        assert hasattr(chinese_nlp, "ECOLOGY_DICT")
        assert hasattr(chinese_nlp, "segment")
        assert hasattr(chinese_nlp, "ner")
        assert hasattr(chinese_nlp, "synonym_search")

    def test_ecology_dict(self):
        from chinese_nlp import ECOLOGY_DICT
        assert isinstance(ECOLOGY_DICT, dict)
        assert "刀鲚" in ECOLOGY_DICT
        assert "鳤" in ECOLOGY_DICT
        assert "江豚" in ECOLOGY_DICT
        assert ECOLOGY_DICT["刀鲚"] == "SPECIES"
        assert ECOLOGY_DICT["长江"] == "LOCATION"
        assert ECOLOGY_DICT["洄游"] == "ECO_PROCESS"
        assert ECOLOGY_DICT["耳石"] == "METHOD"

    def test_dict_types(self):
        """验证词典值只使用定义的分类。"""
        from chinese_nlp import ECOLOGY_DICT
        valid_types = {"SPECIES", "ECO_PROCESS", "HABITAT", "METHOD", "LOCATION"}
        for word, entity_type in ECOLOGY_DICT.items():
            assert entity_type in valid_types, (
                f"词 '{word}' 的类型 '{entity_type}' 不在合法集合中"
            )

    def test_segment_no_jiagu(self):
        """未安装 jiagu 时 segment() 应优雅降级 (返回空列表)。"""
        import chinese_nlp
        result = chinese_nlp.segment("刀鲚是长江三鲜之首")
        # 如果 jiagu 未安装, 返回空列表; 如果已安装, 返回非空
        if result:
            assert isinstance(result[0], tuple)
            assert len(result[0]) == 2  # (word, pos)

    def test_ner_no_jiagu(self):
        """未安装 jiagu 时 ner() 应优雅降级。"""
        import chinese_nlp
        result = chinese_nlp.ner("长江安庆段刀鲚调查")
        if result:
            assert isinstance(result[0], dict)
            assert "text" in result[0]
            assert "type" in result[0]

    def test_synonym_no_synonyms(self):
        """未安装 synonyms 时 synonym_search() 应优雅降级。"""
        import chinese_nlp
        # 自定义同义词不依赖 synonyms 包
        result = chinese_nlp.synonym_search("刀鲚")
        assert len(result) > 0
        assert "长江刀鱼" in result or "Coilia nasus" in result


# ═══════════════════════════════════════════════════════════════
# 4. fish_detector 模块导入
# ═══════════════════════════════════════════════════════════════

class TestFishDetectorImport:
    """验证 fish_detector 模块可正确导入。"""

    def test_import_module(self):
        import fish_detector
        assert hasattr(fish_detector, "detect_image")
        assert hasattr(fish_detector, "process_video")

    def test_check_yolo_dependency(self):
        """验证 check_yolo 函数存在且可调用。"""
        import fish_detector
        assert callable(fish_detector.check_yolo)
        # 不假定 ultralytics 是否已安装, 只验证不抛异常
        result = fish_detector.check_yolo()
        assert isinstance(result, bool)


# ═══════════════════════════════════════════════════════════════
# 5. src 统一入口
# ═══════════════════════════════════════════════════════════════

class TestSrcUnifiedEntrypoint:
    """验证 src/__init__.py 统一导出接口的完整性。"""

    def test_src_imports_unified_emergence(self):
        from infrastructure.src import (
            EmergenceType,
            EmergenceMonitor,
            EmergenceEngine,
            emerge_domains,
        )
        assert EmergenceType is not None
        assert EmergenceMonitor is not None
        assert EmergenceEngine is not None
        assert emerge_domains is not None

    def test_src_imports_fish_classifier(self):
        from infrastructure.src import (
            MODELS,
            classify_60fish,
            extract_features_dinov2,
            download_fishvista,
            fish_benchmark,
        )
        assert isinstance(MODELS, dict)
        assert callable(classify_60fish)
        assert callable(extract_features_dinov2)
        assert callable(download_fishvista)
        assert callable(fish_benchmark)

    def test_src_imports_chinese_nlp(self):
        from infrastructure.src import (
            ECOLOGY_DICT,
            segment,
            ner,
            synonym_search,
        )
        assert isinstance(ECOLOGY_DICT, dict)
        assert callable(segment)
        assert callable(ner)
        assert callable(synonym_search)

    def test_src_imports_fish_detector(self):
        from infrastructure.src import (
            detect_image,
            process_video,
        )
        assert callable(detect_image)
        assert callable(process_video)

    def test_src_all_completeness(self):
        """验证 __all__ 完整度: 至少覆盖核心导出。"""
        from infrastructure.src import __all__
        essential = {
            "EmergenceMonitor", "EmergenceEngine", "emerge_domains",
            "classify_60fish", "extract_features_dinov2",
            "segment", "ner", "synonym_search",
            "detect_image", "process_video",
        }
        assert essential.issubset(set(__all__)), (
            f"__all__ 缺少: {essential - set(__all__)}"
        )


# ═══════════════════════════════════════════════════════════════
# 6. EmergenceEngine 端到端集成
# ═══════════════════════════════════════════════════════════════

class TestEndToEnd:
    """端到端集成: 涌现引擎 + 三条分析流水线。"""

    def test_full_pipeline(self):
        """验证 scan() 产出 anomaly + change_point + theory_match 三种结果。"""
        from unified_emergence import EmergenceEngine

        engine = EmergenceEngine()
        data = {
            "years": [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
            "body_size": [100, 95, 88, 105, 130, 175, 210, 260],
            "diversity": [35, 34, 33, 32, 33, 34, 35, 36],
        }
        results = engine.scan(data=data, species="集成测试")

        types_found = {r["detection_type"] for r in results}
        # 至少应有 theory_match (因为 body_size 上升 + diversity 恢复)
        assert "theory_match" in types_found

        for r in results:
            assert "detection_type" in r
            assert "confidence" in r
            assert isinstance(r["confidence"], (int, float))
            assert 0.0 <= r["confidence"] <= 1.0

    def test_monitor_then_engine(self):
        """实时监控 + 离线分析 组合场景。"""
        from unified_emergence import (
            EmergenceMonitor,
            EmergenceEngine,
            DimensionalLevel,
        )

        # Phase 1: 实时监控
        mon = EmergenceMonitor(
            emergence_threshold_sigma=3.0,
            min_sources=2,
            beneficial_metrics={"body_size", "diversity"},
            harmful_metrics={"extinction_rate"},
        )
        for v in [100, 102, 98, 101, 99]:
            mon.record("body_size", float(v), DimensionalLevel.D1)
            mon.record("diversity", float(v / 3), DimensionalLevel.D1)
        # 无涌现信号 (稳定基线)
        signals = mon.check_emergence()
        assert len(signals) == 0

        # Phase 2: 离线分析
        engine = EmergenceEngine()
        data = {
            "years": [2020, 2021, 2022, 2023, 2024],
            "body_size": [100, 102, 98, 101, 99],
        }
        results = engine.scan(data=data, species="稳定物种")
        # 稳定数据不应误报
        theory_matches = [r for r in results if r["detection_type"] == "theory_match"]
        assert len(theory_matches) == 0

    def test_emerge_domains_requires_feedback(self):
        """emerge_domains 无反馈日志时应返回空列表。"""
        from unified_emergence import emerge_domains
        # 使用一个不存在的反馈文件路径
        suggestions = emerge_domains(
            {},
            feedback_file=Path(_INFRA_DIR) / "tests" / "_no_such_feedback.jsonl",
        )
        assert suggestions == []
