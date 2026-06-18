# infrastructure — Reasonix Unified Emergence Detection Engine

> **道生一 · 一生二 · 二生三 · 三生万物**
>
> Integrated three-project emergence capability: real-time monitoring + three-layer analysis + domain discovery + Chinese NLP + vision detection

---

## Overview

`infrastructure` is the unified foundation engine for the [Reasonix seven-project system](https://github.com/NeroZ02). It fuses emergence detection capabilities from three core projects into a single composable codebase.

| Module | Source | Capability |
|:-----|:-----|:-----|
| **unified_emergence** | Fusion of p/f/c projects | Real-time Z-score monitoring · D₀→D₃ dimensional tracking · Three-layer analysis (anomaly→change→theory) · Self-organizing domain discovery |
| **fish_classifier** | Independent integration | HuggingFace fish recognition: 60fishmodel (60 species) / Fish-Vista (1900 species) / DINOv2 feature extraction |
| **chinese_nlp** | Independent integration | Chinese ecological academic NLP: HanLP + Jiagu segmentation/POS tagging · NER entity recognition · Synonym matching |
| **fish_detector** | Independent integration | FishDet-M + YOLO fish school detection: underwater camera fish counting · species-level monitoring · video frame batch analysis |

### Role in the "三生万物" Ecosystem

```
道 (Operator)
  ├── 一 (IProjectAdapter unified interface)
  │    └── infrastructure provides shared emergence detection for all adapters
  ├── 二 (阳·Extension / 阴·Contraction)
  │    ├── EmergenceMonitor → Real-time yang-face expansion monitoring
  │    └── EmergenceEngine → Offline yin-face convergence analysis
  ├── 三 (Three-loop closure: fish → cognitive → porpoise/coilia/culter)
  │    └── infrastructure cross-cuts all three projects as shared engine
  └── 万物 (P₁...Pn unlimited species specialization)
       └── Any Pn agent can import infrastructure to gain emergence perception
```

**Authoritative architecture doc:** [`docs/SANSHENG_WANWU.md`](https://github.com/NeroZ02/Reasonix/blob/main/docs/SANSHENG_WANWU.md) — supersedes all legacy architectures (WUXING / TAIJI / LAYERS)

---

## Installation

### Base install (core only, no extra dependencies)

```bash
cd infrastructure
pip install -e .
```

Core functionality (EmergenceMonitor / EmergenceEngine / emerge_domains) works out of the box.

### Optional dependency groups

```bash
# Emergence statistical significance (p-value)
pip install -e ".[emergence]"

# Chinese ecological NLP (segmentation/NER/synonyms)
pip install -e ".[nlp]"

# Fish image classification (HuggingFace models)
pip install -e ".[vision]"

# Fish school object detection (YOLO)
pip install -e ".[detection]"

# All-in-one
pip install -e ".[all]"

# Development environment
pip install -e ".[dev]"
```

---

## Quick Start

### 1. Real-time Emergence Monitoring

```python
from infrastructure import EmergenceMonitor, DimensionalLevel

# Initialize monitor
mon = EmergenceMonitor(emergence_threshold_sigma=3.0, min_sources=3)

# Record multi-dimensional metrics
mon.record("recall", 0.85, DimensionalLevel.D1)
mon.record("precision", 0.92, DimensionalLevel.D1)
mon.record("throughput", 1250, DimensionalLevel.D1)

# Batch recording
mon.record_batch({
    "accuracy": 0.88,
    "latency": 45,
    "error_rate": 0.02,
}, DimensionalLevel.D2)

# Check for emergence
signals = mon.check_emergence()
for sig in signals:
    print(f"{sig.description} | σ={sig.deviation_sigma:.1f} | confidence={sig.confidence:.2f}")

# Health report
print(mon.health_report())
```

### 2. Offline Batch Analysis (Three-Layer Scan)

```python
from infrastructure import EmergenceEngine

engine = EmergenceEngine()

# Layer 1→2→3 scan: anomaly → change-point → theory match
data = {
    "years": [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
    "body_size": [100, 95, 88, 105, 130, 175, 210, 260],
    "diversity": [35, 34, 33, 32, 33, 34, 35, 36],
}

results = engine.scan(data=data, species="Ochetobius elongatus")
for r in results:
    if r["detection_type"] == "theory_match":
        print(f"Theory: {r['pattern_name']} ({r['suggested_theory']}) confidence={r['confidence']:.3f}")
```

### 3. Self-Organizing Domain Discovery

```python
from infrastructure import emerge_domains, record_search_result

# Record search feedback
record_search_result("Coilia nasus migration routes", db="fishbase", result_count=12, useful=True)
record_search_result("Coilia nasus conservation", db="cnki", result_count=8, useful=True)

# Discover cross-domain clustering
catalog = {"domains": {"fishbase": {"label": "FishBase"}, "cnki": {"label": "CNKI"}}}
suggestions = emerge_domains(catalog)
for s in suggestions:
    print(f"New domain: {s['label']} (confidence={s['confidence']:.2f})")
```

### 4. Fish Image Classification

```python
from infrastructure import classify_60fish, extract_features_dinov2

# Fast 60 common species classification
predictions = classify_60fish("fish.jpg")
# → [("Carassius auratus", 0.95), ...]

# DINOv2 self-supervised features (low-sample Yangtze endemic species)
features = extract_features_dinov2("rare_fish.jpg")
# → 768-dim vector
```

### 5. Chinese Ecological NLP

```python
from infrastructure import segment, ner, synonym_search

# Segmentation + POS tagging
words = segment("2024年长江安庆段刀鲚资源调查")
# → [("2024年", "TIME"), ("长江", "NS"), ...]

# Named entity recognition
entities = ner("刀鲚洄游群体在鄱阳湖产卵场聚集")
# → [{"text":"刀鲚","type":"SPECIES"}, {"text":"鄱阳湖","type":"LOCATION"}, ...]

# Synonym matching
synonyms = synonym_search("刀鲚")
# → ["长江刀鱼", "Coilia nasus", "刀鱼", "长颌鲚"]
```

### 6. Fish School Object Detection

```python
from infrastructure import detect_image, process_video

# Single image detection
detections = detect_image("underwater.jpg", conf=0.25)
# → [{"bbox":[x1,y1,x2,y2], "confidence":0.87, "class":"fish"}, ...]

# Video frame sampling detection
process_video("survey.mp4", output_dir="results/", fps_sample=10)
```

---

## API Reference

### Core Data Types

| Type | Description |
|:-----|:-----|
| `EmergenceType` | Emergence category enum: BENEFICIAL / NEUTRAL / HARMFUL / PHASE_TRANSITION / ANOMALY |
| `DimensionalLevel` | Dimensional level D₀(Point) / D₁(Line) / D₂(Plane) / D₃(Body) |
| `EmergenceSignal` | Real-time emergence event signal (id, timestamp, sources, deviation_sigma, confidence, ...) |
| `DetectionResult` | Offline batch analysis detection result (detection_type, species, evidence, suggested_theory, ...) |
| `MetricTracker` | Welford online variance tracker (mean, std, deviation_sigma, stats) |

### Core Classes

| Class | Methods | Description |
|:---|:-----|:-----|
| `EmergenceMonitor` | `record()`, `record_batch()`, `check_emergence()`, `health_report()` | Real-time emergence monitoring |
| `DimensionalEmergenceMonitor` | `track_dimension_transition()`, `check_dimensional_emergence()` | Dimensional transition monitoring |
| `EmergenceEngine` | `detect_anomalies()`, `detect_change_points()`, `match_theory()`, `scan()` | Offline three-layer analysis |

### Utility Functions

| Function | Description |
|:-----|:-----|
| `emerge_domains(catalog)` | Self-organizing domain discovery — analyze feedback logs, discover cross-domain DB clusters |
| `record_search_result(query, db, count, useful)` | Record search feedback to log file |

### Module-Level Functions (fish_classifier / chinese_nlp / fish_detector)

| Function | Module | Description |
|:-----|:-----|:-----|
| `classify_60fish(image_path)` | fish_classifier | 60fishmodel classification, returns [(species, confidence), ...] |
| `extract_features_dinov2(image_path)` | fish_classifier | DINOv2 768-dim feature vector |
| `download_fishvista()` | fish_classifier | Download Fish-Vista dataset (1900 species) |
| `segment(text)` | chinese_nlp | Jiagu segmentation + POS tagging |
| `ner(text)` | chinese_nlp | Custom lexicon NER entity recognition |
| `synonym_search(word)` | chinese_nlp | Ecological academic synonym matching |
| `detect_image(image_path, conf)` | fish_detector | YOLO single-image fish detection |
| `process_video(video_path, output_dir, fps_sample)` | fish_detector | Video frame sampling fish school detection |

---

## Project Structure

```
infrastructure/
├── unified_emergence.py    # Canonical emergence engine (fusion of p/f/c projects)
├── fish_classifier.py      # HuggingFace fish recognition (60fish/Fish-Vista/DINOv2)
├── chinese_nlp.py          # Chinese ecological NLP (Jiagu/Synonyms)
├── fish_detector.py        # YOLO fish school detection (FishDet-M)
├── src/
│   └── __init__.py         # Unified export interface
├── tests/
│   ├── test_unified_emergence.py   # Emergence engine tests (30+ items)
│   └── test_integration.py         # Integration tests
├── pyproject.toml          # Project metadata & dependency declarations
└── README.md               # This file
```

> **Note:** `unified_emergence` in infrastructure/ is the **canonical source**. `eon-core/src/unified_emergence.py` re-exports from here.

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run integration tests
pytest tests/test_integration.py -v

# With coverage
pytest tests/ -v --cov=infrastructure --cov-report=term
```

---

## Related Resources

- **Authoritative architecture:** [`docs/SANSHENG_WANWU.md`](https://github.com/NeroZ02/Reasonix/blob/main/docs/SANSHENG_WANWU.md)
- **Execution flow:** [`docs/EXECUTION_FLOW.md`](https://github.com/NeroZ02/Reasonix/blob/main/docs/EXECUTION_FLOW.md)
- **Project relationships:** [`docs/PROJECT_RELATIONSHIPS.md`](https://github.com/NeroZ02/Reasonix/blob/main/docs/PROJECT_RELATIONSHIPS.md)
- **Coordination config:** `coordination.yaml`
- **Corpus library:** `../方隅文库/` — Personal knowledge base (ecological theory · Chinese ecological philosophy · analysis reports · sci-fi)

---

> ⚡ *"Emergence is not merely phenomenon — it is proof that the system is transitioning toward higher complexity."*
