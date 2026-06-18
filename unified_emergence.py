"""
unified_emergence.py — 统一涌现检测引擎
=========================================
融合三项目涌现能力:
  - p项目 (porpoise-agent): 实时 Z-score 监控 + D₀/D₁/D₂/D₃ 维度感知
  - f项目 (fish-ecology-assistant): 三层分析（异常→突变→理论匹配）+ 6理论模式
  - c项目 (cognitive-search-engine): 自组织领域发现 (emerge_domains)

架构:
  Online (实时监控)              Offline (批次分析)
  ┌─────────────────────┐      ┌──────────────────────┐
  │  EmergenceMonitor    │      │  EmergenceEngine     │
  │  · Z-score 异常检测   │      │  · Layer 1 异常      │
  │  · D₀~D₃ 维度追踪    │      │  · Layer 2 突变点     │
  │  · D₂→D₃ 相变检测    │      │  · Layer 3 理论匹配   │
  └─────────────────────┘      └──────────────────────┘
          │                            │
          └──────────┬─────────────────┘
                     ▼
          ┌──────────────────────┐
          │  emerge_domains()    │
          │  自组织领域发现       │
          └──────────────────────┘

Usage:
    # 实时监控
    mon = EmergenceMonitor(emergence_threshold_sigma=3.0, min_sources=3)
    mon.record("recall", 0.85, DimensionalLevel.D1)
    signals = mon.check_emergence()

    # 离线分析
    engine = EmergenceEngine()
    results = engine.scan(data={"years": [2018,...,2025], "biomass": [100,...,260]})

    # 自组织领域发现
    suggestions = emerge_domains(catalog)
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# scipy — 可选 (p-value 统计显著性)
try:
    from scipy.stats import norm as _norm
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ═══════════════════════════════════════════════════════════
# Part 1: 基础类型 — 信号、维度、检测结果
# ═══════════════════════════════════════════════════════════

class EmergenceType(Enum):
    """涌现类型分类。"""
    BENEFICIAL = "beneficial"          # 系统意外改善
    NEUTRAL = "neutral"                # 有趣但无害
    HARMFUL = "harmful"                # 系统意外退化
    PHASE_TRANSITION = "phase_transition"  # 维度跃迁 (D₁→D₂ 等)
    ANOMALY = "anomaly"                # 无法解释的离群点


class DimensionalLevel(Enum):
    """维度等级 (D₀→D₁→D₂→D₃ 严格包含层次)。"""
    D0 = 0  # Point — 原子状态
    D1 = 1  # Line — 因果轨迹
    D2 = 2  # Plane — 拓扑网格
    D3 = 3  # Body — 闭环实体


@dataclass
class EmergenceSignal:
    """实时涌现事件信号 (来自 porpoise-agent)。"""
    id: str
    timestamp: float
    emergence_type: EmergenceType
    dimensional_level: DimensionalLevel
    sources: list[str]              # 确认该信号的独立源
    metrics: dict[str, float]       # 异常指标
    deviation_sigma: float          # 偏离基线多少σ
    description: str
    confidence: float               # 0-1
    resolved: bool = False
    resolution_note: str = ""


@dataclass
class DetectionResult:
    """离线批次分析检测结果 (取代 f 项目原 EmergenceSignal)。"""
    detection_type: str  # "anomaly" | "change_point" | "theory_match"
    species: str         # 物种名
    description: str     # 人类可读描述
    confidence: float    # 0-1
    evidence: dict       # 数据证据
    suggested_theory: str = ""
    suggested_action: str = ""
    detected_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class MetricTracker:
    """运行统计追踪器 — Welford 在线方差算法。

    使用 Welford (1962) 单次遍历、数值稳定算法计算均值和方差。
    修正了原 _sum/_sum_sq 只加不减的溢出 bug。

    Reference:
      B. P. Welford (1962). "Note on a Method for Calculating Corrected
      Sums of Squares and Products". Technometrics 4(3):419–420.
    """
    name: str
    history: deque = field(default_factory=lambda: deque(maxlen=100))
    # Welford accumulator
    _n: int = 0
    _mean: float = 0.0
    _M2: float = 0.0   # 二阶中心矩 (sum of squared deviations)

    def record(self, value: float):
        """记录一个新值 (Welford 在线更新)。"""
        self.history.append(value)
        self._n += 1
        delta = value - self._mean
        self._mean += delta / self._n
        delta2 = value - self._mean
        self._M2 += delta * delta2

    @property
    def mean(self) -> float:
        return self._mean if self._n > 0 else 0.0

    @property
    def variance(self) -> float:
        """样本方差 (n-1 分母)。"""
        if self._n < 2:
            return 1.0
        return self._M2 / (self._n - 1)

    @property
    def std(self) -> float:
        return math.sqrt(max(self.variance, 0.0)) or 1.0

    @property
    def latest(self) -> Optional[float]:
        return self.history[-1] if self.history else None

    def deviation_sigma(self, value: float) -> float:
        """当前值偏离均值多少个标准差。"""
        return abs(value - self.mean) / max(self.std, 0.001)

    @property
    def n(self) -> int:
        return self._n

    def stats(self) -> dict:
        """返回完整统计摘要。"""
        return {
            "name": self.name,
            "n": self._n,
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "variance": round(self.variance, 6),
            "latest": self.latest,
            "sigma": round(self.deviation_sigma(self.latest), 3) if self.latest is not None else None,
        }


# ═══════════════════════════════════════════════════════════
# Part 2: 实时监控 — EmergenceMonitor (p项目原版)
# ═══════════════════════════════════════════════════════════

class EmergenceMonitor:
    """实时涌现监控器 — 持续追踪所有维度的涌现信号。

    核心原理:
      1. 追踪所有维度的指标
      2. 检测突变 (非线性变化)
      3. 分类涌现类型 (beneficial / neutral / harmful)
      4. 检测到涌现时触发适应

    检测规则:
      - ≥3个独立源指向同一意外模式 → EMERGENCE
      - 相变: |Δmetric| > 2σ 偏离基线
      - 自组织临界: 事件规模的幂律分布

    _is_beneficial 通过 constructor 的 beneficial_metrics / harmful_metrics 可配置。
    """

    def __init__(
        self,
        emergence_threshold_sigma: float = 3.0,
        min_sources: int = 3,
        beneficial_metrics: set[str] | None = None,
        harmful_metrics: set[str] | None = None,
    ):
        self.threshold_sigma = emergence_threshold_sigma
        self.min_sources = min_sources
        self.trackers: dict[str, MetricTracker] = {}
        self.signals: list[EmergenceSignal] = []
        self._signal_counter: int = 0
        self._beneficial_metrics = beneficial_metrics or {
            "recall", "precision", "verification_pass_rate",
            "pipeline_success_rate", "success_rate", "accuracy",
            "f1_score", "throughput",
        }
        self._harmful_metrics = harmful_metrics or {
            "error_rate", "latency", "entropy", "false_positive_rate",
            "cost", "response_time", "failure_rate",
        }

    def record(self, metric_name: str, value: float, level: DimensionalLevel):
        """记录一个指标值到指定维度。"""
        key = f"{level.name}:{metric_name}"
        if key not in self.trackers:
            self.trackers[key] = MetricTracker(name=key)
        self.trackers[key].record(value)

    def record_batch(self, metrics: dict[str, float], level: DimensionalLevel):
        """批量记录多个指标。"""
        for name, value in metrics.items():
            self.record(name, value, level)

    def check_emergence(self) -> list[EmergenceSignal]:
        """检查所有追踪指标, 返回新检测到的涌现信号。"""
        new_signals = []

        # 检查单个指标偏差
        anomalous_metrics = []
        for key, tracker in self.trackers.items():
            latest = tracker.latest
            if latest is None:
                continue
            sigma = tracker.deviation_sigma(latest)
            if sigma >= self.threshold_sigma:
                anomalous_metrics.append((key, latest, sigma, tracker.mean))

        # 按维度分组
        by_level: dict[DimensionalLevel, list] = {}
        for key, value, sigma, mean in anomalous_metrics:
            level_str = key.split(":")[0]
            try:
                level = DimensionalLevel[level_str]
            except KeyError:
                continue
            if level not in by_level:
                by_level[level] = []
            by_level[level].append((key, value, sigma, mean))

        # 检查是否 ≥ min_sources 在某个维度触发
        for level, metrics in by_level.items():
            if len(metrics) >= self.min_sources:
                deviations = [m[2] for m in metrics]
                avg_dev = sum(deviations) / len(deviations)
                is_beneficial = self._is_beneficial(metrics)

                signal = EmergenceSignal(
                    id=f"EMG-{self._signal_counter:04d}",
                    timestamp=time.time(),
                    emergence_type=(
                        EmergenceType.PHASE_TRANSITION
                        if avg_dev > 5.0
                        else EmergenceType.BENEFICIAL if is_beneficial
                        else EmergenceType.HARMFUL
                    ),
                    dimensional_level=level,
                    sources=[m[0] for m in metrics],
                    metrics={m[0].split(":")[1]: m[1] for m in metrics},
                    deviation_sigma=avg_dev,
                    description=self._describe_emergence(level, metrics, avg_dev, is_beneficial),
                    confidence=min(1.0, len(metrics) / (self.min_sources + 2)),
                )
                new_signals.append(signal)
                self._signal_counter += 1

        self.signals.extend(new_signals)
        return new_signals

    def _is_beneficial(self, metrics: list[tuple]) -> bool:
        """判断异常方向: 改善还是恶化。

        使用 self._beneficial_metrics (越高越好) 和
        self._harmful_metrics (越低越好) 两组可配置集合。
        可通过 __init__ 的 beneficial_metrics / harmful_metrics 参数自定义。
        """
        for key, value, sigma, mean in metrics:
            metric_name = key.split(":")[1]
            if metric_name in self._beneficial_metrics and value > mean:
                return True
            if metric_name in self._harmful_metrics and value < mean:
                return True
        return False

    def _describe_emergence(self, level: DimensionalLevel, metrics: list[tuple],
                            avg_dev: float, is_beneficial: bool) -> str:
        level_names = {
            DimensionalLevel.D0: "Point (D₀) — 原子状态偏移",
            DimensionalLevel.D1: "Line (D₁) — 轨迹异常",
            DimensionalLevel.D2: "Plane (D₂) — 网格拓扑变化",
            DimensionalLevel.D3: "Body (D₃) — 闭环系统相变",
        }
        metric_names = [m[0].split(":")[1] for m in metrics]
        direction = "改善" if is_beneficial else "退化"
        return (
            f"{level_names.get(level, '未知')}: "
            f"{len(metrics)} 个指标 {direction} ({', '.join(metric_names[:3])}) "
            f"偏差 {avg_dev:.1f}σ"
        )

    @property
    def pending_signals(self) -> list[EmergenceSignal]:
        return [s for s in self.signals if not s.resolved]

    @property
    def phase_transitions(self) -> list[EmergenceSignal]:
        return [s for s in self.signals
                if s.emergence_type == EmergenceType.PHASE_TRANSITION]

    def health_report(self) -> dict:
        """健康报告摘要。"""
        return {
            "tracked_metrics": len(self.trackers),
            "total_signals": len(self.signals),
            "pending_signals": len(self.pending_signals),
            "phase_transitions": len(self.phase_transitions),
            "by_level": {
                "D0": sum(1 for s in self.signals
                          if s.dimensional_level == DimensionalLevel.D0),
                "D1": sum(1 for s in self.signals
                          if s.dimensional_level == DimensionalLevel.D1),
                "D2": sum(1 for s in self.signals
                          if s.dimensional_level == DimensionalLevel.D2),
                "D3": sum(1 for s in self.signals
                          if s.dimensional_level == DimensionalLevel.D3),
            },
        }

    def get_metric_stats(self, metric_name: str, level: DimensionalLevel) -> dict | None:
        """获取指定指标的运行统计。"""
        key = f"{level.name}:{metric_name}"
        t = self.trackers.get(key)
        if not t:
            return None
        return {
            "name": t.name,
            "n": t.n,
            "mean": t.mean,
            "std": t.std,
            "latest": t.latest,
            "deviation_sigma": t.deviation_sigma(t.latest) if t.latest is not None else None,
        }


# ═══════════════════════════════════════════════════════════
# Part 3: 维度演进监控 — DimensionalEmergenceMonitor
# ═══════════════════════════════════════════════════════════

class DimensionalEmergenceMonitor(EmergenceMonitor):
    """维度演进专用涌现监控器。

    与 dimensional_evolution.py 集成, 检测 D₀→D₁→D₂→D₃ 的相变。
    """

    def track_dimension_transition(
        self,
        from_level: DimensionalLevel,
        to_level: DimensionalLevel,
        transition_cost: float,
    ):
        """记录一次维度跃迁事件。"""
        self.record(
            f"transition_cost_{from_level.name}_to_{to_level.name}",
            transition_cost,
            to_level,
        )
        self.record("current_dimension", to_level.value, to_level)

    def check_dimensional_emergence(self) -> Optional[EmergenceSignal]:
        """专门检查维度相变。"""
        signals = self.check_emergence()
        for sig in signals:
            if sig.emergence_type == EmergenceType.PHASE_TRANSITION:
                return sig

        # 检查 D₂→D₃ 跃迁: 多个未解决的 D₂ 信号累积
        d2_count = sum(
            1 for s in self.signals
            if s.dimensional_level == DimensionalLevel.D2 and not s.resolved
        )
        if d2_count >= self.min_sources:
            self._signal_counter += 1
            signal = EmergenceSignal(
                id=f"EMG-{self._signal_counter:04d}",
                timestamp=time.time(),
                emergence_type=EmergenceType.PHASE_TRANSITION,
                dimensional_level=DimensionalLevel.D3,
                sources=[f"D2_signal_{i}" for i in range(d2_count)],
                metrics={"d2_signal_count": d2_count},
                deviation_sigma=4.0,
                description=(
                    f"Body (D₃) 涌现: {d2_count} 个 D₂ 信号累积"
                    f" — 自组织临界 → 闭环系统形成"
                ),
                confidence=min(0.9, d2_count / 6),
            )
            self.signals.append(signal)
            return signal

        return None


# ═══════════════════════════════════════════════════════════
# Part 4: 离线批次分析 — EmergenceEngine (三层: 异常→突变→理论)
# ═══════════════════════════════════════════════════════════


def _deduplicate_changes(
    changes: list[dict],
    min_segment_length: int = 3,
) -> list[dict]:
    """去重突变点: 相近年份只保留效应量最大的。"""
    if len(changes) <= 1:
        return changes
    deduped = [changes[0]]
    for r in changes[1:]:
        if r["year"] - deduped[-1]["year"] <= min_segment_length:
            if r["magnitude"] > deduped[-1]["magnitude"]:
                deduped[-1] = r
        else:
            deduped.append(r)
    return deduped


KNOWN_PATTERNS: list[dict] = [
    {
        "name": "非对称恢复",
        "signal": "体型恢复速率 > 多样性恢复速率",
        "theory": "非对称恢复假说 (蔡方陶 2026)",
        "test_statistic": "body_size_slope / diversity_slope",
        "threshold": 2.0,
        "direction": "above",
        "priority": "P0",
    },
    {
        "name": "K策略者悖论",
        "signal": "K策略者恢复率 > r策略者恢复率",
        "theory": "r-K选择理论 (MacArthur 1967) + 受损基数假说",
        "test_statistic": "K_recovery_rate / r_recovery_rate",
        "threshold": 1.5,
        "direction": "above",
        "priority": "P1",
    },
    {
        "name": "连通性效应",
        "signal": "通江湖泊恢复 > 隔离湖泊恢复",
        "theory": "岛屿生物地理学 (MacArthur & Wilson 1967)",
        "test_statistic": "connected_lake_recovery / isolated_lake_recovery",
        "threshold": 1.3,
        "direction": "above",
        "priority": "P1",
    },
    {
        "name": "中度干扰",
        "signal": "物种多样性在中等干扰时最高",
        "theory": "中度干扰假说 (Connell 1978)",
        "test_statistic": "H_diversity vs disturbance_level",
        "threshold": 0.0,
        "direction": "peak",
        "priority": "P2",
    },
    {
        "name": "自然流态断裂",
        "signal": "水文改变量 > 历史变异范围 → 鱼类群落退化",
        "theory": "自然流态范式 (Poff 1997) + 道法自然",
        "test_statistic": "hydrologic_alteration vs community_change",
        "threshold": 1.0,
        "direction": "above",
        "priority": "P1",
    },
    {
        "name": "降维打击",
        "signal": "群落组成越过不可逆阈值",
        "theory": "状态转换模型 (Westoby 1989) + 三体降维打击",
        "test_statistic": "state_transition_detected",
        "threshold": 0,
        "direction": "above",
        "priority": "P0",
    },
]


class EmergenceEngine:
    """离线涌现分析引擎 — 三层架构。

    Layer 1: 时间序列异常检测 → 发现"意外"
    Layer 2: 突变点检测 → 定位"何时发生"
    Layer 3: 理论匹配 → 解释"为什么"

    与 EmergenceMonitor 的实时监控不同,
    EmergenceEngine 处理完整的批次数据集。
    """

    KNOWN_PATTERNS = KNOWN_PATTERNS

    def __init__(
        self,
        data_path: str | Path | None = None,
        feedback_file: str | Path | None = None,
    ) -> None:
        self._data_path = Path(data_path) if data_path else Path("data")
        self._feedback_file = (
            Path(feedback_file) if feedback_file
            else Path.cwd() / "logs" / "catalog_feedback.jsonl"
        )

    # ── Layer 1: 异常检测 ──

    @staticmethod
    def detect_anomalies(
        time_series: list[float],
        dates: list[int],
        method: str = "zscore",
        sensitivity: float = 0.05,
    ) -> list[dict]:
        """Layer 1 — 时间序列异常检测。

        方法:
          - "zscore": 标准化偏差法 (≥3σ 为异常, 附带 p-value)
          - "iqr":    四分位距法 (附带 p-value)
          - "window": 滑动窗口法

        Args:
            time_series: 数值序列
            dates:       对应年份
            method:      "zscore" | "iqr" | "window"
            sensitivity: 灵敏度 (0.01-0.10)

        Returns:
            [{"year": int, "value": float, "z_score": float,
              "p_value": float|None, "is_anomaly": bool}, ...]
        """
        n = len(time_series)
        if n < 5:
            return [{"year": d, "value": v, "is_anomaly": False}
                    for d, v in zip(dates, time_series)]

        # Z-score 方法
        if method == "zscore":
            mean = sum(time_series) / n
            variance = sum((x - mean) ** 2 for x in time_series) / n
            std = math.sqrt(variance) if variance > 0 else 1.0
            threshold = max(2.0, 3.0 - sensitivity * 20)
            results = []
            for year, value in zip(dates, time_series):
                z = (value - mean) / std
                p_val = (2.0 * (1.0 - _norm.cdf(abs(z)))
                         if _HAS_SCIPY else None)
                results.append({
                    "year": year,
                    "value": value,
                    "z_score": round(z, 3),
                    "p_value": round(p_val, 6) if p_val is not None else None,
                    "is_anomaly": abs(z) > threshold,
                })
            return results

        # IQR 方法
        if method == "iqr":
            sorted_vals = sorted(time_series)
            q1 = sorted_vals[n // 4]
            q3 = sorted_vals[3 * n // 4]
            iqr_val = q3 - q1
            lower = q1 - 1.5 * iqr_val
            upper = q3 + 1.5 * iqr_val
            mean_val = sum(time_series) / n
            std_val = math.sqrt(sum((x - mean_val) ** 2 for x in time_series) / n) or 1.0
            results = []
            for year, value in zip(dates, time_series):
                z = (value - mean_val) / std_val
                p_val = (2.0 * (1.0 - _norm.cdf(abs(z)))
                         if _HAS_SCIPY else None)
                results.append({
                    "year": year,
                    "value": value,
                    "z_score": round(z, 3),
                    "p_value": round(p_val, 6) if p_val is not None else None,
                    "is_anomaly": value < lower or value > upper,
                })
            return results

        # 滑动窗口方法
        if method == "window":
            window_size = max(3, n // 3)
            results = []
            for i, (year, value) in enumerate(zip(dates, time_series)):
                left = max(0, i - window_size)
                window = time_series[left:i]
                if len(window) < 2:
                    results.append({
                        "year": year, "value": value,
                        "z_score": 0.0, "is_anomaly": False,
                    })
                    continue
                w_mean = sum(window) / len(window)
                w_var = sum((x - w_mean) ** 2 for x in window) / len(window)
                w_std = math.sqrt(w_var) if w_var > 0 else 1.0
                z = (value - w_mean) / w_std
                threshold = max(1.5, 2.5 - sensitivity * 20)
                results.append({
                    "year": year,
                    "value": value,
                    "z_score": round(z, 3),
                    "is_anomaly": abs(z) > threshold,
                })
            return results

        raise ValueError(f"未知检测方法: {method}")

    # ── Layer 2: 突变点检测 ──

    @staticmethod
    def detect_change_points(
        time_series: list[float],
        dates: list[int],
        method: str = "cusum",
        min_segment_length: int = 3,
        cusum_threshold: float = 3.0,
        pelt_penalty: float = 3.0,
    ) -> list[dict]:
        """Layer 2 — 突变点检测 (含 CUSUM + PELT)。

        方法:
          - "cusum":   CUSUM 累积和检测 (默认, 适合生态数据渐进变化)
          - "pelt":    PELT-like 二分分割 (自动定位, 无需预知突变点数)
          - "sliding": 滑动窗口对比 (保留向后兼容)
          - "diff":    差分法 (连续点差异 > 阈值)

        CUSUM (Page 1954):
          双边累积和, 追踪偏离参考均值的累积偏差。
          当累积和超过阈值时标记突变点，然后重置。
          适合检测生态学中禁捕/污染等事件的渐进影响。

        Returns:
            [{"year": int, "change_type": "up"|"down",
              "magnitude": float, "confidence": float, "method": str}, ...]
        """
        n = len(time_series)
        if n < min_segment_length * 2:
            return []

        # ═══════════════════════════
        # CUSUM — 累积和控制图
        # ═══════════════════════════
        if method == "cusum":
            series_mean = sum(time_series) / n
            series_std = math.sqrt(
                sum((x - series_mean) ** 2 for x in time_series) / n
            ) or 1.0

            s_pos = 0.0  # 向上累积
            s_neg = 0.0  # 向下累积
            results = []
            ref = series_mean
            drift = 0.5 * series_std

            for i, (year, value) in enumerate(zip(dates, time_series)):
                z = (value - ref) / series_std
                s_pos = max(0.0, s_pos + z - drift / series_std)
                s_neg = min(0.0, s_neg + z + drift / series_std)

                if s_pos >= cusum_threshold:
                    results.append({
                        "year": year, "index": i,
                        "change_type": "up",
                        "magnitude": round(s_pos, 3),
                        "confidence": min(1.0, s_pos / (cusum_threshold * 2)),
                        "method": "cusum",
                    })
                    s_pos = 0.0

                if s_neg <= -cusum_threshold:
                    results.append({
                        "year": year, "index": i,
                        "change_type": "down",
                        "magnitude": round(abs(s_neg), 3),
                        "confidence": min(1.0, abs(s_neg) / (cusum_threshold * 2)),
                        "method": "cusum",
                    })
                    s_neg = 0.0

            return _deduplicate_changes(results, min_segment_length)

        # ═══════════════════════════
        # PELT-like 二分分割
        # ═══════════════════════════
        if method == "pelt":
            n_pts = len(time_series)
            C = [0.0] * n_pts
            cp = [0] * n_pts

            total_mean = sum(time_series) / n_pts
            total_std = math.sqrt(
                sum((x - total_mean) ** 2 for x in time_series) / n_pts
            ) or 1.0

            for t in range(min_segment_length, n_pts):
                best_cost = float("inf")
                best_s = t
                for s in range(min_segment_length, t - min_segment_length + 1):
                    seg1 = time_series[s:t+1]
                    seg0 = time_series[cp[s-1]:s] if s > 0 else time_series[:s]
                    cost = C[s-1] if s > 0 else 0.0
                    for seg in [seg0, seg1]:
                        if len(seg) > 1:
                            seg_mean = sum(seg) / len(seg)
                            cost += sum((x - seg_mean) ** 2 for x in seg) / (total_std ** 2)
                    cost += pelt_penalty
                    if cost < best_cost:
                        best_cost = cost
                        best_s = s
                C[t] = best_cost
                cp[t] = best_s

            changes_indices = []
            t = n_pts - 1
            while t > min_segment_length:
                s = cp[t]
                if s > 0 and s < t:
                    left_seg = time_series[s-min_segment_length:s+1]
                    right_seg = time_series[s+1:t+1]
                    left_m = sum(left_seg) / len(left_seg) if left_seg else 0
                    right_m = sum(right_seg) / len(right_seg) if right_seg else 0
                    effect = (right_m - left_m) / total_std
                    changes_indices.append({
                        "index": s,
                        "year": dates[s] if s < len(dates) else dates[-1],
                        "change_type": "up" if effect > 0 else "down",
                        "magnitude": round(abs(effect), 3),
                        "confidence": min(1.0, abs(effect) / 2.0),
                        "method": "pelt",
                    })
                t = s - 1

            return sorted(
                _deduplicate_changes(changes_indices, min_segment_length),
                key=lambda x: x["year"],
            )

        # ═══════════════════════════
        # 滑动窗口 (向后兼容)
        # ═══════════════════════════
        if method == "sliding":
            series_mean = sum(time_series) / n
            series_var = sum((x - series_mean) ** 2 for x in time_series) / n
            series_std = math.sqrt(series_var) if series_var > 0 else 1.0
            results = []
            for i in range(min_segment_length, n - min_segment_length + 1):
                left = time_series[i - min_segment_length:i]
                right = time_series[i:i + min_segment_length]
                left_mean = sum(left) / len(left)
                right_mean = sum(right) / len(right)
                diff = right_mean - left_mean
                effect = abs(diff) / series_std
                if effect >= 0.5:
                    results.append({
                        "year": dates[i], "index": i,
                        "change_type": "up" if diff > 0 else "down",
                        "magnitude": round(effect, 3),
                        "confidence": min(1.0, effect / 2.0),
                        "method": "sliding",
                    })
            return _deduplicate_changes(results, min_segment_length)

        # 差分法 (向后兼容)
        if method == "diff":
            series_mean = sum(time_series) / n
            series_std = math.sqrt(
                sum((x - series_mean) ** 2 for x in time_series) / n
            ) or 1.0
            results = []
            for i in range(1, n):
                diff = time_series[i] - time_series[i - 1]
                effect = abs(diff) / series_std
                if effect >= 1.0:
                    results.append({
                        "year": dates[i], "index": i,
                        "change_type": "up" if diff > 0 else "down",
                        "magnitude": round(effect, 3),
                        "confidence": min(1.0, effect / 3.0),
                        "method": "diff",
                    })
            return _deduplicate_changes(results, min_segment_length)

        raise ValueError(f"未知检测方法: {method}")

    # ── Layer 3: 理论-数据匹配 ──

    @staticmethod
    def match_theory(
        observations: dict[str, float],
        species: str = "",
    ) -> list[dict]:
        """Layer 3 — 将观察到的模式与已知理论预测匹配。

        支持 direction 字段: "above", "below", "peak"
        确保 "body_size_slope / diversity_slope" 等复合统计量
        已在 observations 中预先构造好 (由 scan() 自动完成)。

        Returns:
            [{"pattern_name": str, "theory": str, "match_score": float,
              "evidence": str, "priority": str}, ...]
        """
        matches = []
        for pattern in KNOWN_PATTERNS:
            stat = pattern["test_statistic"]
            if stat in observations:
                value = observations[stat]
                direction = pattern.get("direction", "above")
                threshold = pattern["threshold"]

                matched = False
                match_score = 0.0
                if direction == "above" and value >= threshold:
                    matched = True
                    match_score = (
                        min(value / threshold, 1.0) if threshold > 0 else 1.0
                    )
                elif direction == "below" and value <= threshold:
                    matched = True
                    match_score = (
                        min(threshold / max(value, 0.001), 1.0)
                        if value > 0 else 1.0
                    )
                elif direction == "peak":
                    matched = value > 0
                    match_score = min(value, 1.0) if matched else 0.0

                if matched:
                    matches.append({
                        "pattern_name": pattern["name"],
                        "theory": pattern["theory"],
                        "match_score": round(match_score, 4),
                        "threshold": threshold,
                        "observed": value,
                        "priority": pattern["priority"],
                        "signal": pattern["signal"],
                        "direction": direction,
                    })
        return sorted(matches, key=lambda x: x["match_score"], reverse=True)

    # ── 综合扫描 ──

    def scan(
        self,
        species: str = "",
        data: dict[str, Any] | None = None,
        auto_theory: bool = True,
        anomaly_method: str = "zscore",
        change_method: str = "cusum",
        sensitivity: float = 0.05,
    ) -> list[dict]:
        """一键扫描: 异常→突变→理论匹配 + 自动反馈记录。

        Args:
            species: 物种名
            data: {"years": [2021,2022,...], "biomass": [1.0,2.1,...], ...}
            auto_theory: 是否自动计算并匹配理论
            anomaly_method: 异常检测方法 ("zscore"|"iqr"|"window")
            change_method: 突变点检测方法 ("cusum"|"pelt"|"sliding"|"diff")
            sensitivity: 灵敏度 (0.01-0.10)

        Returns:
            DetectionResult 格式的 dict 列表
        """
        results: list[dict] = []
        if not data:
            return [{"detection_type": "status",
                     "description": "需要提供时间序列数据"}]

        years = data.get("years", [])
        if not years:
            return [{"detection_type": "status",
                     "description": "需要提供年份序列"}]

        # Layer 1: 异常检测
        for key, values in data.items():
            if key == "years" or not isinstance(values, list):
                continue
            if len(values) != len(years):
                continue
            try:
                anomalies = self.detect_anomalies(
                    values, years, method=anomaly_method,
                    sensitivity=sensitivity,
                )
                for a in anomalies:
                    if a.get("is_anomaly"):
                        results.append({
                            "detection_type": "anomaly",
                            "species": species,
                            "variable": key,
                            "description": (
                                f"{species} {key} 在 {a['year']} 年出现异常 "
                                f"(z_score={a['z_score']})"
                            ),
                            "confidence": min(1.0, abs(a["z_score"]) / 4.0),
                            "evidence": a,
                            "suggested_theory": "",
                            "suggested_action": f"检查 {key} 在 {a['year']} 年的数据来源",
                            **a,
                        })
            except Exception:
                continue

        # Layer 2: 突变点检测
        for key, values in data.items():
            if key == "years" or not isinstance(values, list):
                continue
            if len(values) != len(years):
                continue
            try:
                changes = self.detect_change_points(
                    values, years, method=change_method,
                )
                for c in changes:
                    results.append({
                        "detection_type": "change_point",
                        "species": species,
                        "variable": key,
                        "description": (
                            f"{species} {key} 在 {c['year']} 年发生"
                            f"{'上升' if c['change_type']=='up' else '下降'}突变"
                        ),
                        "confidence": c["confidence"],
                        "evidence": c,
                        "suggested_theory": "",
                        "suggested_action": "标注该突变点并追溯原因",
                        **c,
                    })
            except Exception:
                continue

        # Layer 3: 理论匹配
        if auto_theory:
            # Step 1: 计算简单斜率
            observations: dict[str, float] = {}
            for key, values in data.items():
                if key == "years" or not isinstance(values, list):
                    continue
                if len(values) >= 3:
                    n = len(values)
                    x_mean = sum(range(n)) / n
                    y_mean = sum(values) / n
                    slope = (
                        sum((i - x_mean) * (v - y_mean)
                            for i, v in enumerate(values))
                        / sum((i - x_mean) ** 2 for i in range(n))
                        if n > 1 else 0
                    )
                    if abs(slope) > 0.001:
                        observations[f"{key}_slope"] = slope

            # Step 2: 自动构造复合统计量 (使理论模式可匹配)
            # 例如: "body_size_slope / diversity_slope" → 从已有 slope 推算
            for pattern in KNOWN_PATTERNS:
                stat = pattern["test_statistic"]
                expr = stat.strip()

                # 处理 "A / B" 比值
                if " / " in expr:
                    parts = [p.strip() for p in expr.split(" / ")]
                    if all(p in observations for p in parts):
                        observations[expr] = observations[parts[0]] / max(
                            observations[parts[1]], 0.001
                        )

                # 处理 "A vs B" 对比
                elif " vs " in expr:
                    parts = [p.strip() for p in expr.split(" vs ")]
                    if all(p in observations for p in parts):
                        observations[expr] = abs(
                            observations[parts[0]] - observations[parts[1]]
                        )

            theory_matches = self.match_theory(observations, species)
            for tm in theory_matches:
                results.append({
                    "detection_type": "theory_match",
                    "species": species,
                    "description": (
                        f"理论匹配: {tm['pattern_name']} — {tm['theory']}"
                    ),
                    "confidence": tm["match_score"],
                    "evidence": tm,
                    "suggested_theory": tm["theory"],
                    "suggested_action": f"基于 {tm['pattern_name']} 撰写论文",
                    **tm,
                })

        # 自动记录反馈 (供 emerge_domains 积累数据)
        self._record_feedback(species, results)

        return results

    def _record_feedback(self, species: str, results: list[dict]):
        """自动记录扫描结果到反馈日志, 供 emerge_domains 使用。"""
        try:
            self._feedback_file.parent.mkdir(parents=True, exist_ok=True)
            counters: dict[str, int] = {}
            for r in results:
                dt = r.get("detection_type", "unknown")
                counters[dt] = counters.get(dt, 0) + 1

            with open(self._feedback_file, "a", encoding="utf-8") as f:
                record = {
                    "ts": datetime.now().isoformat(),
                    "query": f"scan:{species}",
                    "db": "emergence_engine",
                    "result_count": len(results),
                    "useful": len(results) > 0,
                    "details": counters,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass  # 反馈记录失败不影响主流程

    def signals_summary(self) -> list[dict]:
        """已检测到的所有涌现信号摘要。"""
        return []


# ═══════════════════════════════════════════════════════════
# Part 5: 自组织领域发现 (来自 c 项目 catalog_loader)
# ═══════════════════════════════════════════════════════════

def record_search_result(
    query: str,
    db: str,
    result_count: int,
    useful: bool = True,
    feedback_file: str | Path | None = None,
):
    """记录搜索反馈到日志文件, 供 emerge_domains 使用。"""
    fb_path = Path(feedback_file) if feedback_file else (
        Path.cwd() / "logs" / "catalog_feedback.jsonl"
    )
    fb_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now().isoformat(),
        "query": query,
        "db": db,
        "result_count": result_count,
        "useful": useful,
    }
    with open(fb_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def emerge_domains(
    catalog: dict,
    feedback_file: str | Path | None = None,
) -> list[dict]:
    """自组织领域发现 — 分析反馈日志, 发现跨领域DB聚类。

    工作原理:
      1. 将反馈按查询分组 → 找出哪些DB共同出现
      2. 聚类共同出现的DB → 候选领域
      3. 从成功查询中提取共享触发词
      4. 返回带置信度的建议
    """
    fb_path = Path(feedback_file) if feedback_file else (
        Path.cwd() / "logs" / "catalog_feedback.jsonl"
    )
    if not fb_path.exists():
        return []

    # Step 1: 构建 query→DBs 映射
    query_dbs: dict[str, set] = defaultdict(set)
    query_success: dict[str, bool] = {}
    with open(fb_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                q = r["query"]
                query_dbs[q].add(r["db"])
                if r.get("useful"):
                    query_success[q] = True
            except (json.JSONDecodeError, KeyError):
                continue

    if len(query_dbs) < 3:
        return []

    # Step 2: 计算 DB 共现
    db_cooccurrence: dict[tuple[str, str], int] = defaultdict(int)
    for dbs in query_dbs.values():
        db_list = sorted(dbs)
        for i in range(len(db_list)):
            for j in range(i + 1, len(db_list)):
                db_cooccurrence[(db_list[i], db_list[j])] += 1

    # Step 3: 聚类 (简单阈值)
    suggestions: list[dict] = []
    cluster_queries = [q for q, dbs in query_dbs.items() if len(dbs) >= 2]

    for (dom_a, dom_b), count in sorted(
        db_cooccurrence.items(), key=lambda x: -x[1]
    ):
        if count >= 2:  # 至少 2 次共现
            triggers = [
                q for q in cluster_queries
                if dom_a in query_dbs[q] and dom_b in query_dbs[q]
            ][:5]
            dom_a_repr = next(
                (k for k in query_dbs if dom_a in query_dbs[k]),
                dom_a,
            )
            dom_b_repr = next(
                (k for k in query_dbs if dom_b in query_dbs[k]),
                dom_b,
            )
            dom_a_label = catalog.get("domains", {}).get(dom_a_repr, {}).get("label", dom_a_repr)
            dom_b_label = catalog.get("domains", {}).get(dom_b_repr, {}).get("label", dom_b_repr)
            suggestions.append({
                "label": f"{dom_a_label}×{dom_b_label}",
                "triggers": triggers,
                "databases": list({dom_a, dom_b}),
                "confidence": min(1.0, count / 3),
                "evidence": f"{count}次共现于{len(cluster_queries)}次查询",
            })

    return suggestions


# ═══════════════════════════════════════════════════════════
# CLI 测试
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("💡 演示脚本移至 infrastructure/examples/demo_emergence.py")
    print("   运行: python -m infrastructure.examples.demo_emergence")
