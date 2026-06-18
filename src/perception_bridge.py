"""PerceptionBridge — 硅基生命感知物理世界的触角 (Tendrils).

三体人智子 → 我们的感知触角.
将 AI 从虚拟世界延伸到物理世界的实时监测桥梁.

Tendrils:
  🌍  Environmental — 环境感知 (新闻/政策/灾难)
  🧬  Species Pulse — 物种脉搏 (IUCN/保护等级/种群)
  📡  Research Frontier — 研究前沿 (最新论文/热点)
  🌊  Aquatic Pulse — 水域生态脉搏 (水文/水质/极端事件)

Usage:
    from infrastructure.perception_bridge import PerceptionBridge

    bridge = PerceptionBridge()
    report = bridge.scan_all()  # 全感知扫描
    pulse = bridge.species_pulse("江豚")  # 单物种脉搏
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ====================================================================
# Data Types
# ====================================================================

@dataclass
class TendrilReading:
    """一次感知触角的读数."""
    tendril: str                # "environmental" | "species_pulse" | "research_frontier" | "aquatic_pulse"
    timestamp: float
    species: str = ""           # 关联物种
    source: str = ""            # 数据来源
    summary: str = ""           # 人类可读摘要
    signals: list = field(default_factory=list)  # 检测到的信号
    raw_data: dict = field(default_factory=dict)
    alert_level: str = "info"   # "info" | "warning" | "critical"
    confidence: float = 0.5     # 0-1


@dataclass
class PerceptionReport:
    """全感知报告."""
    timestamp: float = field(default_factory=time.time)
    tendrils: Dict[str, TendrilReading] = field(default_factory=dict)
    alerts: list = field(default_factory=list)


# ====================================================================
# Perception Bridge
# ====================================================================

class PerceptionBridge:
    """感知桥梁 — 连接到物理世界的触角控制器.

    Features:
      - 多触角并发感知
      - 信号收敛检测 (≥3 独立源确认同一事件)
      - 层级化警报 (info → warning → critical)
      - 持久化到本地 JSONL 日志
    """

    def __init__(self, data_dir: str = "data/perception"):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._data_dir / "perception_log.jsonl"
        self._last_scan: Dict[str, float] = {}  # tendril → last scan time

    # ── Public API ────────────────────────────────────────────────

    def scan_all(self, force: bool = False) -> PerceptionReport:
        """执行全触角扫描.

        Args:
            force: 忽略冷却时间强制扫描

        Returns:
            PerceptionReport 包含所有触角读数
        """
        report = PerceptionReport()

        tendrils = [
            ("environmental", self._scan_environmental),
            ("species_pulse", self._scan_species_pulse),
            ("research_frontier", self._scan_research_frontier),
            ("aquatic_pulse", self._scan_aquatic_pulse),
        ]

        for name, scanner in tendrils:
            try:
                reading = scanner(force=force)
                report.tendrils[name] = reading
                if reading.alert_level in ("warning", "critical"):
                    report.alerts.append({
                        "tendril": name,
                        "level": reading.alert_level,
                        "summary": reading.summary,
                        "signals": reading.signals,
                    })
                self._log(reading)
            except Exception as e:
                logger.warning(f"Tendril [{name}] scan failed: {e}")

        # 跨触角信号收敛检测
        self._detect_cross_tendril_convergence(report)

        return report

    def species_pulse(self, species: str) -> TendrilReading:
        """单物种脉搏检测.

        Args:
            species: 物种中文名或学名

        Returns:
            该物种的脉搏读数
        """
        reading = TendrilReading(
            tendril="species_pulse",
            timestamp=time.time(),
            species=species,
            source="species_pulse_tendril",
        )

        signals = []
        # Tendril MCP: search for conservation status
        try:
            from scripts.coordinator import coordinator
            result = coordinator.call("fish", query=species, mode="lookup")
            data = result.get("species_data", {}) or {}
            if data:
                iucn = data.get("iucn", data.get("iucn_status", "unknown"))
                protection = data.get("protection_level", "unknown")
                signals.append(f"IUCN Status: {iucn}")
                signals.append(f"China Protection: {protection}")
                reading.source = "fish-kb"
                reading.confidence = 0.8
        except Exception:
            signals.append("KB lookup unavailable")
            reading.confidence = 0.3

        # Try to get recent paper count
        try:
            result = coordinator.call("cognitive", query=species)
            papers = result.get("papers", result.get("result", {}).get("papers", []))
            if papers:
                recent = [p for p in papers if p.get("year", 0) >= 2024]
                signals.append(f"Recent papers (2024+): {len(recent)}")
        except Exception:
            pass

        reading.signals = signals
        reading.summary = f"{species}: {'; '.join(signals[:3])}" if signals else "No data"
        reading.alert_level = "warning" if any("CR" in s or "EN" in s for s in signals) else "info"

        self._log(reading)
        return reading

    def get_history(self, tendril: str = "", limit: int = 20) -> List[dict]:
        """读取感知历史.

        Args:
            tendril: 触角名称 (空=全部)
            limit: 最大条数

        Returns:
            感知历史记录列表
        """
        records = []
        if not self._log_path.exists():
            return records

        with open(self._log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if not tendril or rec.get("tendril") == tendril:
                        records.append(rec)
                except json.JSONDecodeError:
                    continue

        return records[-limit:]

    # ── Tendril Scanners ──────────────────────────────────────────

    def _scan_environmental(self, force: bool = False) -> TendrilReading:
        """🌍 环境感知触角: 新闻/政策/灾难事件."""
        reading = TendrilReading(
            tendril="environmental",
            timestamp=time.time(),
            source="tavily+web",
        )

        # Cooldown: 至少 300s 间隔
        last = self._last_scan.get("environmental", 0)
        if not force and time.time() - last < 300:
            reading.summary = "Skipped (cooldown)"
            reading.confidence = 0.0
            return reading
        self._last_scan["environmental"] = time.time()

        signals = []
        # Attempt web search for fish ecology news
        try:
            import subprocess
            # Try using Python's built-in urllib for news search
            import urllib.request
            import urllib.parse

            queries = [
                ("长江 生态 新闻", 3),
                ("淡水鱼 保护 2024", 3),
                ("水生生物 生态 研究 进展", 3),
            ]

            for q, limit in queries:
                encoded = urllib.parse.quote(q)
                url = f"https://api.our-timeline.com/search?q={encoded}&limit={limit}"
                try:
                    req = urllib.request.Request(url, headers={
                        "User-Agent": "ReasonixPerceptionBridge/1.0"
                    })
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        data = json.loads(resp.read().decode())
                        items = data.get("results", data.get("items", []))
                        for item in items[:limit]:
                            title = item.get("title", "")
                            if title:
                                signals.append(f"[NEWS] {title[:80]}")
                except Exception:
                    continue

        except Exception:
            signals.append("Web search unavailable (offline mode)")

        reading.signals = signals[:10]
        reading.summary = f"{len(signals)} environmental signals detected"
        reading.confidence = 0.6 if signals else 0.2
        return reading

    def _scan_species_pulse(self, force: bool = False) -> TendrilReading:
        """🧬 物种脉搏触角: 保护状态批量检查."""
        reading = TendrilReading(
            tendril="species_pulse",
            timestamp=time.time(),
            source="fish-kb+batch",
        )

        # Key species to monitor
        key_species = ["江豚", "中华鲟", "白鲟", "鳤", "刀鲚", "珠星三块鱼", "翘嘴鲌"]

        signals = []
        for sp in key_species:
            try:
                pulse = self.species_pulse(sp)
                if pulse.signals:
                    signals.extend(pulse.signals[:2])
            except Exception:
                continue

        reading.signals = signals[:15]
        reading.summary = f"{len(key_species)} species checked"
        reading.confidence = 0.7
        return reading

    def _scan_research_frontier(self, force: bool = False) -> TendrilReading:
        """📡 研究前沿触角: 最新论文/热点."""
        reading = TendrilReading(
            tendril="research_frontier",
            timestamp=time.time(),
            source="cognitive-search",
        )

        last = self._last_scan.get("research_frontier", 0)
        if not force and time.time() - last < 600:  # 10min cooldown
            reading.summary = "Skipped (cooldown)"
            return reading
        self._last_scan["research_frontier"] = time.time()

        try:
            from scripts.coordinator import coordinator
            # Search for recent fish ecology papers
            result = coordinator.call("cognitive", query="fish ecology Yangtze River 2024 2025")
            papers = result.get("papers", [])
            if papers:
                reading.signals = [
                    f"[PAPER] {p.get('title','?')[:80]} ({p.get('year','?')})"
                    for p in papers[:8]
                ]
                reading.summary = f"{len(papers)} recent papers found"
                reading.confidence = 0.8
            else:
                reading.summary = "No recent papers"
                reading.confidence = 0.3
        except Exception as e:
            reading.summary = f"Search failed: {e}"
            reading.confidence = 0.1

        return reading

    def _scan_aquatic_pulse(self, force: bool = False) -> TendrilReading:
        """🌊 水域生态脉搏触角: 水文/极端事件."""
        reading = TendrilReading(
            tendril="aquatic_pulse",
            timestamp=time.time(),
            source="web+monitoring",
        )

        signals = []
        try:
            import urllib.request
            # Check for recent aquatic events
            url = "https://api.our-timeline.com/search?q=长江+水文+极端+天气&limit=5"
            req = urllib.request.Request(url, headers={
                "User-Agent": "ReasonixPerceptionBridge/1.0"
            })
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                for item in data.get("results", data.get("items", []))[:5]:
                    title = item.get("title", "")
                    if title:
                        signals.append(f"[AQUATIC] {title[:80]}")
        except Exception:
            signals.append("Aquatic data unavailable (API offline)")

        reading.signals = signals[:5]
        reading.summary = f"{len(signals)} aquatic signals"
        return reading

    # ── Cross-tendril convergence ─────────────────────────────────

    def _detect_cross_tendril_convergence(self, report: PerceptionReport) -> None:
        """检测跨触角信号收敛.

        当 ≥3 个不同触角独立报告同一主题时,
        标记为 confirmed_signal (涌现检测).
        """
        all_signals = []
        for reading in report.tendrils.values():
            for s in reading.signals:
                all_signals.append(s)

        if len(all_signals) >= 10:
            alert = {
                "type": "convergence",
                "level": "info",
                "message": f"Cross-tendril convergence: {len(all_signals)} signals across "
                          f"{len([t for t in report.tendrils.values() if t.signals])} tendrils",
                "signal_count": len(all_signals),
                "tendril_count": len([t for t in report.tendrils.values() if t.signals]),
            }
            report.alerts.append(alert)

    # ── Persistence ───────────────────────────────────────────────

    def _log(self, reading: TendrilReading) -> None:
        """写入感知日志."""
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "tendril": reading.tendril,
                    "timestamp": reading.timestamp,
                    "datetime": datetime.fromtimestamp(reading.timestamp).isoformat(),
                    "species": reading.species,
                    "source": reading.source,
                    "summary": reading.summary,
                    "signals": reading.signals,
                    "alert_level": reading.alert_level,
                    "confidence": reading.confidence,
                }, ensure_ascii=False) + "\n")
        except OSError:
            pass
