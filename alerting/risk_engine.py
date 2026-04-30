#!/usr/bin/env python3
"""
Risk scoring and tiered alerting engine.

Implements:
- per-track EMA smoothing for emotion probabilities
- negative-conflict risk mapping
- robust crowd baseline using median + MAD
- tiered alerting (L1/L2/L3) based on z and duration (intensity + persistence)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import math
import time
from collections import deque


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return 0.5 * (s[mid - 1] + s[mid])


def _mad(values: List[float], med: Optional[float] = None) -> float:
    if not values:
        return 0.0
    if med is None:
        med = _median(values)
    abs_dev = [abs(v - med) for v in values]
    return _median(abs_dev)


@dataclass
class TrackState:
    track_id: int
    last_ts: float
    ema_probs: Dict[str, float] = field(default_factory=dict)
    risk_history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=300))
    z_history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=300))
    over_t1_since: Optional[float] = None
    last_level: int = 0


@dataclass(frozen=True)
class AlertEvent:
    ts: float
    level: int  # 1/2/3
    track_id: int
    risk: float
    z: float
    duration_sec: float
    crowd_median: float
    crowd_mad: float
    message: str


class RiskEngine:
    def __init__(self, settings: dict, emotions: List[str]):
        self.settings = settings
        self.emotions = emotions
        self.states: Dict[int, TrackState] = {}
        self.recent_alerts: Deque[AlertEvent] = deque(maxlen=int(settings.get("max_recent_alerts", 200)))

    def _ema_update(self, prev: float, x: float, alpha: float) -> float:
        return alpha * x + (1.0 - alpha) * prev

    def _compute_risk(self, probs: Dict[str, float]) -> float:
        w = self.settings["risk_weights"]
        return (
            w.get("angry", 0.0) * float(probs.get("angry", 0.0))
            + w.get("fear", 0.0) * float(probs.get("fear", 0.0))
            + w.get("disgust", 0.0) * float(probs.get("disgust", 0.0))
            + w.get("sad", 0.0) * float(probs.get("sad", 0.0))
        )

    def update(
        self,
        ts: float,
        track_probs: Dict[int, Dict[str, float]],
    ) -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, float]], List[AlertEvent]]:
        """
        Update engine with current per-track emotion probabilities.

        Returns:
        - per-track smoothed probs
        - per-track risk metrics: {risk, z, duration_sec, level}
        - list of new alerts triggered at this timestamp
        """
        alpha = float(self.settings["ema_alpha"])
        eps = float(self.settings.get("eps", 1e-6))

        # Update per-track EMA and compute risk list for baseline
        risks: Dict[int, float] = {}
        smoothed: Dict[int, Dict[str, float]] = {}

        for tid, probs in track_probs.items():
            st = self.states.get(tid)
            if st is None:
                st = TrackState(track_id=tid, last_ts=ts, ema_probs={e: 0.0 for e in self.emotions})
                self.states[tid] = st

            for e in self.emotions:
                x = float(probs.get(e, 0.0))
                st.ema_probs[e] = self._ema_update(st.ema_probs.get(e, 0.0), x, alpha)

            smoothed[tid] = dict(st.ema_probs)
            r = self._compute_risk(st.ema_probs)
            risks[tid] = r
            st.risk_history.append((ts, r))
            st.last_ts = ts

        active_risks = list(risks.values())
        crowd_med = _median(active_risks)
        crowd_mad = _mad(active_risks, med=crowd_med)
        denom = 1.4826 * crowd_mad + eps

        t1 = float(self.settings["thresholds"]["t1"])
        t2 = float(self.settings["thresholds"]["t2"])
        t3 = float(self.settings["thresholds"]["t3"])
        d1 = float(self.settings["durations"].get("d1_sec", 2.0))
        d2 = float(self.settings["durations"]["d2_sec"])
        d3 = float(self.settings.get("durations_ext", {}).get("d3_sec", 6.0))

        metrics: Dict[int, Dict[str, float]] = {}
        z_now: Dict[int, float] = {}
        for tid, r in risks.items():
            z = float((r - crowd_med) / denom)
            z_now[tid] = z

        new_alerts: List[AlertEvent] = []

        for tid, r in risks.items():
            st = self.states[tid]
            z = z_now[tid]
            st.z_history.append((ts, z))

            if z > t1:
                if st.over_t1_since is None:
                    st.over_t1_since = ts
            else:
                st.over_t1_since = None

            duration = float(0.0 if st.over_t1_since is None else (ts - st.over_t1_since))

            # Determine level
            level = 0
            if z > t1 and duration >= d1:
                level = 1
            if z > t2 and duration >= d2:
                level = 2
            # L3: only when intensity is extremely high (per requirement)
            if z > t3:
                level = 3

            metrics[tid] = {
                "risk": float(r),
                "z": float(z),
                "duration_sec": float(duration),
                "level": float(level),
                "crowd_median": float(crowd_med),
                "crowd_mad": float(crowd_mad),
            }

            # Trigger alert event on escalation or first time reaching a level
            if level > 0 and level >= st.last_level and level != st.last_level:
                msg = self._format_message(level, z, duration)
                ev = AlertEvent(
                    ts=ts,
                    level=level,
                    track_id=tid,
                    risk=float(r),
                    z=float(z),
                    duration_sec=float(duration),
                    crowd_median=float(crowd_med),
                    crowd_mad=float(crowd_mad),
                    message=msg,
                )
                self.recent_alerts.append(ev)
                new_alerts.append(ev)

            st.last_level = level

        # Cleanup stale states (not present in this tick)
        ttl = float(self.settings.get("track_ttl_sec", 10.0))
        to_del = []
        for tid, st in self.states.items():
            if tid not in track_probs and (ts - st.last_ts) > ttl:
                to_del.append(tid)
        for tid in to_del:
            del self.states[tid]

        return smoothed, metrics, new_alerts

    def _format_message(self, level: int, z: float, duration: float) -> str:
        if level == 1:
            return f"一级关注：负面情绪出现 (z={z:.2f})"
        if level == 2:
            return f"二级异常：负面情绪明显偏离 (z={z:.2f})"
        # per requirement: frontend should show wording without duration
        return "高强度负面情绪"

