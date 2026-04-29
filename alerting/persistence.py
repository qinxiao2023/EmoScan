#!/usr/bin/env python3
"""
Persistence for alert evidence:
- keyframe jpeg
- trajectory (recent bboxes)
- curves (emotion probs/risk/z)
- metadata json
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class PersistedAlertRef:
    alert_id: str
    level: int
    track_id: int
    ts: float
    folder: str
    keyframe_jpg: str
    metadata_json: str


class AlertPersistence:
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def persist(
        self,
        session_id: str,
        level: int,
        track_id: int,
        ts: float,
        keyframe_bgr: np.ndarray,
        trajectory: List[Tuple[float, Tuple[float, float, float, float]]],
        curves: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> PersistedAlertRef:
        date = time.strftime("%Y%m%d", time.localtime(ts))
        folder = self.root_dir / date / session_id / f"track_{track_id}"
        folder.mkdir(parents=True, exist_ok=True)

        alert_id = f"{int(ts*1000)}_L{level}_T{track_id}"
        keyframe_path = folder / f"{alert_id}_keyframe.jpg"
        meta_path = folder / f"{alert_id}_meta.json"
        traj_path = folder / f"{alert_id}_trajectory.json"
        curves_path = folder / f"{alert_id}_curves.json"

        # Keyframe
        cv2.imencode(".jpg", keyframe_bgr)[1].tofile(str(keyframe_path))

        # JSON payloads
        with open(traj_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"ts": float(t), "bbox_xyxy": [float(x) for x in bbox]} for (t, bbox) in trajectory],
                f,
                ensure_ascii=False,
                indent=2,
            )
        with open(curves_path, "w", encoding="utf-8") as f:
            json.dump(curves, f, ensure_ascii=False, indent=2)

        meta = dict(metadata)
        meta.update({"alert_id": alert_id, "level": int(level), "track_id": int(track_id), "ts": float(ts)})
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return PersistedAlertRef(
            alert_id=alert_id,
            level=int(level),
            track_id=int(track_id),
            ts=float(ts),
            folder=str(folder),
            keyframe_jpg=str(keyframe_path),
            metadata_json=str(meta_path),
        )

