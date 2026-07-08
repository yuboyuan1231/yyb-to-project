from __future__ import annotations

import numpy as np


class TemporalGrid:
    def __init__(self, clip_len: float = 1.5, max_clips: int = 64) -> None:
        self.clip_len = float(clip_len)
        self.max_clips = int(max_clips)

    def grid_spans(self, duration: float, max_spans: int = 64) -> np.ndarray:
        t = max(1, min(self.max_clips, int(np.ceil(float(duration) / self.clip_len))))
        durations = np.array([2, 4, 6, 8, 12, 16, 24, 32, 48, 64], dtype=np.int32)
        spans = []
        for d in durations:
            if d > t:
                continue
            stride = max(1, d // 4)
            for s in range(0, t - d + 1, stride):
                spans.append((s, s + d))
        if not spans:
            spans = [(0, max(1, min(t, 2))), (0, t)]
        arr = np.array(spans, dtype=np.int32)
        if len(arr) > max_spans:
            idx = np.linspace(0, len(arr) - 1, int(max_spans)).round().astype(np.int64)
            arr = arr[idx]
        return arr

    def clips_to_seconds(self, spans: np.ndarray, duration: float) -> np.ndarray:
        out = spans.astype(np.float32) * self.clip_len
        out[:, 1] = np.minimum(out[:, 1], float(duration))
        out[:, 0] = np.minimum(out[:, 0], np.maximum(0.0, out[:, 1] - 1e-3))
        return out


def iou_1d(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return float(inter / max(union, 1e-8))

