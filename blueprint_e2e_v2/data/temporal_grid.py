from __future__ import annotations

import numpy as np


class TemporalGrid:
    def __init__(self, clip_len: float = 1.5, max_clips: int = 64) -> None:
        self.clip_len = float(clip_len)
        self.max_clips = int(max_clips)

    def num_clips(self, duration: float) -> int:
        return max(1, min(self.max_clips, int(np.ceil(float(duration) / self.clip_len))))

    def seconds_per_clip(self, duration: float) -> float:
        raw_clips = int(np.ceil(float(duration) / self.clip_len))
        if raw_clips > self.max_clips:
            return float(duration) / float(self.max_clips)
        return self.clip_len

    def grid_spans(self, duration: float, max_spans: int = 64) -> np.ndarray:
        t = self.num_clips(duration)
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
        out = spans.astype(np.float32) * self.seconds_per_clip(duration)
        out[:, 1] = np.minimum(out[:, 1], float(duration))
        out[:, 0] = np.minimum(out[:, 0], np.maximum(0.0, out[:, 1] - 1e-3))
        return out

    def seconds_to_clip_span(self, start: float, end: float, duration: float) -> np.ndarray:
        scale = self.seconds_per_clip(duration)
        s = int(np.floor(float(start) / scale))
        e = max(int(np.ceil(float(end) / scale)), s + 1)
        s = int(np.clip(s, 0, self.max_clips - 1))
        e = int(np.clip(e, s + 1, self.max_clips))
        return np.array([[s, e]], dtype=np.int32)


def iou_1d(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return float(inter / max(union, 1e-8))
