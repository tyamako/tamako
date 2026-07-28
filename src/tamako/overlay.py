"""顔への PNG 合成と、検出の途切れを埋める追従。

顔検出は毎フレーム安定して当たるものではない。横を向いた、手が重なった、
ぶれた、というだけで数フレーム落ちる。落ちた瞬間に素顔が出てしまっては
意味がないので、見失っても直前の位置に出し続ける（hold）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

from .faces import FaceBox


class OverlayError(RuntimeError):
    """重ねる画像を読めない、または使えない。"""


def load_mask(path: str | Path) -> np.ndarray:
    """重ねる PNG を BGRA で読む。透過が無い画像は不透明として扱う。"""
    target = Path(path)
    if not target.is_file():
        raise OverlayError(f"重ねる画像が見つかりません: {target}")

    # 日本語を含むパスでも読めるよう、バイト列から復号する。
    buffer = np.frombuffer(target.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OverlayError(f"画像として読めません: {target}")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 3:
        alpha = np.full(image.shape[:2] + (1,), 255, dtype=np.uint8)
        image = np.concatenate([image, alpha], axis=2)
    if image.shape[2] != 4:
        raise OverlayError(f"想定外のチャンネル数です ({image.shape[2]}): {target}")
    return image


def composite(frame: np.ndarray, mask_rgba: np.ndarray, box: FaceBox) -> None:
    """frame の box の位置に mask を合成する。frame を直接書き換える。"""
    height, width = frame.shape[:2]
    x0 = int(math.floor(box.x))
    y0 = int(math.floor(box.y))
    x1 = int(math.ceil(box.x + box.w))
    y1 = int(math.ceil(box.y + box.h))

    # 画面外にはみ出す分は、貼り付ける側も同じだけ切り取る。
    src_x0 = max(0, -x0)
    src_y0 = max(0, -y0)
    dst_x0, dst_y0 = max(0, x0), max(0, y0)
    dst_x1, dst_y1 = min(width, x1), min(height, y1)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return

    target_w, target_h = x1 - x0, y1 - y0
    if target_w <= 0 or target_h <= 0:
        return

    resized = cv2.resize(mask_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)
    patch = resized[src_y0:src_y0 + (dst_y1 - dst_y0), src_x0:src_x0 + (dst_x1 - dst_x0)]

    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    region = frame[dst_y0:dst_y1, dst_x0:dst_x1]
    blended = patch[:, :, :3].astype(np.float32) * alpha + region.astype(np.float32) * (1.0 - alpha)
    frame[dst_y0:dst_y1, dst_x0:dst_x1] = blended.astype(np.uint8)


@dataclass
class _Track:
    """追いかけている 1 つの顔。"""

    box: FaceBox
    missed: int = 0
    seen: int = 1


@dataclass
class MaskTracker:
    """複数の顔を追い、見失っても一定時間は隠し続ける。

    hold_frames を大きくすると顔バレの危険は下がるが、人が去った後も
    画像が残りやすくなる。既定は 0.7 秒相当。
    """

    hold_frames: int
    match_distance_ratio: float = 1.2
    tracks: List[_Track] = field(default_factory=list)
    # 検出が無いまま hold で描いたフレーム数。後で人に知らせるために数える。
    held_frames: int = 0

    def update(self, detections: Sequence[FaceBox]) -> List[FaceBox]:
        """今フレームの検出結果を渡し、実際に隠すべき箱の一覧を受け取る。"""
        unmatched = list(detections)
        for track in self.tracks:
            best_index: Optional[int] = None
            best_distance = float("inf")
            cx, cy = track.box.center
            limit = max(track.box.w, track.box.h) * self.match_distance_ratio
            for index, candidate in enumerate(unmatched):
                dx_, dy_ = candidate.center
                distance = math.hypot(dx_ - cx, dy_ - cy)
                if distance < best_distance and distance <= limit:
                    best_distance, best_index = distance, index
            if best_index is None:
                track.missed += 1
            else:
                track.box = unmatched.pop(best_index)
                track.missed = 0
                track.seen += 1

        for leftover in unmatched:
            self.tracks.append(_Track(box=leftover))

        self.tracks = [t for t in self.tracks if t.missed <= self.hold_frames]

        if detections:
            result = [t.box for t in self.tracks]
        else:
            result = [t.box for t in self.tracks]
            if result:
                self.held_frames += 1
        return result
