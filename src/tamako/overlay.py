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
        raise OverlayError(f"想定外のチャンネル数です ({target.name}: {image.shape[2]} ch)")
    return image


def effective_coverage(mask_rgba: np.ndarray) -> float:
    """マスク中心に置ける最大の不透明矩形が、画像全体に占める割合（辺の比）。

    実際に顔を隠すのは箱ではなく PNG の不透明画素なので、丸や星型のステッカーは
    箱が顔を含んでいても角の透明部分から顔が出る。中心から矩形を広げていき、
    不透明率 99% を保てる最大の大きさを二分探索で求める。円形ならおよそ 0.70。
    """
    alpha = mask_rgba[:, :, 3] >= 128
    height, width = alpha.shape
    if not alpha.any():
        return 0.0
    # 積分画像で任意矩形の不透明画素数を O(1) で数える。
    integral = np.zeros((height + 1, width + 1), dtype=np.int64)
    np.cumsum(np.cumsum(alpha, axis=0), axis=1, out=integral[1:, 1:])

    def opaque_ratio(fraction: float) -> float:
        half_w = max(1, int(width * fraction / 2))
        half_h = max(1, int(height * fraction / 2))
        cx, cy = width // 2, height // 2
        x0, x1 = max(0, cx - half_w), min(width, cx + half_w)
        y0, y1 = max(0, cy - half_h), min(height, cy + half_h)
        count = int(integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0])
        area = (x1 - x0) * (y1 - y0)
        return count / area if area else 0.0

    low, high = 0.0, 1.0
    for _ in range(20):
        mid = (low + high) / 2
        if opaque_ratio(mid) >= 0.99:
            low = mid
        else:
            high = mid
    return low


def effective_scale(scale: float, mask_rgba: np.ndarray, *, floor: float = 0.35) -> float:
    """設定の scale を、マスクの実効被覆で補正した値にする。

    「scale を 2.6 以上にすることを推奨」と人に判断させるのではなく、
    判断を挟まず補正する。透明部分が極端に多い画像で補正が暴走しないよう、
    被覆比には下限を設ける（それ以下は画像の選び直しを促すべき水準）。
    """
    coverage = effective_coverage(mask_rgba)
    return scale / max(coverage, floor)


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
