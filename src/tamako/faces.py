"""顔検出。OpenCV の YuNet（軽量 DNN 検出器）を使う。

顔隠しは「検出漏れ＝顔バレ」なので、精度より取りこぼしの少なさを優先する。
閾値は低め、箱は大きめ、見失っても直前の位置を保持する、という方針にしてある。
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .frames import read_frames, scaled_size

MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


class FaceModelError(RuntimeError):
    """顔検出モデルを用意できなかった。"""


def cache_dir() -> Path:
    """モデルの置き場。OS ごとの標準的なキャッシュ位置に従う。"""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "tamako"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model(explicit: Optional[str | Path] = None) -> Path:
    """モデルファイルの場所を返す。無ければ取得し、必ずハッシュを検証する。"""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FaceModelError(f"指定されたモデルが見つかりません: {path}")
        return path

    target = cache_dir() / MODEL_FILENAME
    if target.is_file() and _sha256(target) == MODEL_SHA256:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".download")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=120) as response, temp.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
    except Exception as exc:  # noqa: BLE001 — 原因を問わず案内を出したい
        temp.unlink(missing_ok=True)
        raise FaceModelError(
            "顔検出モデルを取得できませんでした。ネットワークに繋がらない環境では、\n"
            f"  {MODEL_URL}\n"
            f"を手動で入手して {target} に置くか、--face-model で場所を指定してください。\n"
            f"（元の例外: {exc}）"
        ) from exc

    actual = _sha256(temp)
    if actual != MODEL_SHA256:
        temp.unlink(missing_ok=True)
        raise FaceModelError(
            f"取得したモデルのハッシュが一致しません (期待 {MODEL_SHA256}, 実際 {actual})"
        )
    temp.replace(target)
    return target


@dataclass(frozen=True)
class FaceBox:
    """顔の位置。座標は検出したフレームの画素単位。

    landmarks は右目・左目・鼻先・右口角・左口角の 5 点（YuNet が箱と一緒に
    返す）。両目から面内回転が、鼻と両目の相対位置からヨーが粗く求まるので、
    トラックの照合や箱の非対称な拡大の材料になる。無い場合は None。
    """

    x: float
    y: float
    w: float
    h: float
    score: float
    landmarks: Optional[tuple[tuple[float, float], ...]] = None

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    def scaled(self, factor: float) -> "FaceBox":
        """縮小して検出した結果を元の解像度に戻す。"""
        marks = None
        if self.landmarks:
            marks = tuple((px * factor, py * factor) for px, py in self.landmarks)
        return FaceBox(
            self.x * factor, self.y * factor,
            self.w * factor, self.h * factor, self.score, marks,
        )

    def expanded(self, scale: float, offset_y: float = 0.0) -> "FaceBox":
        """中心を保ったまま拡大する。画面外にはみ出してよい。

        以前ここにあった画面内クランプは削除した。composite() が画面外を
        正しく切り取るので不要であり、しかもクランプは画面端でマスクを
        平行移動（左・上端）または横方向に圧縮（右・下端）させて、
        顔がフレームに出入りする瞬間＝いちばん漏れやすい箇所で顔を露出させていた。

        offset_y は顔の高さに対する割合で上下にずらす（負で上）。検出枠は
        目鼻口が中心なので、髪の量によっては少し上げたほうが収まりがよい。
        """
        cx, cy = self.center
        cy += self.h * offset_y
        new_w, new_h = self.w * scale, self.h * scale
        return FaceBox(cx - new_w / 2, cy - new_h / 2, new_w, new_h, self.score)


class FaceDetector:
    """YuNet の薄い包み。入力サイズが変わるたびに設定し直す必要がある。"""

    def __init__(self, model_path: str | Path, *, score_threshold: float = 0.5, nms_threshold: float = 0.3) -> None:
        self._detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320),
            score_threshold=float(score_threshold),
            nms_threshold=float(nms_threshold),
        )
        self._size: tuple[int, int] | None = None
        self.score_threshold = float(score_threshold)

    def detect(self, frame: np.ndarray) -> List[FaceBox]:
        height, width = frame.shape[:2]
        if self._size != (width, height):
            self._detector.setInputSize((width, height))
            self._size = (width, height)
        _, raw = self._detector.detect(frame)
        if raw is None:
            return []
        boxes: List[FaceBox] = []
        for row in raw:
            # row は 15 要素: x, y, w, h, ランドマーク 5 点 (x,y)×5, スコア
            marks = None
            if len(row) >= 15:
                marks = tuple(
                    (float(row[4 + i * 2]), float(row[5 + i * 2])) for i in range(5)
                )
            boxes.append(
                FaceBox(float(row[0]), float(row[1]), float(row[2]), float(row[3]),
                        float(row[-1]), marks)
            )
        return boxes


def scan_face_presence(
    video_path: str | Path,
    *,
    source_width: int,
    source_height: int,
    source_fps: float,
    detector: FaceDetector,
    sample_fps: float = 3.0,
    detect_width: Optional[int] = 640,
) -> List[float]:
    """顔が写っている時刻（標本点）の一覧を返す。カット判定に使う粗い走査。

    毎フレーム見る必要はないので ffmpeg 側で間引く。復号ごと省けるため、
    長尺でもここは軽い。
    """
    out_w, out_h = scaled_size(source_width, source_height, detect_width)
    filters = [f"fps={sample_fps}"]
    if (out_w, out_h) != (source_width, source_height):
        filters.append(f"scale={out_w}:{out_h}")

    _, frames = read_frames(
        video_path,
        out_width=out_w,
        out_height=out_h,
        fps=sample_fps,
        filters=filters,
    )
    times: List[float] = []
    for index, frame in enumerate(frames):
        if detector.detect(frame):
            times.append(index / sample_fps)
    return times
