"""解析パス。素材の全フレームに顔検出を走らせ、結果を faces.jsonl に落とす。

書き出しから検出を切り離す（二段化）ことで得られるもの:
- 未来のフレームの検出結果を使ったギャップ埋め（tracks.py）
- 検出をやり直さずにマスクの大きさ・位置だけ変えて再書き出し
- 人手修正（faces_manual.jsonl）の受け皿
- カット判定用の 3fps 走査の廃止（ここが両方を兼ねる）

設計上の約束:
- キーはフレーム番号ではなく素材の PTS（秒）。fps=<素材fps> フィルタで CFR に
  正規化した上での i/fps を用いる
- 座標は素材のピクセル。縮小は ffmpeg 側で行い、Python では resize しない
- ファイル先頭にメタ行（モデル SHA・パラメータ・素材の mtime/size）を書き、
  どれかが変わったら検出し直す。「検出は 1 回きり」はこの規則があって初めて成立する
- クリップ間は完全に独立なので、multiprocessing で並列に回す
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .faces import FaceDetector, ensure_model
from .frames import read_frames, scaled_size
from .ordering import Clip

SCHEMA_VERSION = 2

# 場面変化とみなすフレーム間差分（0..255 の平均絶対差）。これを超えた境界では
# tracks.py が補間を止める。低すぎると普通の動きで千切れ、高すぎると転換を跨ぐ。
SCENE_DIFF_THRESHOLD = 28.0


@dataclass(frozen=True)
class DetectParams:
    """検出パスの設定。ハッシュがキャッシュの鍵になる。"""

    score_threshold: float
    detect_width: int
    model_sha: str

    def digest(self) -> str:
        payload = json.dumps(
            {
                "version": SCHEMA_VERSION,
                "score_threshold": round(self.score_threshold, 4),
                "detect_width": self.detect_width,
                "model_sha": self.model_sha,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class FrameRecord:
    """1 フレームぶんの検出結果。boxes は素材ピクセルの [x, y, w, h, score]。"""

    pts: float
    boxes: List[List[float]]
    landmarks: List[List[float]]
    scene: float = 0.0


@dataclass
class ClipDetections:
    """1 素材ぶんの読み込み結果。"""

    clip_path: Path
    fps: float
    width: int
    height: int
    frames_total: int
    records: Dict[int, FrameRecord]  # フレーム番号 → 記録（検出ゼロの行は持たない）

    def grid_pts(self, index: int) -> float:
        return index / self.fps

    def scene_breaks(self, threshold: float = SCENE_DIFF_THRESHOLD) -> List[int]:
        """場面変化のフレーム番号（そのフレームと前フレームの間が境界）。"""
        return sorted(
            index for index, rec in self.records.items() if rec.scene >= threshold
        )


def detections_path(work_dir: str | Path, clip_path: str | Path) -> Path:
    """faces.jsonl の置き場。output/ の外の作業フォルダに置く。

    検出結果は「いつ誰がどこにいたか」の座標時系列＝個人情報でもあるので、
    人に渡す output/ には置かない。一方で消すと再検出になるため、確認後に
    消す運用の対象からも外す。
    """
    stem = Path(clip_path).stem
    digest = hashlib.sha256(str(Path(clip_path)).encode("utf-8")).hexdigest()[:8]
    return Path(work_dir) / "detect" / f"{stem}.{digest}.jsonl"


def _clip_signature(path: Path) -> Tuple[int, int]:
    stat = path.stat()
    return int(stat.st_mtime), int(stat.st_size)


def is_fresh(jsonl_path: Path, clip_path: Path, params: DetectParams) -> bool:
    """既存の faces.jsonl がこの素材・この設定の結果として使えるか。"""
    if not jsonl_path.is_file():
        return False
    try:
        with jsonl_path.open("r", encoding="utf-8") as handle:
            meta = json.loads(handle.readline())
    except (OSError, json.JSONDecodeError):
        return False
    if meta.get("type") != "meta" or meta.get("schema") != SCHEMA_VERSION:
        return False
    if meta.get("params") != params.digest():
        return False
    mtime, size = _clip_signature(clip_path)
    return meta.get("clip_mtime") == mtime and meta.get("clip_size") == size


def detect_clip(
    clip_path: Path,
    out_path: Path,
    *,
    source_width: int,
    source_height: int,
    source_fps: float,
    duration: float,
    params: DetectParams,
    model_path: Path,
) -> Path:
    """1 素材の全フレームを検出して JSONL に書く。ワーカプロセスからも呼ばれる。"""
    detector = FaceDetector(model_path, score_threshold=params.score_threshold)

    small_w, small_h = scaled_size(source_width, source_height, params.detect_width)
    factor = source_width / float(small_w)
    filters = [f"fps={source_fps}"]
    if (small_w, small_h) != (source_width, source_height):
        filters.append(f"scale={small_w}:{small_h}")

    _, frames = read_frames(
        clip_path,
        out_width=small_w,
        out_height=small_h,
        fps=source_fps,
        filters=filters,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp = out_path.with_suffix(".tmp")
    mtime, size = _clip_signature(clip_path)
    frames_total = 0
    prev_small: Optional[np.ndarray] = None

    with temp.open("w", encoding="utf-8") as handle:
        meta = {
            "type": "meta",
            "schema": SCHEMA_VERSION,
            "params": params.digest(),
            "clip": clip_path.name,
            "clip_mtime": mtime,
            "clip_size": size,
            "width": source_width,
            "height": source_height,
            "fps": source_fps,
            "detect_width": params.detect_width,
            "model_sha": params.model_sha,
        }
        handle.write(json.dumps(meta, ensure_ascii=False) + "\n")

        for index, frame in enumerate(frames):
            frames_total += 1
            # 場面変化の検出。8x8 の縮約グレイ同士の平均絶対差で足りる。
            tiny = frame[:: max(1, small_h // 8), :: max(1, small_w // 8)].mean(axis=2)
            scene = 0.0
            if prev_small is not None and tiny.shape == prev_small.shape:
                scene = float(np.abs(tiny - prev_small).mean())
            prev_small = tiny

            boxes = detector.detect(frame)
            if not boxes and scene < SCENE_DIFF_THRESHOLD:
                continue  # 空行は書かない。無いこと＝検出ゼロ

            record = {
                "f": index,
                "pts": round(index / source_fps, 6),
                "boxes": [
                    [round(b.x * factor, 2), round(b.y * factor, 2),
                     round(b.w * factor, 2), round(b.h * factor, 2),
                     round(b.score, 4)]
                    for b in boxes
                ],
            }
            marks = [
                [round(px * factor, 2), round(py * factor, 2)]
                for b in boxes if b.landmarks
                for px, py in b.landmarks
            ]
            if marks:
                record["marks"] = marks
            if scene >= SCENE_DIFF_THRESHOLD:
                record["scene"] = round(scene, 2)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        tail = {"type": "end", "frames_total": frames_total}
        handle.write(json.dumps(tail) + "\n")

    temp.replace(out_path)
    return out_path


def _worker(args: tuple) -> Tuple[str, float]:
    """並列実行の入口。cv2 のスレッドを絞り、プロセス間で取り合わない。"""
    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:  # noqa: BLE001 — スレッド設定の失敗は致命ではない
        pass
    (clip_path, out_path, width, height, fps, duration, params, model_path) = args
    started = time.monotonic()
    detect_clip(
        Path(clip_path), Path(out_path),
        source_width=width, source_height=height,
        source_fps=fps, duration=duration,
        params=params, model_path=Path(model_path),
    )
    return str(clip_path), time.monotonic() - started


def detect_clips(
    clips: Sequence[Clip],
    *,
    work_dir: str | Path,
    score_threshold: float,
    detect_width: int,
    face_model: Optional[str | Path] = None,
    jobs: Optional[int] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Dict[Path, Path]:
    """全素材を検出する。済んでいるものは飛ばす。素材パス → jsonl の対応を返す。"""
    from .faces import MODEL_SHA256, _sha256

    model_path = ensure_model(face_model)
    model_sha = MODEL_SHA256 if face_model is None else _sha256(Path(model_path))
    params = DetectParams(
        score_threshold=float(score_threshold),
        detect_width=int(detect_width),
        model_sha=model_sha,
    )

    result: Dict[Path, Path] = {}
    pending: List[tuple] = []
    for clip in clips:
        out_path = detections_path(work_dir, clip.path)
        result[clip.path] = out_path
        if is_fresh(out_path, clip.path, params):
            continue
        pending.append((
            str(clip.path), str(out_path),
            clip.info.width, clip.info.height, clip.info.fps, clip.info.duration,
            params, str(model_path),
        ))

    if not pending:
        return result

    if jobs is None:
        jobs = max(1, min(len(pending), (os.cpu_count() or 2) - 1, 8))

    if on_progress:
        on_progress(f"顔検出 {len(pending)} 本（並列 {jobs}）")

    if jobs <= 1 or len(pending) <= 1:
        for args in pending:
            path, took = _worker(args)
            if on_progress:
                on_progress(f"検出済み {Path(path).name} ({took:.1f}s)")
    else:
        with multiprocessing.Pool(processes=jobs) as pool:
            for path, took in pool.imap_unordered(_worker, pending):
                if on_progress:
                    on_progress(f"検出済み {Path(path).name} ({took:.1f}s)")
    return result


def load_detections(jsonl_path: str | Path) -> ClipDetections:
    """faces.jsonl を読み込む。end 行が無ければ書きかけとして拒否する。"""
    path = Path(jsonl_path)
    records: Dict[int, FrameRecord] = {}
    meta: Optional[dict] = None
    frames_total = -1
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            kind = data.get("type")
            if kind == "meta":
                meta = data
                continue
            if kind == "end":
                frames_total = int(data["frames_total"])
                continue
            marks = data.get("marks") or []
            records[int(data["f"])] = FrameRecord(
                pts=float(data["pts"]),
                boxes=[list(map(float, b)) for b in data.get("boxes", [])],
                landmarks=[list(map(float, m)) for m in marks],
                scene=float(data.get("scene", 0.0)),
            )
    if meta is None or frames_total < 0:
        raise ValueError(f"検出結果が壊れています（書きかけの可能性）: {path}")
    return ClipDetections(
        clip_path=path,
        fps=float(meta["fps"]),
        width=int(meta["width"]),
        height=int(meta["height"]),
        frames_total=frames_total,
        records=records,
    )
