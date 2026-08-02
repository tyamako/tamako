"""カットした区間を繋ぎ、顔を隠して 1 本に書き出す。

検出はここではしない。`detect` が素材時間軸で全フレーム検出し、`tracks` が
途切れを埋めた結果を受け取って、貼るだけにしてある。おかげで
「マスクの大きさだけ変えて出し直す」が再検出なしでできる。

映像は Python 側でフレームごとに合成する必要があるため生フレームで扱うが、
音声は ffmpeg のフィルタだけで完結するので触らない。映像の符号化は 1 回で
済ませてある（切ってから重ねる、で 2 回符号化すると画質と時間を損なう）。
"""

from __future__ import annotations

import collections
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .ffmpeg import find_ffmpeg, run
from .frames import FrameWriter, even, fit_filters, read_frames
from .ordering import Clip
from .overlay import composite, effective_scale, load_mask
from .segments import CutPlan
from .timeline import Segment, Timeline, build_timeline
from .tracks import SOURCE_DETECTED, PlacedBox, TrackedClip

ProgressFn = Callable[[str, float], None]


@dataclass
class Geometry:
    width: int
    height: int
    fps: float


class IntervalCollector:
    """フレーム単位の出来事を区間（サイト）に畳む。

    時刻を 1 個ずつリストに積むと、人が写っていない 10 分の素材で 18000 個の
    float になり、報告としても読めない。人が見る単位は区間なので、連続する
    フレームはここでまとめる。
    """

    def __init__(self, fps: float) -> None:
        self._fps = fps
        self._gap = 1.5 / fps  # 1 フレーム強の途切れは同じ区間とみなす
        self.intervals: List[List[float]] = []

    def add(self, timestamp: float) -> None:
        if self.intervals and timestamp - self.intervals[-1][1] <= self._gap:
            self.intervals[-1][1] = timestamp
        else:
            self.intervals.append([timestamp, timestamp])

    def as_tuples(self) -> List[Tuple[float, float]]:
        return [(s, e + 1.0 / self._fps) for s, e in self.intervals]


@dataclass
class RenderReport:
    """書き出し結果と、人が見返すべき箇所の記録。"""

    output: Path
    geometry: Geometry
    segments: List[Segment]
    duration: float = 0.0
    frames_total: int = 0
    frames_with_mask: int = 0
    frames_estimated: int = 0
    # いずれも出力時刻の区間。フレーム単位の生記録は持たない。
    no_face_intervals: List[Tuple[float, float]] = field(default_factory=list)
    uncertain_intervals: List[Tuple[float, float]] = field(default_factory=list)
    effective_mask_scale: float = 0.0

    @property
    def mask_coverage(self) -> float:
        if self.frames_total == 0:
            return 0.0
        return self.frames_with_mask / self.frames_total

    @property
    def longest_no_face_sec(self) -> float:
        return max((e - s for s, e in self.no_face_intervals), default=0.0)


def choose_geometry(clips: Sequence[Clip]) -> Geometry:
    """出力の枠を決める。最も多い解像度・フレームレートに合わせる。"""
    sizes = collections.Counter((c.info.width, c.info.height) for c in clips)
    rates = collections.Counter(round(c.info.fps, 3) for c in clips)
    (width, height), _ = sizes.most_common(1)[0]
    fps, _ = rates.most_common(1)[0]
    return Geometry(width=even(width), height=even(height), fps=float(fps) or 30.0)


def source_to_output(
    src_w: int, src_h: int, out_w: int, out_h: int
) -> Tuple[float, float, float]:
    """素材ピクセル → 出力ピクセル の変換 (倍率, x オフセット, y オフセット)。

    fit_filters が縦横比を保って縮小し、余白を中央寄せで足すので、その逆算。
    検出結果は素材座標で持ってあるため、描く直前にここを通す。
    """
    if src_w <= 0 or src_h <= 0:
        return 1.0, 0.0, 0.0
    ratio = min(out_w / src_w, out_h / src_h)
    scaled_w, scaled_h = src_w * ratio, src_h * ratio
    return ratio, (out_w - scaled_w) / 2.0, (out_h - scaled_h) / 2.0


def _place(box: PlacedBox, ratio: float, dx: float, dy: float,
           scale: float, offset_y: float) -> Tuple[float, float, float, float]:
    """箱を出力座標に移し、マスクの倍率と上下補正を掛けた矩形にする。"""
    cx = (box.x + box.w / 2) * ratio + dx
    cy = (box.y + box.h / 2) * ratio + dy
    w, h = box.w * ratio * scale, box.h * ratio * scale
    cy += box.h * ratio * offset_y
    return cx - w / 2, cy - h / 2, w, h


def _build_audio(
    segments: Sequence[Segment], out_path: Path, *, bitrate: str = "192k"
) -> None:
    """区間の音声を切り出して繋ぐ。音の無い素材は無音で埋める。

    トリム長は必ず「確定したフレーム数 ÷ fps」から取る。素材の秒数 (end-start)
    で切ると、映像側（フレーム数の連結）との差が区間ごとに累積して音がずれる。
    """
    inputs: List[Path] = []
    index_of: Dict[Path, int] = {}
    for segment in segments:
        path = segment.clip.path
        if path not in index_of:
            index_of[path] = len(inputs)
            inputs.append(path)

    lines: List[str] = []
    labels: List[str] = []
    for i, segment in enumerate(segments):
        label = f"a{i}"
        labels.append(f"[{label}]")
        if segment.clip.info.has_audio:
            source = index_of[segment.clip.path]
            lines.append(
                f"[{source}:a]atrim=start={segment.start:.6f}"
                f":end={segment.start + segment.duration:.6f},"
                f"asetpts=N/SR/TB,aresample=48000,"
                f"aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"apad,atrim=duration={segment.duration:.6f}[{label}]"
            )
        else:
            lines.append(
                f"anullsrc=r=48000:cl=stereo,atrim=duration={segment.duration:.6f},"
                f"asetpts=N/SR/TB[{label}]"
            )
    lines.append(f"{''.join(labels)}concat=n={len(segments)}:v=0:a=1[out]")

    # 区間数が多いとコマンド行が長くなりすぎる（Windows は特に厳しい）ので
    # フィルタは必ずファイル経由で渡す。
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
        handle.write(";\n".join(lines))
        script_path = Path(handle.name)

    try:
        cmd = [find_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error"]
        for path in inputs:
            cmd += ["-i", str(path)]
        cmd += [
            "-filter_complex_script", str(script_path),
            "-map", "[out]", "-c:a", "aac", "-b:a", bitrate, str(out_path),
        ]
        run(cmd)
    finally:
        script_path.unlink(missing_ok=True)


def _mux(video: Path, audio: Path, out_path: Path) -> None:
    """映像と音声を 1 つのファイルにまとめる。どちらも再符号化しない。"""
    run([
        find_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "copy",
        "-movflags", "+faststart", "-shortest", str(out_path),
    ])


def render(
    clip_plans: Sequence[Tuple[Clip, CutPlan]],
    tracked: Dict[Path, TrackedClip],
    *,
    output_path: str | Path,
    mask_path: str | Path,
    mask_scale: float = 2.0,
    mask_offset_y: float = 0.0,
    crf: int = 20,
    preset: str = "medium",
    pix_fmt: str = "yuv420p",
    audio_bitrate: str = "192k",
    diagnostic: bool = False,
    draw_log: Optional[Path] = None,
    work_dir: Optional[Path] = None,
    on_progress: Optional[ProgressFn] = None,
) -> RenderReport:
    """カット・顔隠し・音声結合を行い、1 本の動画として書き出す。"""
    geometry = choose_geometry([clip for clip, _ in clip_plans])
    timeline = build_timeline(clip_plans, geometry.fps)
    segments = timeline.segments
    if not segments:
        raise ValueError(
            "残す区間がひとつもありません。cut.mode を 'both' にする、"
            "silence_db を下げる、min_keep_sec を小さくする、などを試してください。"
        )

    mask_image = load_mask(mask_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    work = Path(work_dir) if work_dir else output.parent / ".tamako_work"
    work.mkdir(parents=True, exist_ok=True)
    temp_video = work / f"video_{output.stem}.mp4"
    temp_audio = work / f"audio_{output.stem}.m4a"

    report = RenderReport(output=output, geometry=geometry, segments=list(segments))
    total_expected = timeline.total_frames

    # マスクの透明部分（丸・星型ステッカーの角）から顔が出ないよう、
    # 実効被覆で scale を自動補正する。人に判断させない。
    scale_eff = effective_scale(mask_scale, mask_image)
    report.effective_mask_scale = scale_eff

    no_face = IntervalCollector(geometry.fps)
    uncertain = IntervalCollector(geometry.fps)
    log_handle = draw_log.open("w", encoding="utf-8") if draw_log else None

    try:
        with FrameWriter(
            temp_video,
            width=geometry.width,
            height=geometry.height,
            fps=geometry.fps,
            crf=crf,
            preset=preset,
            pix_fmt=pix_fmt,
        ) as writer:
            produced = 0
            for segment in segments:
                filters = fit_filters(
                    segment.clip.info.width,
                    segment.clip.info.height,
                    geometry.width,
                    geometry.height,
                    fps=geometry.fps,
                )
                _, frames = read_frames(
                    segment.clip.path,
                    out_width=geometry.width,
                    out_height=geometry.height,
                    fps=geometry.fps,
                    filters=filters,
                    start=segment.start,
                    duration=segment.duration,
                    expected_frames=segment.frames,
                )
                clip_tracks = tracked.get(segment.clip.path)
                ratio, dx, dy = source_to_output(
                    segment.clip.info.width, segment.clip.info.height,
                    geometry.width, geometry.height,
                )

                for local_index, frame in enumerate(frames):
                    frame = frame.copy()  # ffmpeg のバッファは読み取り専用
                    out_index = segment.out_start_frame + local_index
                    timestamp = out_index / geometry.fps
                    pts = segment.source_pts(local_index)
                    boxes = clip_tracks.at_pts(pts) if clip_tracks else []

                    if boxes:
                        report.frames_with_mask += 1
                        if any(b.source != SOURCE_DETECTED for b in boxes):
                            report.frames_estimated += 1
                        if any(b.uncertain for b in boxes):
                            uncertain.add(timestamp)
                    else:
                        no_face.add(timestamp)

                    for box in boxes:
                        x, y, w, h = _place(box, ratio, dx, dy, scale_eff, mask_offset_y)
                        if diagnostic:
                            _draw_diagnostic(frame, mask_image, box, x, y, w, h)
                        else:
                            composite(frame, mask_image,
                                      _Rect(x=x, y=y, w=w, h=h))

                    if log_handle is not None:
                        parts = ";".join(
                            f"{b.track_id},{b.source},{b.x:.1f},{b.y:.1f},{b.w:.1f},{b.h:.1f}"
                            for b in boxes
                        )
                        log_handle.write(
                            f"{out_index}\t{segment.clip.path.name}\t{pts:.6f}\t{parts}\n"
                        )

                    writer.write(frame)
                    report.frames_total += 1
                    produced += 1
                    if on_progress and produced % 30 == 0 and total_expected > 0:
                        on_progress("顔を隠して書き出し中", min(1.0, produced / total_expected))
    finally:
        if log_handle is not None:
            log_handle.close()

    if on_progress:
        on_progress("音声を結合中", 1.0)
    _build_audio(segments, temp_audio, bitrate=audio_bitrate)
    _mux(temp_video, temp_audio, output)

    report.duration = report.frames_total / geometry.fps
    report.no_face_intervals = no_face.as_tuples()
    report.uncertain_intervals = uncertain.as_tuples()
    temp_video.unlink(missing_ok=True)
    temp_audio.unlink(missing_ok=True)
    try:
        work.rmdir()
    except OSError:
        pass
    return report


@dataclass(frozen=True)
class _Rect:
    """composite() が必要とする最小限の形（x, y, w, h）。"""

    x: float
    y: float
    w: float
    h: float


# 由来ごとの色（BGR）。診断モードで「なぜここに箱があるか」を見えるようにする。
_DIAGNOSTIC_COLORS = {
    "detected": (0, 220, 0),
    "dilate": (0, 200, 200),
    "interp": (0, 140, 255),
    "extrap": (0, 0, 255),
    "manual": (255, 0, 255),
}


def _draw_diagnostic(frame: np.ndarray, mask_rgba: np.ndarray, box: PlacedBox,
                     x: float, y: float, w: float, h: float) -> None:
    """実際に隠れる範囲（マスクの不透明画素）を半透明で塗り、由来を書く。

    箱を塗ってはいけない。隠しているのは箱ではなく PNG の不透明画素なので、
    箱を見せると「覆えている」と誤解する。丸いステッカーの角は透明で、
    そこは素顔が出る——それが見えることがこのモードの目的。
    """
    color = _DIAGNOSTIC_COLORS.get(box.source, (255, 255, 255))
    # マスクのアルファをそのまま使い、色だけ由来の色に差し替えたものを合成する。
    tinted = mask_rgba.copy()
    tinted[:, :, 0] = color[0]
    tinted[:, :, 1] = color[1]
    tinted[:, :, 2] = color[2]
    tinted[:, :, 3] = (tinted[:, :, 3].astype(np.float32) * 0.55).astype(np.uint8)
    composite(frame, tinted, _Rect(x=x, y=y, w=w, h=h))

    x0, y0 = int(round(x)), int(round(y))
    x1, y1 = int(round(x + w)), int(round(y + h))
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 1)
    label = box.source[:6] + ("!" if box.uncertain else "")
    cv2.putText(frame, label, (x0 + 3, max(12, y0 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
