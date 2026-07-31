"""カットした区間を繋ぎ、顔を隠して 1 本に書き出す。

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

from .faces import FaceBox, FaceDetector
from .ffmpeg import find_ffmpeg, run
from .frames import FrameWriter, even, fit_filters, read_frames, scaled_size
from .ordering import Clip
from .overlay import MaskTracker, composite, load_mask
from .segments import CutPlan
from .timeline import Segment, Timeline, build_timeline

ProgressFn = Callable[[str, float], None]


@dataclass
class Geometry:
    width: int
    height: int
    fps: float


@dataclass
class RenderReport:
    """書き出し結果と、人が見返すべき箇所の記録。"""

    output: Path
    geometry: Geometry
    segments: List[Segment]
    duration: float = 0.0
    frames_total: int = 0
    frames_with_mask: int = 0
    frames_held: int = 0
    frames_no_face: List[float] = field(default_factory=list)
    low_confidence: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def mask_coverage(self) -> float:
        if self.frames_total == 0:
            return 0.0
        return self.frames_with_mask / self.frames_total


def choose_geometry(clips: Sequence[Clip]) -> Geometry:
    """出力の枠を決める。最も多い解像度・フレームレートに合わせる。"""
    sizes = collections.Counter((c.info.width, c.info.height) for c in clips)
    rates = collections.Counter(round(c.info.fps, 3) for c in clips)
    (width, height), _ = sizes.most_common(1)[0]
    fps, _ = rates.most_common(1)[0]
    return Geometry(width=even(width), height=even(height), fps=float(fps) or 30.0)


def _detect_scaled(
    detector: FaceDetector, frame: np.ndarray, detect_width: Optional[int]
) -> List[FaceBox]:
    """必要なら縮小して検出し、座標を元の大きさに戻す。"""
    height, width = frame.shape[:2]
    if not detect_width or detect_width >= width:
        return detector.detect(frame)
    small_w, small_h = scaled_size(width, height, detect_width)
    small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
    factor = width / float(small_w)
    return [box.scaled(factor) for box in detector.detect(small)]


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
    *,
    output_path: str | Path,
    mask_path: str | Path,
    detector: FaceDetector,
    mask_scale: float = 2.0,
    mask_offset_y: float = 0.0,
    hold_sec: float = 0.7,
    detect_width: Optional[int] = 640,
    detect_every_n_frames: int = 1,
    low_score_warn: float = 0.75,
    crf: int = 20,
    preset: str = "medium",
    pix_fmt: str = "yuv420p",
    audio_bitrate: str = "192k",
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
    temp_video = work / "video_masked.mp4"
    temp_audio = work / "audio_cut.m4a"

    report = RenderReport(output=output, geometry=geometry, segments=list(segments))
    total_expected = timeline.total_frames
    hold_frames = max(0, int(round(hold_sec * geometry.fps)))
    step = max(1, int(detect_every_n_frames))

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
            # カットの前後で場面が飛ぶので、追従は区間ごとに作り直す。
            tracker = MaskTracker(hold_frames=hold_frames)
            drawn: List[FaceBox] = []

            for local_index, frame in enumerate(frames):
                frame = frame.copy()  # ffmpeg のバッファは読み取り専用
                if local_index % step == 0:
                    boxes = _detect_scaled(detector, frame, detect_width)
                    drawn = tracker.update(boxes)
                    if not boxes and drawn:
                        report.frames_held += 1
                    for box in boxes:
                        if box.score < low_score_warn:
                            report.low_confidence.append(
                                (segment.out_start + local_index / geometry.fps, box.score)
                            )

                timestamp = segment.out_start + local_index / geometry.fps
                if drawn:
                    report.frames_with_mask += 1
                    for box in drawn:
                        composite(
                            frame,
                            mask_image,
                            box.expanded(
                                mask_scale,
                                (geometry.width, geometry.height),
                                offset_y=mask_offset_y,
                            ),
                        )
                else:
                    report.frames_no_face.append(round(timestamp, 3))

                writer.write(frame)
                report.frames_total += 1
                produced += 1
                if on_progress and produced % 30 == 0 and total_expected > 0:
                    on_progress("顔を隠して書き出し中", min(1.0, produced / total_expected))

    if on_progress:
        on_progress("音声を結合中", 1.0)
    _build_audio(segments, temp_audio, bitrate=audio_bitrate)
    _mux(temp_video, temp_audio, output)

    report.duration = report.frames_total / geometry.fps
    temp_video.unlink(missing_ok=True)
    temp_audio.unlink(missing_ok=True)
    try:
        work.rmdir()
    except OSError:
        pass
    return report
