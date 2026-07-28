"""ffmpeg のパイプ越しに生フレームを読み書きする。

OpenCV の VideoCapture で直接読むこともできるが、コーデックによって seek が
ずれる・fps 間引きが自前になる、という問題がある。復号は ffmpeg に任せ、
Python 側は BGR の生バイト列だけを受け取るほうが素直で速い。
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np

from .ffmpeg import FFmpegError, find_ffmpeg


def even(value: int) -> int:
    """H.264 の yuv420p は幅高とも偶数でなければならない。"""
    return value if value % 2 == 0 else value + 1


def scaled_size(width: int, height: int, target_width: Optional[int]) -> tuple[int, int]:
    """縦横比を保ったまま目標幅に合わせる。拡大はしない。"""
    if not target_width or target_width >= width:
        return width, height
    ratio = target_width / width
    return even(target_width), even(int(round(height * ratio)))


@dataclass
class FrameStream:
    """読み出し中のフレーム列の形。時刻計算に fps が要る。"""

    width: int
    height: int
    fps: float


def fit_filters(
    source_width: int,
    source_height: int,
    out_width: int,
    out_height: int,
    fps: Optional[float] = None,
) -> list[str]:
    """縦横比を保って所定の枠に収め、余白を黒で埋めるフィルタ列を組む。

    素材ごとに解像度や縦横比が違っても、1 本に繋ぐ以上は同じ枠に揃える必要が
    ある。切り取るのではなく余白で合わせる（画が欠けるほうが困る）。
    """
    filters: list[str] = []
    if fps:
        filters.append(f"fps={fps}")
    if (source_width, source_height) != (out_width, out_height):
        filters.append(
            f"scale={out_width}:{out_height}:force_original_aspect_ratio=decrease"
        )
        filters.append(
            f"pad={out_width}:{out_height}:(ow-iw)/2:(oh-ih)/2:color=black"
        )
    filters.append("setsar=1")
    return filters


def read_frames(
    path: str | Path,
    *,
    out_width: int,
    out_height: int,
    fps: float,
    filters: Sequence[str] = (),
    start: Optional[float] = None,
    duration: Optional[float] = None,
) -> tuple[FrameStream, Iterator[np.ndarray]]:
    """動画を BGR フレームの列として読み出す。

    出力の大きさは呼び出し側が filters と併せて決める。間引きや拡縮を ffmpeg
    側でやってしまうほうが、復号ごと省けて速い。
    """
    out_w, out_h = out_width, out_height

    cmd = [find_ffmpeg(), "-hide_banner", "-loglevel", "error"]
    # -ss を -i の前に置くと、ffmpeg が復号を飛ばしつつ再符号化時は正確に合わせる。
    if start is not None and start > 0:
        cmd += ["-ss", f"{start:.6f}"]
    cmd += ["-i", str(path)]
    if duration is not None:
        cmd += ["-t", f"{duration:.6f}"]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += ["-an", "-sn", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]

    stream = FrameStream(width=out_w, height=out_h, fps=fps)

    def generate() -> Iterator[np.ndarray]:
        frame_bytes = out_w * out_h * 3
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdout is not None
        try:
            while True:
                buffer = proc.stdout.read(frame_bytes)
                if not buffer or len(buffer) < frame_bytes:
                    break
                yield np.frombuffer(buffer, dtype=np.uint8).reshape(out_h, out_w, 3)
        finally:
            proc.stdout.close()
            stderr = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
            if proc.stderr:
                proc.stderr.close()
            code = proc.wait()
            # 途中で読むのをやめた場合は SIGPIPE 相当で落ちるため、それは無視する。
            if code not in (0, None) and "Broken pipe" not in stderr and stderr.strip():
                raise FFmpegError(f"フレーム読み出しに失敗しました ({path}): {stderr.strip()[:500]}")

    return stream, generate()


class FrameWriter:
    """BGR フレームを受け取って動画に符号化する。音声は持たない。"""

    def __init__(
        self,
        path: str | Path,
        *,
        width: int,
        height: int,
        fps: float,
        crf: int = 20,
        preset: str = "medium",
        pix_fmt: str = "yuv420p",
    ) -> None:
        self.path = Path(path)
        self.width = width
        self.height = height
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._proc = subprocess.Popen(
            [
                find_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "pipe:0",
                "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                "-pix_fmt", pix_fmt, str(self.path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise FFmpegError(
                f"フレームの大きさが違います: {frame.shape[1]}x{frame.shape[0]} "
                f"(期待 {self.width}x{self.height})"
            )
        assert self._proc.stdin is not None
        self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self._proc.stdin and not self._proc.stdin.closed:
            self._proc.stdin.close()
        stderr = self._proc.stderr.read().decode("utf-8", "replace") if self._proc.stderr else ""
        if self._proc.stderr:
            self._proc.stderr.close()
        if self._proc.wait() != 0:
            raise FFmpegError(f"書き出しに失敗しました ({self.path}): {stderr.strip()[:500]}")

    def __enter__(self) -> "FrameWriter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        if exc_info[0] is None:
            self.close()
        else:
            # 例外で抜けるときは書きかけを畳むだけにして、元の例外を隠さない。
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
            self._proc.wait()
