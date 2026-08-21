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


# 精密トリムの前に置く粗いシークの余裕。キーフレーム間隔より十分大きくとる。
_COARSE_SEEK_MARGIN = 2.0


def read_frames(
    path: str | Path,
    *,
    out_width: int,
    out_height: int,
    fps: float,
    filters: Sequence[str] = (),
    start: Optional[float] = None,
    duration: Optional[float] = None,
    expected_frames: Optional[int] = None,
) -> tuple[FrameStream, Iterator[np.ndarray]]:
    """動画を BGR フレームの列として読み出す。

    出力の大きさは呼び出し側が filters と併せて決める。間引きや拡縮を ffmpeg
    側でやってしまうほうが、復号ごと省けて速い。

    時刻の精密さについて: -ss を -i の前に置くだけだと、シークの着地点と
    フィルタの位相が区間ごとに揺れ、出力フレームと素材時刻の対応が最大
    1 フレームずれる。人手修正の記録（素材時刻を指す）がその上に載るので、
    -ss は 2 秒手前への粗いシークに限定し、精密な切り出しは trim フィルタで行う。

    expected_frames を与えると、必ずその枚数を返す。足りない分は最終フレームの
    複製で埋め、余った分は読み捨てる。区間のフレーム数を round(duration×fps) で
    先に確定させる timeline の契約（音ズレ防止）の映像側の実装がこれ。
    """
    out_w, out_h = out_width, out_height

    cmd = [find_ffmpeg(), "-hide_banner", "-loglevel", "error"]
    pre_filters: list[str] = []
    if start is not None and start > 0:
        coarse = max(0.0, start - _COARSE_SEEK_MARGIN)
        if coarse > 0:
            cmd += ["-ss", f"{coarse:.6f}"]
        trim_start = start - coarse
        if duration is not None:
            pre_filters.append(f"trim=start={trim_start:.6f}:end={trim_start + duration:.6f}")
        else:
            pre_filters.append(f"trim=start={trim_start:.6f}")
        pre_filters.append("setpts=PTS-STARTPTS")
    elif duration is not None:
        pre_filters.append(f"trim=end={duration:.6f}")
        pre_filters.append("setpts=PTS-STARTPTS")
    cmd += ["-i", str(path)]
    all_filters = pre_filters + list(filters)
    if all_filters:
        cmd += ["-vf", ",".join(all_filters)]
    cmd += ["-an", "-sn", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]

    stream = FrameStream(width=out_w, height=out_h, fps=fps)

    def generate() -> Iterator[np.ndarray]:
        frame_bytes = out_w * out_h * 3
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdout is not None
        produced = 0
        last: Optional[np.ndarray] = None
        closed_cleanly = False
        try:
            while True:
                if expected_frames is not None and produced >= expected_frames:
                    proc.kill()
                    break
                buffer = proc.stdout.read(frame_bytes)
                if not buffer:
                    break
                if len(buffer) < frame_bytes:
                    # 端数は「たまたま短い動画」ではなく復号の失敗。黙って
                    # 捨てると以降の全フレームの時刻がずれるので、必ず落とす。
                    raise FFmpegError(
                        f"フレームが途中で切れました ({path}): "
                        f"{len(buffer)}/{frame_bytes} バイト"
                    )
                last = np.frombuffer(buffer, dtype=np.uint8).reshape(out_h, out_w, 3)
                produced += 1
                yield last

            # ループを自力で抜けた。プロセスを畳み、成否を検査してから枚数を揃える。
            killed = expected_frames is not None and produced >= expected_frames
            proc.stdout.close()
            stderr = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
            if proc.stderr:
                proc.stderr.close()
            code = proc.wait()
            closed_cleanly = True
            # 自分で止めた場合以外の異常終了は、stderr が空でも失敗として扱う。
            if not killed and code not in (0, None):
                detail = stderr.strip()[:500] or f"終了コード {code}"
                raise FFmpegError(f"フレーム読み出しに失敗しました ({path}): {detail}")
            if expected_frames is not None and produced < expected_frames:
                if last is None:
                    raise FFmpegError(f"フレームを 1 枚も読めませんでした ({path})")
                # 復号器の丸めで 1 枚足りないことがある。時間の権威はフレーム数の
                # 側なので、最終フレームを複製して枚数を合わせる。
                for _ in range(expected_frames - produced):
                    yield last
        finally:
            # 例外や消費側の中断で抜けた場合の畳み方。ここでは例外を出さない。
            if not closed_cleanly:
                proc.kill()
                if proc.stdout and not proc.stdout.closed:
                    proc.stdout.close()
                if proc.stderr and not proc.stderr.closed:
                    proc.stderr.close()
                proc.wait()

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
