"""ffmpeg の silencedetect フィルタで無音区間を取り出す。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from .ffmpeg import find_ffmpeg, run
from .segments import Interval, merge

_START_RE = re.compile(r"silence_start:\s*(-?[\d.]+)")
_END_RE = re.compile(r"silence_end:\s*(-?[\d.]+)")


def detect_silence(
    path: str | Path,
    *,
    duration: float,
    has_audio: bool,
    noise_db: float = -32.0,
    min_silence_sec: float = 0.8,
) -> List[Interval]:
    """無音とみなせる区間の一覧を返す。

    音声トラックが無い素材では空を返す。「全部無音」と解釈すると、mode=any の
    ときに素材が丸ごと消えてしまい、事故にしかならない。
    """
    if not has_audio:
        return []

    proc = run(
        [
            find_ffmpeg(), "-hide_banner", "-nostats", "-i", str(path),
            "-af", f"silencedetect=noise={noise_db}dB:d={min_silence_sec}",
            "-f", "null", "-",
        ],
        check=False,
    )
    text = proc.stderr or ""

    intervals: List[Interval] = []
    pending: float | None = None
    for line in text.splitlines():
        if "silencedetect" not in line:
            continue
        if (m := _START_RE.search(line)):
            pending = max(0.0, float(m.group(1)))
        if (m := _END_RE.search(line)):
            end = min(duration, float(m.group(1)))
            start = pending if pending is not None else 0.0
            if end > start:
                intervals.append((start, end))
            pending = None

    # 末尾が無音のまま終わると silence_end が出ないので、尺の終わりで閉じる。
    if pending is not None and duration > pending:
        intervals.append((pending, duration))

    return merge(intervals)
