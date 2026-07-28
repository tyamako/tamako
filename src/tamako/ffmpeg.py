"""ffmpeg / ffprobe の場所探しと実行、および素材のメタデータ取得。

ffprobe があればそれを使う。無い環境（pip の imageio-ffmpeg は ffmpeg しか
同梱しない）では `ffmpeg -i` の標準エラー出力を解析して同じ情報を取り出す。
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


class FFmpegError(RuntimeError):
    """ffmpeg / ffprobe の呼び出しが失敗した。"""


@lru_cache(maxsize=1)
def find_ffmpeg() -> str:
    """ffmpeg の実行パス。PATH 上のものを優先し、無ければ同梱版を使う。"""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise FFmpegError(
            "ffmpeg が見つかりません。`pip install imageio-ffmpeg` を実行するか、"
            "ffmpeg を PATH の通った場所に置いてください。"
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


@lru_cache(maxsize=1)
def find_ffprobe() -> Optional[str]:
    """ffprobe の実行パス。無ければ None（呼び出し側が代替経路に落ちる）。"""
    return shutil.which("ffprobe")


def run(cmd: Sequence[str], *, capture: bool = True, check: bool = True) -> subprocess.CompletedProcess:
    """外部コマンドを実行する。失敗時は末尾のログを添えて例外にする。"""
    proc = subprocess.run(
        list(cmd),
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-15:]
        raise FFmpegError(
            f"コマンドが失敗しました (終了コード {proc.returncode}):\n"
            f"  {' '.join(cmd[:3])} ...\n" + "\n".join("  " + line for line in tail)
        )
    return proc


@dataclass
class MediaInfo:
    """1 本の素材について、並べ替えと解析に必要な最小限の情報。"""

    path: Path
    duration: float
    fps: float
    width: int
    height: int
    has_audio: bool
    creation_time: Optional[_dt.datetime] = None
    # 撮影時刻をどこから得たか。順序を人が検算できるように残す。
    time_source: str = "unknown"

    @property
    def frame_count_estimate(self) -> int:
        return int(round(self.duration * self.fps))


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)")
_VIDEO_RE = re.compile(r"Stream #\d+:\d+.*?:\s*Video:.*?(\d{2,5})x(\d{2,5})")
_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*fps")
_AUDIO_RE = re.compile(r"Stream #\d+:\d+.*?:\s*Audio:")
_CREATION_RE = re.compile(r"creation_time\s*:\s*(\S+)")


def _parse_iso_time(value: str) -> Optional[_dt.datetime]:
    text = value.strip().replace("Z", "+00:00")
    for candidate in (text, text.split(".")[0]):
        try:
            parsed = _dt.datetime.fromisoformat(candidate)
        except ValueError:
            continue
        # タイムゾーンの有無が混ざると比較できないので naive UTC に揃える。
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(_dt.timezone.utc).replace(tzinfo=None)
        return parsed
    return None


def _probe_with_ffprobe(path: Path, ffprobe: str) -> MediaInfo:
    proc = run([
        ffprobe, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ])
    payload: Dict[str, Any] = json.loads(proc.stdout or "{}")
    streams: List[Dict[str, Any]] = payload.get("streams", [])
    fmt: Dict[str, Any] = payload.get("format", {})

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise FFmpegError(f"映像トラックがありません: {path.name}")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    duration = float(fmt.get("duration") or video.get("duration") or 0.0)
    fps = _parse_rational(video.get("avg_frame_rate")) or _parse_rational(video.get("r_frame_rate")) or 30.0

    created = None
    source = "none"
    for tags in (video.get("tags") or {}, fmt.get("tags") or {}):
        for key in ("creation_time", "com.apple.quicktime.creationdate", "date"):
            if tags.get(key):
                created = _parse_iso_time(str(tags[key]))
                if created:
                    source = f"metadata:{key}"
                    break
        if created:
            break

    return MediaInfo(
        path=path,
        duration=duration,
        fps=fps,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        has_audio=has_audio,
        creation_time=created,
        time_source=source,
    )


def _parse_rational(value: Any) -> Optional[float]:
    if not value or not isinstance(value, str) or "/" not in value:
        return None
    num, _, den = value.partition("/")
    try:
        numerator, denominator = float(num), float(den)
    except ValueError:
        return None
    if denominator == 0:
        return None
    result = numerator / denominator
    return result if result > 0 else None


def _probe_with_ffmpeg(path: Path, ffmpeg: str) -> MediaInfo:
    """ffprobe が無い環境向け。`ffmpeg -i` の診断出力を読む。"""
    proc = run([ffmpeg, "-hide_banner", "-i", str(path)], check=False)
    text = proc.stderr or ""

    duration = 0.0
    if (m := _DURATION_RE.search(text)):
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))

    width = height = 0
    fps = 30.0
    if (m := _VIDEO_RE.search(text)):
        width, height = int(m.group(1)), int(m.group(2))
        line_end = text.find("\n", m.end())
        video_line = text[m.start(): line_end if line_end != -1 else len(text)]
        if (f := _FPS_RE.search(video_line)):
            fps = float(f.group(1)) or 30.0

    if width == 0 or height == 0:
        raise FFmpegError(f"映像トラックを読み取れません: {path.name}")

    created = None
    source = "none"
    if (m := _CREATION_RE.search(text)):
        created = _parse_iso_time(m.group(1))
        if created:
            source = "metadata:creation_time"

    return MediaInfo(
        path=path,
        duration=duration,
        fps=fps,
        width=width,
        height=height,
        has_audio=bool(_AUDIO_RE.search(text)),
        creation_time=created,
        time_source=source,
    )


def probe(path: str | Path) -> MediaInfo:
    """素材のメタデータを取得する。ffprobe があれば使い、無ければ ffmpeg で代用。"""
    target = Path(path)
    if not target.is_file():
        raise FFmpegError(f"ファイルが見つかりません: {target}")

    ffprobe = find_ffprobe()
    info = _probe_with_ffprobe(target, ffprobe) if ffprobe else _probe_with_ffmpeg(target, find_ffmpeg())

    if info.duration <= 0:
        raise FFmpegError(f"再生時間を取得できませんでした: {target.name}")
    return info
