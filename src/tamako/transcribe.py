"""収録した音声を文字起こしして字幕（SRT）にする。

Whisper（faster-whisper 実装）を使う。日本語は語の切れ目に空白が無いので、
折り返しは文字数で数える。長すぎる発話はそのまま出すと読めないため、
文字位置に比例した時刻で分割する。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

ProgressFn = Callable[[str, float], None]

# 行頭に来ると読みにくい約物。折り返し位置をひとつ後ろにずらす判断に使う。
_NO_LINE_START = "、。，．！？」』）】〉》’”ぁぃぅぇぉっゃゅょゎヵヶァィゥェォッャュョヮー"
_NO_LINE_END = "「『（【〈《‘“"


class TranscribeError(RuntimeError):
    """文字起こしができなかった。"""


@dataclass
class Cue:
    """字幕 1 枚。"""

    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def srt_timestamp(seconds: float) -> str:
    """SRT の時刻表記。ミリ秒はコンマ区切り。"""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def wrap_japanese(text: str, max_chars: int) -> List[str]:
    """文字数で折り返す。行頭・行末に置きたくない文字は避ける。"""
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    lines: List[str] = []
    rest = text
    while len(rest) > max_chars:
        cut = max_chars
        # 句読点などが行頭に来てしまう場合は 1 文字ぶん送る。
        while cut < len(rest) and rest[cut] in _NO_LINE_START:
            cut += 1
        while cut > 1 and rest[cut - 1] in _NO_LINE_END:
            cut -= 1
        lines.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        lines.append(rest)
    return lines


def split_long_cue(cue: Cue, max_chars: int, max_lines: int = 2) -> List[Cue]:
    """1 枚に収まらない発話を、文字位置に比例した時刻で分け."""
    capacity = max_chars * max_lines
    if capacity <= 0 or len(cue.text) <= capacity:
        return [cue]

    pieces: List[Cue] = []
    total = len(cue.text)
    offset = 0
    while offset < total:
        chunk = cue.text[offset: offset + capacity]
        start = cue.start + cue.duration * (offset / total)
        end = cue.start + cue.duration * (min(offset + capacity, total) / total)
        pieces.append(Cue(start=start, end=end, text=chunk))
        offset += capacity
    return pieces


def format_srt(cues: Sequence[Cue], max_chars_per_line: int = 20, max_lines: int = 2) -> str:
    """SRT 本文を組み立てる。"""
    blocks: List[str] = []
    index = 1
    for cue in cues:
        for piece in split_long_cue(cue, max_chars_per_line, max_lines):
            body = "\n".join(wrap_japanese(piece.text.strip(), max_chars_per_line))
            if not body:
                continue
            blocks.append(
                f"{index}\n"
                f"{srt_timestamp(piece.start)} --> {srt_timestamp(piece.end)}\n"
                f"{body}\n"
            )
            index += 1
    return "\n".join(blocks)


def write_srt(
    cues: Sequence[Cue],
    path: str | Path,
    *,
    max_chars_per_line: int = 20,
    max_lines: int = 2,
) -> Path:
    """SRT を書き出す。BOM 無し UTF-8（ffmpeg と各種編集ソフトが読める形）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        format_srt(cues, max_chars_per_line, max_lines), encoding="utf-8"
    )
    return target


def read_srt(path: str | Path) -> List[Cue]:
    """SRT を読み戻す。手で直した字幕を焼き込み直すために使う。"""
    text = Path(path).read_text(encoding="utf-8-sig")
    pattern = re.compile(
        r"(\d+)\s*\n(\d\d):(\d\d):(\d\d)[,.](\d{1,3})\s*-->\s*"
        r"(\d\d):(\d\d):(\d\d)[,.](\d{1,3})\s*\n(.*?)(?=\n\s*\n|\Z)",
        re.S,
    )
    cues: List[Cue] = []
    for match in pattern.finditer(text):
        start = (
            int(match.group(2)) * 3600 + int(match.group(3)) * 60
            + int(match.group(4)) + int(match.group(5).ljust(3, "0")) / 1000
        )
        end = (
            int(match.group(6)) * 3600 + int(match.group(7)) * 60
            + int(match.group(8)) + int(match.group(9).ljust(3, "0")) / 1000
        )
        cues.append(Cue(start=start, end=end, text=match.group(10).strip()))
    if not cues:
        raise TranscribeError(f"字幕を読み取れませんでした: {path}")
    return cues


def transcribe(
    audio_path: str | Path,
    *,
    model_size: str = "small",
    language: Optional[str] = "ja",
    device: str = "cpu",
    compute_type: str = "int8",
    on_progress: Optional[ProgressFn] = None,
) -> List[Cue]:
    """音声を文字起こしして字幕の元を返す。"""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscribeError(
            "faster-whisper が入っていません。`pip install faster-whisper` を実行してください。"
        ) from exc

    source = Path(audio_path)
    if not source.is_file():
        raise TranscribeError(f"音声ファイルが見つかりません: {source}")

    if on_progress:
        on_progress(f"文字起こしの準備中 (model={model_size})", 0.0)

    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as exc:  # noqa: BLE001 — 取得失敗も設定不備もまとめて案内する
        raise TranscribeError(
            f"文字起こしモデル '{model_size}' を用意できませんでした。\n"
            "初回はモデルの取得に通信が要ります。うまくいかない場合は "
            "model を 'tiny' や 'base' にして試してください。\n"
            f"（元の例外: {exc}）"
        ) from exc

    segments, info = model.transcribe(
        str(source),
        language=language,
        vad_filter=True,
        beam_size=5,
    )

    total = float(getattr(info, "duration", 0.0) or 0.0)
    cues: List[Cue] = []
    for segment in segments:
        text = (segment.text or "").strip()
        if text:
            cues.append(Cue(start=float(segment.start), end=float(segment.end), text=text))
        if on_progress and total > 0:
            on_progress("文字起こし中", min(1.0, float(segment.end) / total))
    return cues
