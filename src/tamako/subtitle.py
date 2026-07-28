"""字幕の焼き込みと、収録音声の差し替え。

字幕ファイルのパスは ffmpeg のフィルタ式に埋め込まれるため、Windows の
バックスラッシュやコロン、日本語ファイル名で壊れやすい。作業用の一時
フォルダに ASCII 名で複製し、そこを作業ディレクトリにして呼ぶことで、
エスケープの問題そのものを避けている。
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .ffmpeg import FFmpegError, find_ffmpeg, probe, run


@dataclass
class SubtitleStyle:
    """焼き込む字幕の見た目。"""

    font: str = "Yu Gothic UI"
    font_size: int = 42
    outline: int = 3
    shadow: int = 1
    margin_v: int = 60
    primary_colour: str = "&H00FFFFFF"   # 白
    outline_colour: str = "&H00000000"   # 黒
    alignment: int = 2                   # 下中央

    def to_force_style(self) -> str:
        return ",".join([
            f"FontName={self.font}",
            f"FontSize={self.font_size}",
            f"PrimaryColour={self.primary_colour}",
            f"OutlineColour={self.outline_colour}",
            "BorderStyle=1",
            f"Outline={self.outline}",
            f"Shadow={self.shadow}",
            f"MarginV={self.margin_v}",
            f"Alignment={self.alignment}",
        ])


def burn_subtitles(
    video: str | Path,
    srt: str | Path,
    output: str | Path,
    *,
    audio: Optional[str | Path] = None,
    audio_mode: str = "replace",
    original_volume: float = 0.15,
    style: Optional[SubtitleStyle] = None,
    crf: int = 20,
    preset: str = "medium",
    audio_bitrate: str = "192k",
) -> Path:
    """字幕を焼き込み、必要なら音声を差し替えて書き出す。

    audio_mode:
      replace — 収録音声で元の音を置き換える
      mix     — 元の音を original_volume まで下げて重ねる
    """
    video_path = Path(video).resolve()
    srt_path = Path(srt).resolve()
    out_path = Path(output).resolve()
    style = style or SubtitleStyle()

    if not video_path.is_file():
        raise FFmpegError(f"動画が見つかりません: {video_path}")
    if not srt_path.is_file():
        raise FFmpegError(f"字幕が見つかりません: {srt_path}")
    if audio_mode not in ("replace", "mix"):
        raise ValueError(f"audio_mode が不正です: {audio_mode!r} (replace / mix)")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="tamako_sub_") as workspace:
        work = Path(workspace)
        # フィルタ式に載せるのは常にこの ASCII 名。元の名前は問わない。
        local_srt = work / "subs.srt"
        shutil.copyfile(srt_path, local_srt)

        video_filter = f"subtitles=subs.srt:force_style='{style.to_force_style()}'"

        cmd: List[str] = [find_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
                          "-i", str(video_path)]
        if audio is not None:
            cmd += ["-i", str(Path(audio).resolve())]

        if audio is None:
            cmd += ["-vf", video_filter, "-map", "0:v:0"]
            if probe(video_path).has_audio:
                cmd += ["-map", "0:a:0", "-c:a", "copy"]
        elif audio_mode == "replace":
            cmd += ["-vf", video_filter, "-map", "0:v:0", "-map", "1:a:0",
                    "-c:a", "aac", "-b:a", audio_bitrate]
        else:
            has_original = probe(video_path).has_audio
            if not has_original:
                # 元に音が無いなら混ぜようがない。置き換えと同じ扱いにする。
                cmd += ["-vf", video_filter, "-map", "0:v:0", "-map", "1:a:0",
                        "-c:a", "aac", "-b:a", audio_bitrate]
            else:
                filter_complex = (
                    f"[0:v]{video_filter}[v];"
                    f"[0:a]volume={original_volume}[a0];"
                    f"[1:a]volume=1.0[a1];"
                    f"[a0][a1]amix=inputs=2:duration=longest:normalize=0[a]"
                )
                cmd += ["-filter_complex", filter_complex,
                        "-map", "[v]", "-map", "[a]",
                        "-c:a", "aac", "-b:a", audio_bitrate]

        cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "-shortest", str(out_path)]

        # 作業ディレクトリを一時フォルダにして、字幕を相対名で参照させる。
        proc = subprocess.run(
            cmd, cwd=str(work), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip().splitlines()[-15:]
            raise FFmpegError(
                "字幕の焼き込みに失敗しました:\n"
                + "\n".join("  " + line for line in tail)
                + f"\n\nフォント '{style.font}' が入っていない可能性があります。"
                " subtitle.font を端末にあるフォント名にしてください。"
            )

    return out_path


def check_audio_length(video: str | Path, audio: str | Path, tolerance: float = 1.0) -> Optional[str]:
    """動画と収録音声の長さのずれを調べ、問題があれば説明を返す。

    `-shortest` で書き出すため、ずれていると尻切れになる。黙って切るのではなく
    先に知らせる。
    """
    video_info = probe(video)
    audio_info = probe(audio) if Path(audio).suffix.lower() in {".mp4", ".mov", ".mkv"} else None

    if audio_info is None:
        proc = run([find_ffmpeg(), "-hide_banner", "-i", str(audio)], check=False)
        import re
        match = re.search(r"Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)", proc.stderr or "")
        if not match:
            return None
        audio_duration = (
            int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))
        )
    else:
        audio_duration = audio_info.duration

    difference = audio_duration - video_info.duration
    if abs(difference) <= tolerance:
        return None
    if difference > 0:
        return (
            f"収録音声のほうが {difference:.1f} 秒長いです"
            f"（映像 {video_info.duration:.1f} 秒 / 音声 {audio_duration:.1f} 秒）。"
            "映像の終わりに合わせて切られるため、音声の末尾が入りません。"
        )
    return (
        f"収録音声のほうが {-difference:.1f} 秒短いです"
        f"（映像 {video_info.duration:.1f} 秒 / 音声 {audio_duration:.1f} 秒）。"
        "音声の終わりに合わせて切られるため、映像の末尾が落ちます。"
    )
