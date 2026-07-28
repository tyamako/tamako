"""素材を撮影順に並べる。

ファイル名の順は撮影順とは限らない（連番が日をまたいで巻き戻る、カメラを
複数台使った、といったことが普通に起きる）。撮影時刻のメタデータを第一の
根拠とし、無ければファイルの更新時刻で代用する。
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from .ffmpeg import FFmpegError, MediaInfo, probe


@dataclass
class Clip:
    """並べ替えを済ませた 1 本の素材。"""

    info: MediaInfo
    order: int
    sort_time: _dt.datetime
    sort_basis: str

    @property
    def path(self) -> Path:
        return self.info.path

    @property
    def name(self) -> str:
        return self.info.path.name


def find_videos(directory: str | Path, extensions: Sequence[str]) -> List[Path]:
    """フォルダ直下の動画ファイルを集める。拡張子の大文字小文字は無視する。"""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"入力フォルダが見つかりません: {root}")
    allowed = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    return sorted(
        (p for p in root.iterdir() if p.is_file() and p.suffix.lower() in allowed),
        key=lambda p: p.name.lower(),
    )


def order_clips(paths: Sequence[str | Path]) -> tuple[List[Clip], List[tuple[Path, str]]]:
    """撮影順に並べた素材と、読めなかったファイルの一覧を返す。

    1 本読めなくても全体を止めない。撮影の現場では壊れたファイルや、動画で
    ないものが同じフォルダに紛れ込むことがある。
    """
    clips: List[Clip] = []
    failures: List[tuple[Path, str]] = []

    for path in paths:
        target = Path(path)
        try:
            info = probe(target)
        except FFmpegError as exc:
            failures.append((target, str(exc).splitlines()[0]))
            continue

        if info.creation_time is not None:
            sort_time, basis = info.creation_time, info.time_source
        else:
            sort_time = _dt.datetime.fromtimestamp(target.stat().st_mtime)
            basis = "file_mtime"

        clips.append(Clip(info=info, order=0, sort_time=sort_time, sort_basis=basis))

    # 同時刻の素材はファイル名で決める。実行するたびに順が変わると困る。
    clips.sort(key=lambda c: (c.sort_time, c.name.lower()))
    for index, clip in enumerate(clips):
        clip.order = index

    return clips, failures


def describe_order(clips: Sequence[Clip]) -> str:
    """並び順を人が検算できる形にする。自動で決めた順序は必ず見せる。"""
    if not clips:
        return "（素材がありません）"
    lines = []
    for clip in clips:
        stamp = clip.sort_time.strftime("%Y-%m-%d %H:%M:%S")
        lines.append(
            f"  {clip.order + 1:3d}. {clip.name}  "
            f"{clip.info.duration:7.2f}s  {clip.info.width}x{clip.info.height}  "
            f"{stamp} ({clip.sort_basis})"
        )
    return "\n".join(lines)
