"""出力の時間軸と素材の時間軸の対応。ここが唯一の権威。

人手修正の記録は「この素材のこの時刻のこの位置」を指す。出力側の時刻は
素材を 1 本足しただけで意味が変わる揮発性の識別子なので、耐久性のある記録は
必ず (素材, 素材時刻) で持ち、出力時刻との変換はこのモジュールに閉じ込める。

区間のフレーム数は round(duration × fps) で先に確定させる。映像は必ずこの
枚数を出し、音声のトリム長もここから逆算する。こうしないと、映像（ffmpeg が
出した枚数の連結）と音声（秒数指定の連結）の差が区間ごとに累積し、
100 区間で数秒の音ズレになる。
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .ordering import Clip
from .segments import CutPlan


def segment_frame_count(duration: float, fps: float) -> int:
    """区間が出力に占めるフレーム数。1 枚未満の区間も 1 枚は出す。"""
    return max(1, int(round(duration * fps)))


@dataclass
class Segment:
    """出力に載せる 1 区間。どの素材のどこから来たかを保持する。

    frames が時間の権威で、out_start / duration はそこからの導出にすぎない。
    """

    clip: Clip
    start: float
    end: float
    fps: float
    out_start_frame: int = 0

    @property
    def frames(self) -> int:
        return segment_frame_count(self.end - self.start, self.fps)

    @property
    def duration(self) -> float:
        """出力上の長さ（秒）。フレーム数から導出する。"""
        return self.frames / self.fps

    @property
    def out_start(self) -> float:
        return self.out_start_frame / self.fps

    def source_pts(self, local_index: int) -> float:
        """区間内 local_index 枚目の出力フレームに対応する素材時刻。"""
        return self.start + local_index / self.fps


@dataclass
class Timeline:
    """(素材, 素材時刻) ↔ 出力フレーム番号 の全単射。"""

    segments: List[Segment] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._starts = [s.out_start_frame for s in self.segments]

    @property
    def total_frames(self) -> int:
        if not self.segments:
            return 0
        last = self.segments[-1]
        return last.out_start_frame + last.frames

    @property
    def fps(self) -> float:
        return self.segments[0].fps if self.segments else 30.0

    def to_source(self, out_index: int) -> Tuple[Segment, float]:
        """出力フレーム番号 → (区間, 素材時刻)。範囲外は ValueError。"""
        if not (0 <= out_index < self.total_frames):
            raise ValueError(f"出力フレーム番号が範囲外です: {out_index}")
        pos = bisect.bisect_right(self._starts, out_index) - 1
        segment = self.segments[pos]
        return segment, segment.source_pts(out_index - segment.out_start_frame)

    def to_out_indices(self, clip_path, pts: float) -> List[int]:
        """(素材, 素材時刻) → 出力フレーム番号の一覧。

        同じ素材時刻が複数区間に載ることは通常ないが、0 件（カットされた）は
        普通に起きるので一覧で返す。
        """
        result: List[int] = []
        for segment in self.segments:
            if segment.clip.path != clip_path:
                continue
            local = int(round((pts - segment.start) * segment.fps))
            if 0 <= local < segment.frames:
                result.append(segment.out_start_frame + local)
        return result

    def out_time_to_source(self, seconds: float) -> Optional[Tuple[Segment, float]]:
        """出力の時刻（秒）→ (区間, 素材時刻)。範囲外は None。"""
        index = int(seconds * self.fps)
        if not (0 <= index < self.total_frames):
            return None
        return self.to_source(index)


def build_timeline(clip_plans: Sequence[Tuple[Clip, CutPlan]], fps: float) -> Timeline:
    """素材ごとの残す区間を、出力フレーム番号の通し番地付きで一列に並べる。"""
    segments: List[Segment] = []
    cursor = 0
    for clip, plan in clip_plans:
        for start, end in plan.keep:
            segment = Segment(clip=clip, start=start, end=end, fps=fps, out_start_frame=cursor)
            segments.append(segment)
            cursor += segment.frames
    return Timeline(segments=segments)
