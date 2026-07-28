"""区間（開始秒・終了秒）の演算と、残す区間の決定。

無音区間と「顔が写っていない区間」という 2 つの根拠から、最終的に残す区間を
組み立てる。ここは動画に一切触れない純粋な計算なので、単体で試験できる。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

Interval = Tuple[float, float]

EPS = 1e-6


def merge(intervals: Iterable[Interval], gap: float = 0.0) -> List[Interval]:
    """重なり合う区間を統合する。`gap` 以下の隙間は繋がっているものとして扱う。"""
    ordered = sorted((s, e) for s, e in intervals if e - s > EPS)
    if not ordered:
        return []
    result = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start - result[-1][1] <= gap + EPS:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return [(s, e) for s, e in result]


def invert(intervals: Sequence[Interval], duration: float) -> List[Interval]:
    """[0, duration] のうち、与えられた区間に含まれない部分を返す。"""
    result: List[Interval] = []
    cursor = 0.0
    for start, end in merge(intervals):
        if start - cursor > EPS:
            result.append((cursor, min(start, duration)))
        cursor = max(cursor, end)
        if cursor >= duration:
            break
    if duration - cursor > EPS:
        result.append((cursor, duration))
    return [(s, e) for s, e in result if e - s > EPS]


def intersect(a: Sequence[Interval], b: Sequence[Interval]) -> List[Interval]:
    """2 つの区間集合の共通部分。"""
    left, right = merge(a), merge(b)
    result: List[Interval] = []
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end - start > EPS:
            result.append((start, end))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return result


def union(a: Sequence[Interval], b: Sequence[Interval]) -> List[Interval]:
    """2 つの区間集合の和。"""
    return merge(list(a) + list(b))


def drop_short(intervals: Sequence[Interval], minimum: float) -> List[Interval]:
    """指定より短い区間を捨てる。"""
    return [(s, e) for s, e in intervals if e - s >= minimum - EPS]


def pad(intervals: Sequence[Interval], amount: float, duration: float) -> List[Interval]:
    """各区間を前後に広げる。0 と duration からはみ出さない。"""
    if amount <= 0:
        return merge(intervals)
    grown = [(max(0.0, s - amount), min(duration, e + amount)) for s, e in intervals]
    return merge(grown)


def total(intervals: Sequence[Interval]) -> float:
    return sum(e - s for s, e in intervals)


def samples_to_intervals(times: Sequence[float], step: float, hold: float = 0.0) -> List[Interval]:
    """「その時刻に該当した」標本列を区間に変換する。

    顔検出は取りこぼしが起きる。`hold` 秒以下の空白は検出が途切れただけとみなし、
    繋いでしまう。これをしないと一瞬の見失いで細切れのカットが大量に出る。
    """
    return merge([(t, t + step) for t in times], gap=hold)


@dataclass
class CutPlan:
    """1 本の素材について、残す区間と捨てる区間、およびその根拠。"""

    duration: float
    keep: List[Interval]
    cut: List[Interval]
    silent: List[Interval]
    faceless: List[Interval]

    @property
    def kept_seconds(self) -> float:
        return total(self.keep)

    @property
    def cut_seconds(self) -> float:
        return total(self.cut)


def build_cut_plan(
    *,
    duration: float,
    silent: Sequence[Interval],
    face_present: Sequence[Interval],
    mode: str = "any",
    min_cut_sec: float = 0.5,
    min_keep_sec: float = 0.6,
    padding_sec: float = 0.25,
) -> CutPlan:
    """無音区間と顔在区間から、残す区間を決める。

    mode:
      any     — 無音 または 顔なし を切る（既定）
      both    — 無音 かつ 顔なし のときだけ切る（最も安全）
      silence — 無音だけを根拠に切る
      face    — 顔なしだけを根拠に切る
    """
    silent_merged = merge(silent)
    faceless = invert(face_present, duration)

    if mode == "any":
        cut = union(silent_merged, faceless)
    elif mode == "both":
        cut = intersect(silent_merged, faceless)
    elif mode == "silence":
        cut = silent_merged
    elif mode == "face":
        cut = faceless
    else:
        raise ValueError(
            f"cut.mode が不正です: {mode!r} (any / both / silence / face のいずれか)"
        )

    # 短すぎるカットは、細切れになるだけで見づらいので実行しない。
    cut = drop_short(cut, min_cut_sec)
    keep = invert(cut, duration)
    # 短すぎる残しは、瞬きのようなカットの間に挟まった無意味な断片なので捨てる。
    keep = drop_short(keep, min_keep_sec)
    # 語頭・語尾が切れないよう前後に余白を足す。
    keep = pad(keep, padding_sec, duration)

    return CutPlan(
        duration=duration,
        keep=keep,
        cut=invert(keep, duration),
        silent=silent_merged,
        faceless=faceless,
    )
