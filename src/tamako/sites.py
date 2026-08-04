"""人が見るべき箇所（サイト）の集約と危険度。

フレーム単位の記録は人が読める単位ではない。18000 フレームを並べ替えても
18000 行のリストにしかならず、確認作業はそこで破綻する。連続する危険フレームを
区間に畳み、危険度順に並べたものだけを人に渡す。

危険度に使う材料:
- 箱の由来（検出 / 膨張 / 補間 / 外挿）。推定の度合いがそのまま不確かさ
- 位置を保証できない印（拡大が上限に達した）
- **音**。silence.py がすでに計算している無音区間の裏返し。
  「音が鳴っている（喋っている）のに顔が無い」は、人が写っている強い証拠であり、
  かつ視覚の検出器とは完全に独立した唯一の手がかり。検出器同士は暗所・ブレ・
  小ささで揃って失敗するので、独立したモダリティを 1 つ持つ意味は大きい。
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .segments import Interval, intersect, invert, merge
from .tracks import SOURCE_DETECTED, SOURCE_MANUAL, SOURCE_RISK, TrackedClip

# 「推定ではない」箱。人が置いた箱は推定ではなく決定なので、検出と同格に扱う。
AUTHORITATIVE = {SOURCE_DETECTED, SOURCE_MANUAL}

# サイトの種類
KIND_NO_MASK = "no_mask"        # 何も描いていない
KIND_SHRUNK = "shrunk"          # 人が箱を小さくした（素顔を出しうる唯一の操作）
KIND_LOST_TRACK = "lost_track"  # 同じ場面で、いた人の 1 人が検出から消えた
KIND_UNCERTAIN = "uncertain"    # 描いたが位置を保証できない
KIND_ESTIMATED = "estimated"    # 推定だけで描いている

KIND_LABEL = {
    KIND_NO_MASK: "何も隠していない",
    KIND_SHRUNK: "人が小さくした",
    KIND_LOST_TRACK: "人数が減った",
    KIND_UNCERTAIN: "位置を保証できない",
    KIND_ESTIMATED: "推定で覆っている",
}

# 種類ごとに「何をすればよいか」を 1 行。危険度だけを見せても人は動けない。
KIND_ACTION = {
    KIND_NO_MASK: "顔があるか探してください。あれば箱を足す",
    KIND_SHRUNK: "小さくした箱から顔が出ていないか見てください",
    KIND_LOST_TRACK: "もう 1 人の顔が隠れているか見てください",
    KIND_UNCERTAIN: "箱が顔に乗っているか見てください",
    KIND_ESTIMATED: "はみ出していないか見てください",
}

# サイト 1 件の長さの上限（秒）。既定 5.0。
# 上限が無いと 20 秒の no_mask が 1 件でき、「サイト全体に効かせる」を
# 既定にした瞬間、静止した箱を 20 秒に効かせることになる。
MAX_SITE_SEC = 5.0

# トラックが途切れたあと「人数が減った」と見なし続ける長さ（秒）。
LOST_TRACK_SEC = 2.0


@dataclass
class Site:
    """人が確認すべき 1 区間。時刻は**素材**の時間軸（耐久性のある座標）。"""

    clip: Path
    start: float
    end: float
    kind: str
    risk: float
    voiced: bool = False
    track_ids: List[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def label(self) -> str:
        mark = "／音あり" if self.voiced else ""
        return f"{KIND_LABEL.get(self.kind, self.kind)}{mark}"

    def action(self) -> str:
        return KIND_ACTION.get(self.kind, "見てください")


def _runs(flags: Dict[int, str], fps: float) -> List[Tuple[str, float, float]]:
    """フレーム番号 → 種類 の対応を、連続する区間に畳む。"""
    if not flags:
        return []
    result: List[Tuple[str, float, float]] = []
    keys = sorted(flags)
    start = prev = keys[0]
    kind = flags[start]
    for f in keys[1:]:
        if f == prev + 1 and flags[f] == kind:
            prev = f
            continue
        result.append((kind, start / fps, (prev + 1) / fps))
        start = prev = f
        kind = flags[f]
    result.append((kind, start / fps, (prev + 1) / fps))
    return result


def lost_track_frames(tracked: TrackedClip, *, window_frames: int) -> Dict[int, List[str]]:
    """同じ場面で、あるトラックが途切れたのに他が続いている箇所。

    **今の 3 種類が原理的に拾えない事象がここにある。** フレームに箱が 1 つでも
    あれば no_mask にしないので、2 人写って 1 人だけ検出できているフレームは
    一覧に一切出ない。複数人の撮影で最も起きやすい漏れが、リストに現れない。

    見るのは boxes ではなく **tracks**（＝検出できたフレーム）。boxes には
    補間・外挿で埋めた分が入っているので、そこから導くと「埋まっているから
    問題なし」に見えてしまう。埋めた区間はもともと KIND_ESTIMATED で出る。

    打ち切る条件は 3 つ:
    - 場面転換を跨いだら見ない（別の絵なので、いなくて当然）
    - 新しいトラックが始まったら見ない（取り直せている）
    - クリップの終端
    """
    tracks = [t for t in tracked.tracks if t.frames]
    if len(tracks) < 2:
        return {}

    breaks = sorted(tracked.scene_breaks)
    starts = sorted(t.first for t in tracks)
    lost: Dict[int, List[str]] = {}
    for track in tracks:
        last = track.last
        limit = min(last + max(0, window_frames), tracked.frames_total - 1)
        nxt = bisect.bisect_right(breaks, last)
        if nxt < len(breaks):
            limit = min(limit, breaks[nxt] - 1)
        fresh = bisect.bisect_right(starts, last)
        if fresh < len(starts):
            limit = min(limit, starts[fresh] - 1)
        for frame in range(last + 1, limit + 1):
            if any(o is not track and o.first <= frame <= o.last for o in tracks):
                lost.setdefault(frame, []).append(track.track_id)
    return lost


def _split_long(start: float, end: float, limit: float) -> List[Tuple[float, float]]:
    """長すぎるサイトを等分する。端に極端に短い切れ端を作らない。"""
    if limit <= 0 or end - start <= limit:
        return [(start, end)]
    count = int(math.ceil((end - start) / limit))
    step = (end - start) / count
    return [
        (start + i * step, end if i == count - 1 else start + (i + 1) * step)
        for i in range(count)
    ]


def build_sites(
    clip_path: Path,
    tracked: TrackedClip,
    keep: Sequence[Interval],
    silent: Sequence[Interval],
    *,
    duration: float,
    min_site_sec: float = 0.05,
    max_site_sec: float = MAX_SITE_SEC,
    lost_track_sec: float = LOST_TRACK_SEC,
    skip: Sequence[Interval] = (),
    shrunk: Sequence[Interval] = (),
) -> List[Site]:
    """1 素材ぶんのサイトを作る。残す区間の中だけを見る。

    捨てる区間の顔漏れは出力に出ないので、人に見せると本物の危険が埋もれる。
    skip には、まだ有効な「確認済み」の区間を渡す。前回見て問題なかった箇所を
    毎回出し直すのが、確認作業が破綻する最大の原因。

    shrunk には「人が箱を小さくした」区間を渡す。縮小は人手操作で唯一、
    素顔を出しうる操作でありながら、結果の箱は source=manual / uncertain=False
    なので下のどの分岐にも入らない——**確認を無効に戻してもサイトが生まれない**。
    明示的に 1 度は出す必要がある。
    """
    fps = tracked.fps
    skipped = merge(list(skip)) if skip else []
    shrunk_spans = merge(list(shrunk)) if shrunk else []
    lost = lost_track_frames(
        tracked, window_frames=int(round(lost_track_sec * fps))
    ) if lost_track_sec > 0 else {}

    def is_shrunk(f: int) -> bool:
        t0 = f / fps
        return any(s <= t0 < e for s, e in shrunk_spans)

    def is_skipped(f: int) -> bool:
        t0 = f / fps
        return any(s <= t0 < e for s, e in skipped)
    voiced = invert(list(silent), duration) if silent else [(0.0, duration)]

    def is_voiced(t: float) -> bool:
        return any(s <= t < e for s, e in voiced)

    flags: Dict[int, str] = {}
    tracks_at: Dict[int, List[str]] = {}
    for start, end in merge(list(keep)):
        first = int(round(start * fps))
        last = int(round(end * fps))
        for f in range(first, last):
            if is_skipped(f):
                continue
            boxes = tracked.at_frame(f)
            if not boxes:
                flags[f] = KIND_NO_MASK
            elif is_shrunk(f):
                flags[f] = KIND_SHRUNK
                tracks_at[f] = [b.track_id for b in boxes]
            elif f in lost:
                flags[f] = KIND_LOST_TRACK
                tracks_at[f] = lost[f]
            elif any(b.uncertain for b in boxes):
                flags[f] = KIND_UNCERTAIN
                tracks_at[f] = [b.track_id for b in boxes]
            elif all(b.source not in AUTHORITATIVE for b in boxes):
                flags[f] = KIND_ESTIMATED
                tracks_at[f] = [b.track_id for b in boxes]

    sites: List[Site] = []
    for kind, run_start, run_end in _runs(flags, fps):
        for start, end in _split_long(run_start, run_end, max_site_sec):
            if end - start < min_site_sec:
                continue
            middle = (start + end) / 2
            heard = is_voiced(middle)
            ids: List[str] = []
            for f in range(int(round(start * fps)), int(round(end * fps))):
                for tid in tracks_at.get(f, []):
                    if tid and tid not in ids:
                        ids.append(tid)
            sites.append(
                Site(clip=clip_path, start=start, end=end, kind=kind,
                     risk=0.0, voiced=heard, track_ids=ids)
            )

    for site in sites:
        site.risk = score_site(site, tracked)
    sites.sort(key=lambda s: s.risk, reverse=True)
    return sites


def score_site(site: Site, tracked: TrackedClip) -> float:
    """危険度。0〜1 に収まるよう頭打ちにする。

    重み付けは当面ヒューリスティック。人手修正のログが溜まったら、
    「人がそのサイトを実際に直したか」を目的変数にした較正に置き換える。
    """
    if site.kind == KIND_NO_MASK:
        base = 0.6
    elif site.kind == KIND_SHRUNK:
        # 縮小は人手操作で唯一、素顔を出しうる操作。
        base = 0.6
    elif site.kind == KIND_LOST_TRACK:
        # 「もう 1 人が丸ごと素顔」の可能性。何も隠していないのに近い。
        base = 0.55
    elif site.kind == KIND_UNCERTAIN:
        base = 0.5
    else:
        # 推定で覆っている場合は、由来の重みで測る。
        fps = tracked.fps
        weights = [
            SOURCE_RISK.get(b.source, 0.5)
            for f in range(int(site.start * fps), int(site.end * fps) + 1)
            for b in tracked.at_frame(f)
        ]
        base = 0.3 * (max(weights) if weights else 0.5)

    # 音がしているのにマスクが無い＝人が写っている強い証拠。検出器と独立。
    if site.voiced and site.kind == KIND_NO_MASK:
        base += 0.35
    elif site.voiced and site.kind == KIND_LOST_TRACK:
        # 喋っているのに人数が減った＝消えた人はまだ画面にいる可能性が高い。
        base += 0.2
    elif site.voiced:
        base += 0.1

    # 長いほど危ない（1 秒で頭打ち）。
    base += 0.1 * min(1.0, site.duration)
    return min(1.0, base)


def uncovered_intervals(sites: Sequence[Site]) -> List[Interval]:
    """C-2 の対象＝覆えていない区間。

    「何も描いていない」＋「位置を保証できない」を対象とし、推定で覆っている
    だけの区間は含めない（それは覆えてはいる）。

    「人数が減った」は**入れない**。あれは覆えていない証拠ではなく疑いなので、
    これで落とすと「人が枠外に出ただけ」の場面を機械が勝手に削ることになる。
    「人が小さくした」も入れない（人の決定を機械が取り消す形になる）。
    """
    return merge([
        (s.start, s.end) for s in sites
        if s.kind in (KIND_NO_MASK, KIND_UNCERTAIN)
    ])


def apply_cut_policy(keep: Sequence[Interval], sites: Sequence[Site],
                     *, min_keep_sec: float) -> List[Interval]:
    """uncovered_policy=cut のとき、覆えない区間を残す区間から取り除く。

    フレームを個別に落とすのではなく、残す区間そのものを削る。フレームを
    間引くと映像の枚数と音声の秒数が食い違い、時間軸の契約（timeline）が壊れる。
    """
    bad = uncovered_intervals(sites)
    if not bad:
        return list(keep)
    total = max((e for _, e in keep), default=0.0)
    good = invert(bad, total) if total > 0 else []
    result = intersect(list(keep), good)
    return [(s, e) for s, e in result if e - s >= min_keep_sec]


def summarize_sites(sites: Sequence[Site], *, out_duration: float,
                    limit_per_10min: int = 50) -> str:
    """端末に出す要約。件数と、多すぎる場合の警告。

    **「網羅」と偽らない。** 暗所で 2 人とも取れない、といった事象は
    KIND_LOST_TRACK を足しても原理的に拾えない。件数を進捗の分母の主役に
    据えると「23 件を消化した＝終わった」と読まれるので、表示で嘘をつかない。
    """
    if not sites:
        return ("  機械が気づいた箇所: 0 箇所（これで全部ではありません）\n"
                "    ※ 0 箇所でも、全編を通しで見る工程は省けません。")

    by_kind: Dict[str, int] = {}
    for site in sites:
        by_kind[site.kind] = by_kind.get(site.kind, 0) + 1

    lines = [f"  機械が気づいた箇所: {len(sites)} 箇所（これで全部ではありません）"]
    for kind, count in sorted(by_kind.items(), key=lambda kv: -kv[1]):
        lines.append(f"    {KIND_LABEL.get(kind, kind)}: {count} 箇所")

    minutes = max(out_duration / 60.0, 1e-6)
    density = len(sites) / minutes * 10.0
    # 短い素材では密度が跳ねて意味を持たないので、実数でも一定数を超えたときだけ言う。
    if density > limit_per_10min and len(sites) >= 10:
        lines.append(
            f"  ※ 10 分あたり {density:.0f} 箇所は多すぎます（目安 {limit_per_10min} 箇所）。"
        )
        lines.append(
            "     人が直す前に、mask.scale を上げる・score_threshold を下げる"
            "・dilate_frames を増やす、を先に試してください。"
        )
    return "\n".join(lines)


def describe_sites(sites: Sequence[Site], limit: int = 10) -> str:
    """危険度順に上から並べる。時刻順ではない（危ない順に見てもらうため）。"""
    from .report import timecode

    if not sites:
        return "  なし"
    lines = []
    for i, site in enumerate(sites[:limit], start=1):
        lines.append(
            f"  {i:2d}. [{site.risk:.2f}] {site.clip.name} "
            f"{timecode(site.start)}-{timecode(site.end)}  {site.label()}"
        )
        lines.append(f"      → {site.action()}")
    if len(sites) > limit:
        lines.append(f"      … 他 {len(sites) - limit} 箇所")
    return "\n".join(lines)


def sites_to_json(sites: Sequence[Site]) -> List[dict]:
    from .report import timecode

    return [
        {
            "clip": s.clip.name,
            "start": round(s.start, 3),
            "end": round(s.end, 3),
            "start_tc": timecode(s.start),
            "end_tc": timecode(s.end),
            "duration": round(s.duration, 3),
            "kind": s.kind,
            "risk": round(s.risk, 3),
            "voiced": s.voiced,
            "tracks": s.track_ids,
        }
        for s in sites
    ]
