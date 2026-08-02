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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .segments import Interval, intersect, invert, merge
from .tracks import SOURCE_DETECTED, SOURCE_MANUAL, SOURCE_RISK, TrackedClip

# 「推定ではない」箱。人が置いた箱は推定ではなく決定なので、検出と同格に扱う。
AUTHORITATIVE = {SOURCE_DETECTED, SOURCE_MANUAL}

# サイトの種類
KIND_NO_MASK = "no_mask"        # 何も描いていない
KIND_UNCERTAIN = "uncertain"    # 描いたが位置を保証できない
KIND_ESTIMATED = "estimated"    # 推定だけで描いている

KIND_LABEL = {
    KIND_NO_MASK: "何も隠していない",
    KIND_UNCERTAIN: "位置を保証できない",
    KIND_ESTIMATED: "推定で覆っている",
}


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


def build_sites(
    clip_path: Path,
    tracked: TrackedClip,
    keep: Sequence[Interval],
    silent: Sequence[Interval],
    *,
    duration: float,
    min_site_sec: float = 0.05,
    skip: Sequence[Interval] = (),
) -> List[Site]:
    """1 素材ぶんのサイトを作る。残す区間の中だけを見る。

    捨てる区間の顔漏れは出力に出ないので、人に見せると本物の危険が埋もれる。
    skip には、まだ有効な「確認済み」の区間を渡す。前回見て問題なかった箇所を
    毎回出し直すのが、確認作業が破綻する最大の原因。
    """
    fps = tracked.fps
    skipped = merge(list(skip)) if skip else []

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
            elif any(b.uncertain for b in boxes):
                flags[f] = KIND_UNCERTAIN
                tracks_at[f] = [b.track_id for b in boxes]
            elif all(b.source not in AUTHORITATIVE for b in boxes):
                flags[f] = KIND_ESTIMATED
                tracks_at[f] = [b.track_id for b in boxes]

    sites: List[Site] = []
    for kind, start, end in _runs(flags, fps):
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
    elif site.voiced:
        base += 0.1

    # 長いほど危ない（1 秒で頭打ち）。
    base += 0.1 * min(1.0, site.duration)
    return min(1.0, base)


def uncovered_intervals(sites: Sequence[Site]) -> List[Interval]:
    """C-2 の対象＝覆えていない区間。

    「何も描いていない」＋「位置を保証できない」を対象とし、推定で覆っている
    だけの区間は含めない（それは覆えてはいる）。
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
    """端末に出す要約。件数と、多すぎる場合の警告。"""
    if not sites:
        return "  要確認のサイト: 0 箇所"

    by_kind: Dict[str, int] = {}
    for site in sites:
        by_kind[site.kind] = by_kind.get(site.kind, 0) + 1

    lines = [f"  要確認のサイト: {len(sites)} 箇所"]
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
