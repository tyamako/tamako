"""検出結果からトラックを組み、途切れを埋める。

不変条件: **この処理はすべて素材クリップの時間軸上、カットを適用する前に行う。**
カットは最後に適用する。区間ごとに処理すると、区間の先頭・末尾（＝カット境界）が
片側の文脈しか持てず、いちばん弱い箇所で補間が効かなくなる。

埋め方は 3 段。安全側の倒し方が違うので分けてある。

- 補間 (interp): ギャップの前後に検出がある。位置は推定できるが、推定が外れる
  ぶんを箱の大きさで吸収する。拡大量は「時間」ではなく「速度 × 時間」に比例させる。
  静止した人の 2 秒ギャップに拡大は要らず、速いパンの 3 フレームには要るため。
- 外挿 (extrap): 片側にしか検出が無い（トラックの端、場面転換の直後など）。
  直前の速度で伸ばし、伸ばした時間に比例して広げる。
- 膨張 (dilate): 前後 ±k フレームの箱を取り込む。等速の仮定も破綻判定も要らない
  素朴な保険。補間・外挿の効果はこれとの差分で測る。

どれも上限に達したら uncertain を立て、C-2（覆えないフレームの扱い）に送る。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .detect import SCENE_DIFF_THRESHOLD, ClipDetections

# 箱の由来。危険度の並べ替えと D-3 の焼き込みが使う。
SOURCE_DETECTED = "detected"
SOURCE_INTERP = "interp"
SOURCE_EXTRAP = "extrap"
SOURCE_DILATE = "dilate"
SOURCE_MANUAL = "manual"

# 由来ごとの危険度の重み。大きいほど「人が見るべき」。
SOURCE_RISK = {
    SOURCE_DETECTED: 0.0,
    SOURCE_DILATE: 0.2,
    SOURCE_INTERP: 0.5,
    SOURCE_EXTRAP: 0.8,
    SOURCE_MANUAL: 0.0,
}


@dataclass(frozen=True)
class PlacedBox:
    """描くべき 1 つの箱。素材ピクセル座標。"""

    x: float
    y: float
    w: float
    h: float
    source: str
    track_id: str = ""
    score: float = 0.0
    # 推定が上限に達し、位置を保証できない箱。C-2 の判断材料。
    uncertain: bool = False

    @property
    def center(self) -> Tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    def grown(self, factor: float) -> "PlacedBox":
        if factor <= 1.0:
            return self
        cx, cy = self.center
        nw, nh = self.w * factor, self.h * factor
        return replace(self, x=cx - nw / 2, y=cy - nh / 2, w=nw, h=nh)

    def contains(self, other: "PlacedBox", tol: float = 1.0) -> bool:
        return (
            self.x <= other.x + tol
            and self.y <= other.y + tol
            and self.x + self.w >= other.x + other.w - tol
            and self.y + self.h >= other.y + other.h - tol
        )


@dataclass
class TrackConfig:
    """後処理の設定。既定値は「安全側だが過剰マスクを暴走させない」ところ。"""

    score_threshold: float = 0.5
    # 照合の許容移動量（箱の長辺に対する倍率）。1 フレームあたり。
    # 繋ぎ損なうと速度が取れず、外挿も拡大も効かなくなって静かに穴が開く。
    # 繋ぎすぎ（別人を同一視）は補間が大きめの箱になるだけなので、緩めに取る。
    match_distance_ratio: float = 2.0
    # 膨張の半径（フレーム）。前後これだけの箱を取り込む。
    dilate_frames: int = 2
    # 外挿を続ける長さ（フレーム）。これを超えたら諦めて uncertain。
    extrap_frames: int = 21
    # 補間するギャップの上限（フレーム）。これより長いと別人の可能性が高い。
    interp_max_gap: int = 90
    # 拡大の強さ。radius[px] = expand_per_velocity × 速度[px/s] × 時間[s]
    expand_per_velocity: float = 0.5
    # 拡大の上限（倍）。超える場合は上限で止めて uncertain を立てる。
    expand_limit: float = 4.0


@dataclass
class Track:
    """1 人ぶんの軌跡。frames は「検出できたフレーム」だけの疎な記録。"""

    track_id: str
    frames: Dict[int, PlacedBox] = field(default_factory=dict)

    @property
    def first(self) -> int:
        return min(self.frames)

    @property
    def last(self) -> int:
        return max(self.frames)


def _make_track_id(clip_name: str, first_frame: int, box: Sequence[float], fps: float) -> str:
    """内容から導く ID。検出設定を少し変えても初検出フレームは大抵変わらないので、
    パラメータを振っても人手修正の参照が生き残る確率が上がる。
    """
    cx = int((box[0] + box[2] / 2) // 16)
    cy = int((box[1] + box[3] / 2) // 16)
    seed = f"{clip_name}|{round(first_frame / fps, 1)}|{cx},{cy}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]


def build_tracks(det: ClipDetections, config: TrackConfig) -> List[Track]:
    """フレームごとの検出をトラックに繋ぐ。

    照合は「全ペアの距離を昇順に見て、まだ空いている組から確定させる」。
    トラックの作成順に最寄りを取らせると、2 人が交差したときに入れ替わり先が
    古いトラック側に固定され、人手修正の参照が別人に化ける。
    """
    clip_name = det.clip_path.stem
    tracks: List[Track] = []
    # (track, 最後に当たったフレーム, 速度[px/フレーム])
    active: List[List] = []
    scene_breaks = set(det.scene_breaks())

    for index in sorted(det.records):
        record = det.records[index]
        boxes = [b for b in record.boxes if b[4] >= config.score_threshold]

        # 場面転換を跨いだ照合はしない。別の絵なので、近い座標でも別人。
        if index in scene_breaks:
            active = []

        # 見失って久しいトラックは照合対象から外す。
        active = [a for a in active if index - a[1] <= config.interp_max_gap]

        if not boxes:
            continue

        pairs: List[Tuple[float, int, int]] = []
        for ti, (track, seen, velocity) in enumerate(active):
            prev = track.frames[seen]
            elapsed = index - seen
            # 直前の位置ではなく「そこから等速で進んだ予測位置」に対して照合する。
            # 素の位置と比べると、速く動く顔でトラックが 1 フレームごとに千切れる。
            pcx = prev.center[0] + velocity[0] * elapsed
            pcy = prev.center[1] + velocity[1] * elapsed
            span = max(prev.w, prev.h) * config.match_distance_ratio
            limit = span * max(1, elapsed)
            for bi, box in enumerate(boxes):
                bcx = box[0] + box[2] / 2
                bcy = box[1] + box[3] / 2
                distance = math.hypot(bcx - pcx, bcy - pcy)
                if distance <= limit:
                    pairs.append((distance, ti, bi))

        pairs.sort()
        used_tracks: set = set()
        used_boxes: set = set()
        assigned: Dict[int, int] = {}
        for _, ti, bi in pairs:
            if ti in used_tracks or bi in used_boxes:
                continue
            used_tracks.add(ti)
            used_boxes.add(bi)
            assigned[bi] = ti

        for bi, box in enumerate(boxes):
            placed = PlacedBox(
                x=box[0], y=box[1], w=box[2], h=box[3],
                source=SOURCE_DETECTED, score=box[4],
            )
            if bi in assigned:
                entry = active[assigned[bi]]
                track, seen, velocity = entry
                prev = track.frames[seen]
                track.frames[index] = replace(placed, track_id=track.track_id)
                elapsed = max(1, index - seen)
                fresh = (
                    (placed.center[0] - prev.center[0]) / elapsed,
                    (placed.center[1] - prev.center[1]) / elapsed,
                )
                # 速度はならして持つ。1 フレームの検出のぶれで予測が暴れないように。
                alpha = 0.6
                entry[0], entry[1] = track, index
                entry[2] = (
                    velocity[0] * (1 - alpha) + fresh[0] * alpha,
                    velocity[1] * (1 - alpha) + fresh[1] * alpha,
                )
            else:
                track_id = _make_track_id(clip_name, index, box, det.fps)
                # 同じ ID が二度生まれないよう連番で逃がす。
                if any(t.track_id == track_id for t in tracks):
                    track_id = f"{track_id}-{len(tracks)}"
                track = Track(track_id=track_id)
                track.frames[index] = replace(placed, track_id=track_id)
                tracks.append(track)
                active.append([track, index, (0.0, 0.0)])

    return tracks


def _velocity(track: Track, at: int, *, forward: bool, fps: float,
              keys: Sequence[int], pos_of: Dict[int, int]) -> Tuple[float, float]:
    """at 付近の速度（px/秒）。近傍 2 点から取る。取れなければ 0。

    keys / pos_of は呼び出し側で 1 度だけ作る。ここで毎回 sorted や index を
    やると、長いトラックで O(n^2) になる。
    """
    if len(keys) < 2:
        return 0.0, 0.0
    pos = pos_of[at]
    if forward:
        other = keys[pos - 1] if pos > 0 else None
    else:
        other = keys[pos + 1] if pos + 1 < len(keys) else None
    if other is None:
        return 0.0, 0.0
    a, b = track.frames[other], track.frames[at]
    dt = (at - other) / fps
    if abs(dt) < 1e-9:
        return 0.0, 0.0
    (ax, ay), (bx, by) = a.center, b.center
    return (bx - ax) / dt, (by - ay) / dt


def _lerp_box(a: PlacedBox, b: PlacedBox, u: float, source: str) -> PlacedBox:
    """x, y, w, h をすべて補間する。人が奥から手前に来れば箱も大きくなる。"""
    return PlacedBox(
        x=a.x + (b.x - a.x) * u,
        y=a.y + (b.y - a.y) * u,
        w=a.w + (b.w - a.w) * u,
        h=a.h + (b.h - a.h) * u,
        source=source,
        track_id=a.track_id,
        score=min(a.score, b.score),
    )


def _growth(config: TrackConfig, speed: float, mismatch: float, seconds: float,
            box: PlacedBox) -> Tuple[float, bool]:
    """推定の不確かさを、箱を何倍にして吸収するか。上限に達したら True を返す。"""
    radius = config.expand_per_velocity * (speed + mismatch) * seconds
    base = max(box.w, box.h) * 0.5
    if base <= 0:
        return 1.0, True
    factor = 1.0 + radius / base
    if factor >= config.expand_limit:
        return config.expand_limit, True
    return factor, False


def fill_track(track: Track, config: TrackConfig, *, fps: float,
               scene_breaks: Iterable[int]) -> Dict[int, PlacedBox]:
    """1 トラックのギャップを埋め、フレーム番号 → 箱 の密な対応を返す。"""
    import bisect

    breaks = sorted(scene_breaks)
    dense: Dict[int, PlacedBox] = dict(track.frames)
    keys = sorted(track.frames)
    pos_of = {k: i for i, k in enumerate(keys)}

    def crosses_break(f0: int, f1: int) -> bool:
        i = bisect.bisect_right(breaks, f0)
        return i < len(breaks) and breaks[i] <= f1

    # --- ギャップの補間 ---
    for f0, f1 in zip(keys, keys[1:]):
        gap = f1 - f0
        if gap <= 1:
            continue
        a, b = track.frames[f0], track.frames[f1]
        if gap > config.interp_max_gap or crosses_break(f0, f1):
            # 跨げない。両端から外挿だけして、中央は諦める。
            _extrapolate(dense, track, f0, +1, config, fps=fps, stop=f1,
                         keys=keys, pos_of=pos_of)
            _extrapolate(dense, track, f1, -1, config, fps=fps, stop=f0,
                         keys=keys, pos_of=pos_of)
            continue

        v0 = _velocity(track, f0, forward=True, fps=fps, keys=keys, pos_of=pos_of)
        v1 = _velocity(track, f1, forward=False, fps=fps, keys=keys, pos_of=pos_of)
        speed = max(math.hypot(*v0), math.hypot(*v1))
        mismatch = math.hypot(v1[0] - v0[0], v1[1] - v0[1])

        for f in range(f0 + 1, f1):
            u = (f - f0) / gap
            box = _lerp_box(a, b, u, SOURCE_INTERP)
            # 不確かさは両端から遠いほど大きい。
            seconds = min(f - f0, f1 - f) / fps
            factor, capped = _growth(config, speed, mismatch, seconds, box)
            dense[f] = replace(box.grown(factor), uncertain=capped)

    # --- トラックの端の外挿 ---
    if keys:
        _extrapolate(dense, track, keys[-1], +1, config, fps=fps,
                     keys=keys, pos_of=pos_of)
        _extrapolate(dense, track, keys[0], -1, config, fps=fps,
                     keys=keys, pos_of=pos_of)
    return dense


def _extrapolate(dense: Dict[int, PlacedBox], track: Track, anchor: int, direction: int,
                 config: TrackConfig, *, fps: float, keys: Sequence[int],
                 pos_of: Dict[int, int], stop: Optional[int] = None) -> None:
    """anchor から direction 方向へ、速度で伸ばしながら広げる。"""
    base = track.frames[anchor]
    vx, vy = _velocity(track, anchor, forward=(direction > 0), fps=fps,
                       keys=keys, pos_of=pos_of)
    speed = math.hypot(vx, vy)
    for step in range(1, config.extrap_frames + 1):
        f = anchor + direction * step
        if f < 0 or f in dense:
            break
        if stop is not None and ((direction > 0 and f >= stop) or (direction < 0 and f <= stop)):
            break
        seconds = step / fps
        moved = replace(
            base,
            x=base.x + vx * seconds * direction,
            y=base.y + vy * seconds * direction,
            source=SOURCE_EXTRAP,
        )
        factor, capped = _growth(config, speed, 0.0, seconds, moved)
        dense[f] = replace(moved.grown(factor), uncertain=capped)


def _dedupe(boxes: Sequence[PlacedBox]) -> List[PlacedBox]:
    """他の箱に完全に含まれる箱を落とす。被覆は変わらず、描画回数だけ減る。"""
    ordered = sorted(boxes, key=lambda b: b.w * b.h, reverse=True)
    kept: List[PlacedBox] = []
    for box in ordered:
        if not any(k.contains(box) for k in kept):
            kept.append(box)
    return kept


def _dilate(dense: Dict[int, PlacedBox], config: TrackConfig) -> Dict[int, List[PlacedBox]]:
    """前後 ±k フレームの箱を取り込む。窓内を包む 1 つの箱にまとめる。

    「それぞれ描く」と 1 トラック 1 フレームあたり最大 2k+1 個の合成になる。
    窓は 0.2 秒程度なので、包む箱にしても大きくなりすぎない（なりすぎる場合は
    包まずに個別に描く）。被覆はどちらも元の集合以上。
    """
    k = max(0, config.dilate_frames)
    if k == 0:
        return {f: [b] for f, b in dense.items()}

    keys = sorted(dense)
    result: Dict[int, List[PlacedBox]] = {}
    import bisect

    for f in keys:
        lo = bisect.bisect_left(keys, f - k)
        hi = bisect.bisect_right(keys, f + k)
        window = [dense[keys[i]] for i in range(lo, hi)]
        here = dense[f]
        if len(window) <= 1:
            result[f] = [here]
            continue
        x0 = min(b.x for b in window)
        y0 = min(b.y for b in window)
        x1 = max(b.x + b.w for b in window)
        y1 = max(b.y + b.h for b in window)
        # 包む箱が元の箱に対して大きくなりすぎるなら、個別に描く。
        if (x1 - x0) <= here.w * config.expand_limit and (y1 - y0) <= here.h * config.expand_limit:
            source = here.source if here.source != SOURCE_DETECTED else SOURCE_DILATE
            result[f] = [replace(here, x=x0, y=y0, w=x1 - x0, h=y1 - y0,
                                 source=source if len(window) > 1 else here.source)]
        else:
            result[f] = _dedupe(window)
    return result


@dataclass
class TrackedClip:
    """1 素材ぶんの、フレーム番号 → 描くべき箱。"""

    fps: float
    width: int
    height: int
    frames_total: int
    boxes: Dict[int, List[PlacedBox]] = field(default_factory=dict)
    tracks: List[Track] = field(default_factory=list)
    scene_breaks: List[int] = field(default_factory=list)

    def at_pts(self, pts: float) -> List[PlacedBox]:
        return self.boxes.get(int(round(pts * self.fps)), [])

    def at_frame(self, index: int) -> List[PlacedBox]:
        return self.boxes.get(index, [])


def track_clip(det: ClipDetections, config: TrackConfig) -> TrackedClip:
    """検出結果 → 描くべき箱。ここまでが素材時間軸の仕事。"""
    tracks = build_tracks(det, config)
    breaks = det.scene_breaks()
    merged: Dict[int, List[PlacedBox]] = {}
    for track in tracks:
        dense = fill_track(track, config, fps=det.fps, scene_breaks=breaks)
        for frame, boxes in _dilate(dense, config).items():
            merged.setdefault(frame, []).extend(boxes)
    for frame in merged:
        merged[frame] = _dedupe(merged[frame])
    return TrackedClip(
        fps=det.fps, width=det.width, height=det.height,
        frames_total=det.frames_total, boxes=merged, tracks=tracks,
        scene_breaks=breaks,
    )
