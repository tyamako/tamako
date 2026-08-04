"""人手修正の記録とマージ。窓（fix.py）から切り離してある。

**記録はトラック ID を参照しない。** 検出設定を変えて faces.jsonl を作り直すと
ID の対応は崩れる。「警告を出す」では人が 1 時間かけた作業が警告付きで消えるだけ
なので、位置（アンカー点）で参照する。文書のコメントアンカーと同じ考え方で、
ID が変わっても、その場所にその人がいる限り正しく解決される。
「消す」は本来「あそこに出る誤検出を消したい」なので、意味としてもこちらが正しい。

記録は追記のみ。undo は打ち消しの記録を足すことで表現する（元の行は消さない）。
壊れても前の状態に戻せるし、diff で何をしたかが読める。
"""

from __future__ import annotations

import datetime as _dt
import json
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .tracks import SOURCE_MANUAL, PlacedBox, TrackedClip

# 操作の種類（9.3 の一式）
OP_ADD = "add"          # 隠れていない顔に箱を足す
OP_DELETE = "delete"    # 顔でないものを隠している箱を消す
OP_ADJUST = "adjust"    # 位置・大きさを直す
OP_HOLD = "hold"        # 画面外に出る人をもう少し覆い続ける
OP_SPLIT = "split"      # 入れ替わったトラックを分ける
OP_CONFIRM = "confirm"  # 見て問題なかった
OP_CUT = "cut"          # 機械にも人にも直せない。区間ごと落とす
OP_UNDO = "undo"        # 直前の操作を打ち消す

# アンカーからこの倍率までを「同じ対象」とみなす（箱の長辺に対して）。
ANCHOR_RADIUS_RATIO = 1.5

# hold が「元にする箱」を探して遡る上限（秒）。
HOLD_SCAN_SEC = 2.0


@dataclass
class Operation:
    """1 回の操作。時刻は素材の PTS（耐久性のある座標）。"""

    op: str
    clip: str
    start: float
    end: float
    data: Dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    at: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))

    def to_json(self) -> Dict:
        return {
            "id": self.id, "at": self.at, "op": self.op, "clip": self.clip,
            "start": round(self.start, 6), "end": round(self.end, 6), **self.data,
        }

    @classmethod
    def from_json(cls, raw: Dict) -> "Operation":
        known = {"id", "at", "op", "clip", "start", "end"}
        return cls(
            op=raw["op"], clip=raw["clip"],
            start=float(raw["start"]), end=float(raw["end"]),
            data={k: v for k, v in raw.items() if k not in known},
            id=raw.get("id", uuid.uuid4().hex[:12]),
            at=raw.get("at", ""),
        )


class ManualEdits:
    """faces_manual.jsonl の読み書き。追記のみ。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.operations: List[Operation] = []
        if self.path.is_file():
            self._load()

    def _load(self) -> None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                self.operations.append(Operation.from_json(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue  # 壊れた行は飛ばす。ここで全部を失うほうが困る

    def append(self, operation: Operation) -> Operation:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(operation.to_json(), ensure_ascii=False) + "\n")
        self.operations.append(operation)
        return operation

    def undo_last(self, clip: Optional[str] = None) -> Optional[Operation]:
        """直前の（打ち消されていない）操作を打ち消す。"""
        cancelled = {
            op.data.get("target") for op in self.operations if op.op == OP_UNDO
        }
        for operation in reversed(self.operations):
            if operation.op == OP_UNDO or operation.id in cancelled:
                continue
            if clip is not None and operation.clip != clip:
                continue
            return self.append(Operation(
                op=OP_UNDO, clip=operation.clip,
                start=operation.start, end=operation.end,
                data={"target": operation.id},
            ))
        return None

    def effective(self, clip: Optional[str] = None) -> List[Operation]:
        """打ち消されていない操作だけを、記録順に返す。"""
        cancelled = {
            op.data.get("target") for op in self.operations if op.op == OP_UNDO
        }
        return [
            op for op in self.operations
            if op.op != OP_UNDO
            and op.id not in cancelled
            and (clip is None or op.clip == clip)
        ]


def _near(box: PlacedBox, anchor: Sequence[float], radius: Optional[float] = None) -> bool:
    """箱がアンカー点の近くにあるか。ID ではなく位置で対象を決める。"""
    cx, cy = box.center
    limit = radius if radius is not None else max(box.w, box.h) * ANCHOR_RADIUS_RATIO
    return ((cx - anchor[0]) ** 2 + (cy - anchor[1]) ** 2) ** 0.5 <= limit


def _keyframe_box(keys: Sequence[Sequence[float]], pts: float) -> Optional[Tuple[float, ...]]:
    """キーフレーム列 [[pts, x, y, w, h], ...] から pts の位置を補間する。

    1 点しか無くても使える（その値をそのまま使う）。9.3 のとおり、人に 2 点
    打たせるのは工数が 2 倍なので、1 点で成立することを優先している。
    """
    if not keys:
        return None
    ordered = sorted(keys, key=lambda k: k[0])
    if len(ordered) == 1 or pts <= ordered[0][0]:
        return tuple(ordered[0][1:5])
    if pts >= ordered[-1][0]:
        return tuple(ordered[-1][1:5])
    for a, b in zip(ordered, ordered[1:]):
        if a[0] <= pts <= b[0]:
            span = b[0] - a[0]
            u = 0.0 if span <= 0 else (pts - a[0]) / span
            return tuple(a[i] + (b[i] - a[i]) * u for i in range(1, 5))
    return tuple(ordered[-1][1:5])


def apply_manual(
    tracked: TrackedClip,
    operations: Sequence[Operation],
    *,
    negative_regions: Sequence[Dict] = (),
) -> TrackedClip:
    """自動の結果に人手修正を重ねる。人手が常に勝つ。

    - add は自動検出と**和**（人が足したからといって自動の分を消さない）
    - delete は明示的な否定。単に書かないのではなく「描かない」と記録してある
    - adjust / hold は対象の箱を置き換える
    """
    fps = tracked.fps
    boxes: Dict[int, List[PlacedBox]] = {f: list(v) for f, v in tracked.boxes.items()}

    def frame_range(start: float, end: float) -> range:
        return range(int(round(start * fps)), max(int(round(end * fps)), int(round(start * fps)) + 1))

    # 恒常的な誤検出領域（ポスター・鏡・人形）は、毎回消す手間を省いて先に落とす。
    for region in negative_regions:
        x0, y0, x1, y1 = region["rect"]
        lo = region.get("start", 0.0)
        hi = region.get("end", tracked.frames_total / fps)
        for f in frame_range(lo, hi):
            if f not in boxes:
                continue
            boxes[f] = [
                b for b in boxes[f]
                if not (x0 <= b.center[0] <= x1 and y0 <= b.center[1] <= y1)
            ]

    for op in operations:
        if op.op == OP_ADD:
            keys = op.data.get("keyframes") or []
            for f in frame_range(op.start, op.end):
                found = _keyframe_box(keys, f / fps)
                if not found:
                    continue
                x, y, w, h = found
                boxes.setdefault(f, []).append(PlacedBox(
                    x=x, y=y, w=w, h=h, source=SOURCE_MANUAL,
                    track_id=op.id, score=1.0,
                ))
        elif op.op == OP_DELETE:
            anchor = op.data.get("anchor")
            for f in frame_range(op.start, op.end):
                if f not in boxes:
                    continue
                if anchor is None:
                    boxes[f] = []
                else:
                    boxes[f] = [b for b in boxes[f] if not _near(b, anchor)]
        elif op.op == OP_ADJUST:
            anchor = op.data.get("anchor")
            factor = float(op.data.get("scale", 1.0))
            dx = float(op.data.get("dx", 0.0))
            dy = float(op.data.get("dy", 0.0))
            for f in frame_range(op.start, op.end):
                updated = []
                for b in boxes.get(f, []):
                    if anchor is None or _near(b, anchor):
                        # uncertain は落とさない。人が位置を触ったことは、
                        # 推定の不確かさが消えたことを意味しない。落とすと
                        # 1 フレームのナッジで区間全体が要確認から消える。
                        moved = replace(b, x=b.x + dx, y=b.y + dy,
                                        source=SOURCE_MANUAL)
                        updated.append(moved.resized(factor) if factor != 1.0 else moved)
                    else:
                        updated.append(b)
                boxes[f] = updated
        elif op.op == OP_HOLD:
            anchor = op.data.get("anchor")
            start_f = int(round(op.start * fps))
            # 遡る距離に上限を切る。顔が一度も検出されていない場所で押すと
            # クリップ先頭まで線形に遡り、しかも refresh のたびに再実行される。
            floor_f = max(-1, start_f - int(round(HOLD_SCAN_SEC * fps)))
            source: Optional[PlacedBox] = None
            for f in range(start_f, floor_f, -1):
                candidates = [b for b in boxes.get(f, []) if anchor is None or _near(b, anchor)]
                if candidates:
                    source = candidates[0]
                    break
            if source is None:
                continue
            for f in frame_range(op.start, op.end):
                boxes.setdefault(f, []).append(
                    replace(source, source=SOURCE_MANUAL, uncertain=False)
                )

    return TrackedClip(
        fps=tracked.fps, width=tracked.width, height=tracked.height,
        frames_total=tracked.frames_total, boxes=boxes,
        tracks=tracked.tracks, scene_breaks=tracked.scene_breaks,
    )


def manual_cut_intervals(operations: Sequence[Operation]) -> List[Tuple[float, float]]:
    """人が「ここは落とす」と決めた区間（C-2 への手動の導線）。"""
    return [(op.start, op.end) for op in operations if op.op == OP_CUT]


def shrunk_intervals(operations: Sequence[Operation]) -> List[Tuple[float, float]]:
    """人が箱を小さくした区間。

    縮小は人手操作で唯一、素顔を出しうる操作。しかも結果の箱は
    source=manual / uncertain=False になるので sites.py のどの分岐にも
    入らず、確認済みを無効に戻してもサイトが生まれない。明示的に拾う。

    op.data の未知キーは Operation.data に落ちるだけなので、shrunk を
    足しても faces_manual.jsonl の形式は変わらない（tamako edit は無改造）。
    """
    return [
        (op.start, op.end) for op in operations
        if op.op == OP_ADJUST
        and (bool(op.data.get("shrunk")) or float(op.data.get("scale", 1.0)) < 1.0)
    ]


# ---------------------------------------------------------------- 確認済み

SIGNATURE_W, SIGNATURE_H = 32, 18


def coverage_signature(
    tracked: TrackedClip, start: float, end: float, mask_alpha: np.ndarray,
    *, mask_scale: float = 1.0, offset_y: float = 0.0, step: int = 1,
) -> str:
    """区間で実際に覆ったアルファの和集合を、32x18 のビット列にして返す。

    箱ではなくアルファで取るのが要点。隠しているのは箱ではないので、箱で
    比べると「丸いステッカーの角から顔が出ている」変化を見逃す。

    範囲は build_sites と同じ半開区間 [start, end)、step は 1。以前は終端が
    1 フレーム長く、しかも 3 フレームに 2 枚を署名から落としていたので、
    落ちたフレームで被覆が減っても確認済みが維持されてしまっていた。
    """
    fps = tracked.fps
    grid = np.zeros((SIGNATURE_H, SIGNATURE_W), dtype=bool)
    alpha = mask_alpha >= 128
    first = int(round(start * fps))
    last = max(int(round(end * fps)), first + 1)  # 1 フレーム未満でも 1 枚は見る
    for f in range(first, last, max(1, step)):
        for box in tracked.at_frame(f):
            cx, cy = box.center
            cy += box.h * offset_y
            w, h = box.w * mask_scale, box.h * mask_scale
            x0 = (cx - w / 2) / tracked.width * SIGNATURE_W
            y0 = (cy - h / 2) / tracked.height * SIGNATURE_H
            x1 = (cx + w / 2) / tracked.width * SIGNATURE_W
            y1 = (cy + h / 2) / tracked.height * SIGNATURE_H
            gx0, gy0 = max(0, int(x0)), max(0, int(y0))
            gx1, gy1 = min(SIGNATURE_W, int(np.ceil(x1))), min(SIGNATURE_H, int(np.ceil(y1)))
            if gx1 <= gx0 or gy1 <= gy0:
                continue
            patch = _resize_bool(alpha, gx1 - gx0, gy1 - gy0)
            grid[gy0:gy1, gx0:gx1] |= patch
    return "".join("1" if v else "0" for v in grid.flatten())


def _resize_bool(alpha: np.ndarray, width: int, height: int) -> np.ndarray:
    """真偽の配列を最近傍で目的の大きさにする（cv2 に頼らない）。"""
    h, w = alpha.shape
    ys = (np.arange(height) * h // max(1, height)).clip(0, h - 1)
    xs = (np.arange(width) * w // max(1, width)).clip(0, w - 1)
    return alpha[np.ix_(ys, xs)]


def confirmation_still_valid(old: str, new: str) -> bool:
    """確認済みを維持してよいか。**被覆が減っていなければ維持する。**

    箱の一致で判定すると、閾値を 0.01 動かしただけで全部の確認が無効になる。
    しかも人が設定を変える動機の大半は安全側（scale を上げる・閾値を下げる）
    なので、最も安全な操作が最も確認を壊すという最悪の組み合わせになる。
    新しい被覆が古い被覆の上位集合なら、新たな漏れは原理的に発生しない。
    """
    if len(old) != len(new):
        return False
    return not any(o == "1" and n == "0" for o, n in zip(old, new))


def confirmed_intervals(
    operations: Sequence[Operation], signatures: Dict[str, str]
) -> List[Tuple[float, float]]:
    """まだ有効な確認済み区間。被覆が減った区間は外れる。"""
    result = []
    for op in operations:
        if op.op != OP_CONFIRM:
            continue
        old = op.data.get("signature", "")
        new = signatures.get(op.id)
        if new is None or confirmation_still_valid(old, new):
            result.append((op.start, op.end))
    return result


def load_negative_regions(path: str | Path) -> List[Dict]:
    """恒常的な誤検出領域（ポスター・鏡・人形）。毎回「消す」を繰り返さないため。"""
    target = Path(path)
    if not target.is_file():
        return []
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data.get("regions", []) if isinstance(data, dict) else list(data)
