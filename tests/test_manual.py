"""人手修正の記録・マージ・確認済みの試験。窓は要らない。"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from tamako.manual import (
    OP_ADD, OP_ADJUST, OP_CONFIRM, OP_CUT, OP_DELETE, OP_HOLD,
    ManualEdits, Operation, apply_manual, confirmation_still_valid,
    coverage_signature, manual_cut_intervals,
)
from tamako.tracks import SOURCE_DETECTED, SOURCE_MANUAL, PlacedBox, TrackedClip

FPS = 10.0


def _tracked(boxes_by_frame=None) -> TrackedClip:
    return TrackedClip(fps=FPS, width=640, height=360, frames_total=100,
                       boxes=dict(boxes_by_frame or {}))


def _box(x=100, y=100, w=40, h=40, source=SOURCE_DETECTED) -> PlacedBox:
    return PlacedBox(x=x, y=y, w=w, h=h, source=source, track_id="t1", score=0.9)


def test_add_is_union_with_auto() -> None:
    """人が箱を足しても、自動の箱は消えない（和になる）。"""
    tracked = _tracked({f: [_box()] for f in range(10)})
    op = Operation(op=OP_ADD, clip="a.mp4", start=0.2, end=0.5,
                   data={"keyframes": [[0.2, 300, 200, 50, 50]]})
    merged = apply_manual(tracked, [op])
    boxes = merged.at_frame(3)
    assert len(boxes) == 2, f"自動の箱が消えている: {boxes}"
    assert any(b.source == SOURCE_MANUAL for b in boxes)
    assert any(b.source == SOURCE_DETECTED for b in boxes)
    # 範囲外には出ない
    assert len(merged.at_frame(8)) == 1


def test_add_single_keyframe_spans_range() -> None:
    """キーフレーム 1 点でも区間全体に効く（2 点要求は工数が 2 倍）。"""
    tracked = _tracked({f: [] for f in range(10)})
    op = Operation(op=OP_ADD, clip="a.mp4", start=0.0, end=0.9,
                   data={"keyframes": [[0.4, 10, 20, 30, 40]]})
    merged = apply_manual(tracked, [op])
    for f in range(9):
        assert merged.at_frame(f), f"フレーム {f} に箱が無い"


def test_add_interpolates_between_keyframes() -> None:
    tracked = _tracked({f: [] for f in range(10)})
    op = Operation(op=OP_ADD, clip="a.mp4", start=0.0, end=1.0,
                   data={"keyframes": [[0.0, 0, 0, 40, 40], [1.0, 100, 0, 40, 40]]})
    merged = apply_manual(tracked, [op])
    assert abs(merged.at_frame(5)[0].x - 50) < 1e-6


def test_delete_uses_anchor_not_track_id() -> None:
    """消す対象は位置で決める。ID が変わっても効き続ける。"""
    tracked = _tracked({f: [_box(100, 100), _box(400, 300)] for f in range(10)})
    op = Operation(op=OP_DELETE, clip="a.mp4", start=0.0, end=1.0,
                   data={"anchor": [120, 120]})
    merged = apply_manual(tracked, [op])
    for f in range(10):
        centers = [b.center[0] for b in merged.at_frame(f)]
        assert 120 not in centers
        assert any(abs(c - 420) < 1 for c in centers), "無関係な箱まで消えている"


def test_adjust_scales_and_nudges() -> None:
    tracked = _tracked({0: [_box(100, 100, 40, 40)]})
    op = Operation(op=OP_ADJUST, clip="a.mp4", start=0.0, end=0.1,
                   data={"anchor": [120, 120], "scale": 2.0, "dx": 10, "dy": -5})
    merged = apply_manual(tracked, [op])
    box = merged.at_frame(0)[0]
    assert abs(box.w - 80) < 1e-6 and abs(box.h - 80) < 1e-6
    assert abs(box.center[0] - 130) < 1e-6 and abs(box.center[1] - 115) < 1e-6


def test_hold_extends_last_box() -> None:
    """画面外に出る人を、あと少し覆い続ける。"""
    tracked = _tracked({f: [_box()] for f in range(5)})
    op = Operation(op=OP_HOLD, clip="a.mp4", start=0.5, end=0.9, data={})
    merged = apply_manual(tracked, [op])
    for f in range(5, 9):
        boxes = merged.at_frame(f)
        assert boxes and boxes[0].source == SOURCE_MANUAL, f"{f} が覆われていない"


def test_negative_regions_drop_boxes() -> None:
    """恒常的な誤検出（ポスター・鏡）は最初から出さない。"""
    tracked = _tracked({f: [_box(100, 100), _box(500, 50)] for f in range(5)})
    merged = apply_manual(tracked, [], negative_regions=[
        {"rect": [480, 30, 560, 110]},
    ])
    for f in range(5):
        assert len(merged.at_frame(f)) == 1
        assert abs(merged.at_frame(f)[0].center[0] - 120) < 1


def test_undo_is_a_record_not_a_deletion() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        edits = ManualEdits(Path(tmp) / "m.jsonl")
        first = edits.append(Operation(op=OP_ADD, clip="a.mp4", start=0, end=1,
                                       data={"keyframes": [[0, 1, 2, 3, 4]]}))
        edits.append(Operation(op=OP_DELETE, clip="a.mp4", start=0, end=1,
                               data={"anchor": [5, 5]}))
        assert len(edits.effective()) == 2
        edits.undo_last()
        remaining = edits.effective()
        assert len(remaining) == 1 and remaining[0].id == first.id
        # 記録は消えていない（追記のみ）
        assert len(edits.path.read_text(encoding="utf-8").strip().splitlines()) == 3
        # 読み直しても同じ
        again = ManualEdits(edits.path)
        assert len(again.effective()) == 1


def test_manual_cut_intervals() -> None:
    ops = [
        Operation(op=OP_CUT, clip="a.mp4", start=1.0, end=2.0),
        Operation(op=OP_ADD, clip="a.mp4", start=3.0, end=4.0),
    ]
    assert manual_cut_intervals(ops) == [(1.0, 2.0)]


def test_confirmation_survives_more_coverage() -> None:
    """被覆が増えるだけの変更では、確認済みが無効にならない。

    箱の一致で判定すると、閾値を 0.01 動かしただけで全部の確認が飛ぶ。
    しかも人が設定を変える動機の大半は安全側なので、最も安全な操作が
    最も確認を壊すという最悪の組み合わせになる。
    """
    alpha = np.full((64, 64), 255, dtype=np.uint8)
    small = _tracked({f: [_box(100, 100, 40, 40)] for f in range(10)})
    big = _tracked({f: [_box(90, 90, 60, 60)] for f in range(10)})

    sig_small = coverage_signature(small, 0.0, 0.9, alpha)
    sig_big = coverage_signature(big, 0.0, 0.9, alpha)

    assert confirmation_still_valid(sig_small, sig_big), \
        "箱を大きくしたのに確認済みが無効になった"
    assert not confirmation_still_valid(sig_big, sig_small), \
        "箱を小さくしたのに確認済みが維持された"
    assert confirmation_still_valid(sig_small, sig_small)


def main() -> None:
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"  {name}: OK")
    print("manual の試験: OK")


if __name__ == "__main__":
    main()
