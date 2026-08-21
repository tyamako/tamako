"""トラック構築と後処理の試験。純粋な計算なので動画は要らない。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tamako.detect import ClipDetections, FrameRecord
from tamako.tracks import (
    SOURCE_DETECTED, SOURCE_EXTRAP, SOURCE_INTERP, TrackConfig,
    build_tracks, track_clip,
)


def _detections(boxes_by_frame, *, fps=30.0, width=640, height=360, total=None):
    records = {
        f: FrameRecord(pts=f / fps, boxes=[list(b) for b in boxes], landmarks=[])
        for f, boxes in boxes_by_frame.items()
    }
    return ClipDetections(
        clip_path=Path("test.jsonl"), fps=fps, width=width, height=height,
        frames_total=total if total is not None else max(boxes_by_frame) + 1,
        records=records,
    )


def test_single_track() -> None:
    """等速で動く 1 人が 1 本のトラックにまとまる。"""
    det = _detections({f: [[100 + f * 4, 100, 40, 40, 0.9]] for f in range(10)})
    tracks = build_tracks(det, TrackConfig())
    assert len(tracks) == 1, f"トラックが分裂した: {len(tracks)}"
    assert sorted(tracks[0].frames) == list(range(10))


def test_two_people_do_not_swap() -> None:
    """すれ違う 2 人が入れ替わらない（全ペア距離の昇順マッチ）。"""
    frames = {}
    for f in range(10):
        frames[f] = [
            [100 + f * 10, 100, 40, 40, 0.9],   # 右へ
            [400 - f * 10, 100, 40, 40, 0.9],   # 左へ
        ]
    tracks = build_tracks(_detections(frames), TrackConfig())
    assert len(tracks) == 2, f"トラック数が想定外: {len(tracks)}"
    for track in tracks:
        xs = [track.frames[f].x for f in sorted(track.frames)]
        # 各トラックの x は単調（入れ替わったらジグザグになる）
        increasing = all(b >= a for a, b in zip(xs, xs[1:]))
        decreasing = all(b <= a for a, b in zip(xs, xs[1:]))
        assert increasing or decreasing, f"トラックが入れ替わった: {xs}"


def test_gap_is_interpolated() -> None:
    """検出が途切れた区間が、前後から補間で埋まる。"""
    frames = {f: [[100 + f * 5, 100, 40, 40, 0.9]] for f in range(20)}
    for f in range(8, 14):          # 6 フレーム落とす
        del frames[f]
    config = TrackConfig(dilate_frames=0)
    clip = track_clip(_detections(frames, total=20), config)

    for f in range(8, 14):
        boxes = clip.at_frame(f)
        assert boxes, f"ギャップ {f} が埋まっていない"
        box = boxes[0]
        assert box.source == SOURCE_INTERP, f"由来が想定外: {box.source}"
        # 補間位置は真値（100 + f*5 が左上、中心は +20）を含んでいること
        true_cx = 100 + f * 5 + 20
        assert box.x <= true_cx <= box.x + box.w, \
            f"補間した箱が真の中心を覆っていない: f={f} box={box}"


def test_interp_grows_with_speed() -> None:
    """同じギャップ長でも、速い動きのほうが箱を大きく広げる。"""
    config = TrackConfig(dilate_frames=0)

    def middle_width(step: int) -> float:
        frames = {f: [[100 + f * step, 100, 40, 40, 0.9]] for f in range(20)}
        for f in range(8, 14):
            del frames[f]
        clip = track_clip(_detections(frames, total=20), config)
        return clip.at_frame(11)[0].w

    slow, fast = middle_width(1), middle_width(20)
    assert fast > slow * 1.5, f"速度で拡大が変わっていない: slow={slow} fast={fast}"


def test_extrapolation_at_end() -> None:
    """トラックの端で外挿し、離れるほど広がる。"""
    frames = {f: [[100 + f * 5, 100, 40, 40, 0.9]] for f in range(10)}
    config = TrackConfig(dilate_frames=0, extrap_frames=15)
    clip = track_clip(_detections(frames, total=40), config)

    assert clip.at_frame(9)[0].source == SOURCE_DETECTED
    near, far = clip.at_frame(11), clip.at_frame(20)
    assert near and near[0].source == SOURCE_EXTRAP
    assert far and far[0].source == SOURCE_EXTRAP
    assert far[0].w > near[0].w, "外挿が伸びても広がっていない"
    # 打ち切りの先には出さない
    assert not clip.at_frame(30), "外挿が上限を超えて続いている"


def test_scene_break_blocks_interpolation() -> None:
    """場面転換を跨いで補間しない（跨ぐとマスクが瞬間移動して両側で漏れる）。"""
    frames = {f: [[100, 100, 40, 40, 0.9]] for f in range(6)}
    frames.update({f: [[500, 200, 40, 40, 0.9]] for f in range(14, 20)})
    det = _detections(frames, total=20)
    det.records[14].scene = 99.0        # ここが場面転換

    clip = track_clip(det, TrackConfig(dilate_frames=0))
    middle = clip.at_frame(10)
    for box in middle:
        # 転換前の位置(100)と後の位置(500)の中間に箱を置いていないこと
        cx = box.x + box.w / 2
        assert not (200 < cx < 400), f"場面転換を跨いで補間している: {box}"


def test_fast_motion_stays_one_track() -> None:
    """箱の幅ぶん動く速さでもトラックが千切れない。

    繋ぎ損なうと速度が取れず、外挿も拡大も効かないまま静かに穴が開く。
    """
    det = _detections({f: [[50 + f * 40, 100, 40, 40, 0.9]] for f in range(10)})
    tracks = build_tracks(det, TrackConfig())
    assert len(tracks) == 1, f"速い動きでトラックが千切れた: {len(tracks)} 本"


def test_uncertain_flag_on_long_extrapolation() -> None:
    """拡大の上限に達した箱には印が付く（C-2 に送る材料）。"""
    frames = {f: [[100 + f * 20, 100, 40, 40, 0.9]] for f in range(6)}
    config = TrackConfig(dilate_frames=0, extrap_frames=30, expand_limit=2.0)
    clip = track_clip(_detections(frames, total=40), config)
    tracks = build_tracks(_detections(frames, total=40), config)
    assert len(tracks) == 1, f"前提が崩れている（トラックが分裂）: {len(tracks)}"
    tail = clip.at_frame(20)
    assert tail and tail[0].uncertain, f"上限に達したのに印が付いていない: {tail}"


def test_dilation_covers_neighbours() -> None:
    """膨張が前後のフレームの位置を取り込む。"""
    frames = {f: [[100 + f * 10, 100, 40, 40, 0.9]] for f in range(20)}
    plain = track_clip(_detections(frames), TrackConfig(dilate_frames=0))
    dilated = track_clip(_detections(frames), TrackConfig(dilate_frames=3))
    a, b = plain.at_frame(10)[0], dilated.at_frame(10)[0]
    assert b.w > a.w, "膨張しても箱が広がっていない"
    # 前後 3 フレームの位置を含んでいる
    assert b.x <= 100 + 7 * 10 and b.x + b.w >= 100 + 13 * 10 + 40


def main() -> None:
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"  {name}: OK")
    print("tracks の試験: OK")


if __name__ == "__main__":
    main()
