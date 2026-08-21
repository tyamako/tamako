"""サイト集約・危険度・C-2（覆えない区間の扱い）の試験。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tamako.sites import (
    KIND_ESTIMATED, KIND_LOST_TRACK, KIND_NO_MASK, KIND_SHRUNK, KIND_UNCERTAIN,
    apply_cut_policy, build_sites, uncovered_intervals,
)
from tamako.tracks import (
    SOURCE_DETECTED, SOURCE_EXTRAP, SOURCE_INTERP, PlacedBox, Track, TrackedClip,
)

FPS = 10.0
CLIP = Path("a.mp4")


def _tracked(boxes_by_frame, *, scene_breaks=(), tracks=()) -> TrackedClip:
    return TrackedClip(fps=FPS, width=640, height=360, frames_total=100,
                       boxes=dict(boxes_by_frame),
                       tracks=list(tracks), scene_breaks=list(scene_breaks))


def _track(track_id: str, frames) -> Track:
    return Track(track_id=track_id,
                 frames={f: _box(track_id=track_id) for f in frames})


def _box(source=SOURCE_DETECTED, uncertain=False, track_id="t1", x=10) -> PlacedBox:
    return PlacedBox(x=x, y=10, w=40, h=40, source=source,
                     track_id=track_id, score=0.9, uncertain=uncertain)


def test_no_mask_frames_become_one_site() -> None:
    """連続する未被覆フレームが 1 つの区間に畳まれる。"""
    boxes = {f: [_box()] for f in range(50)}
    for f in range(20, 26):
        boxes[f] = []
    sites = build_sites(CLIP, _tracked(boxes), [(0.0, 5.0)], [], duration=5.0)
    no_mask = [s for s in sites if s.kind == KIND_NO_MASK]
    assert len(no_mask) == 1, f"区間に畳まれていない: {len(no_mask)} 件"
    assert abs(no_mask[0].start - 2.0) < 1e-6
    assert abs(no_mask[0].end - 2.6) < 1e-6


def test_sites_only_inside_kept_regions() -> None:
    """捨てる区間の穴は報告しない（出力に出ないので、見せると本物が埋もれる）。"""
    boxes = {f: [] for f in range(50)}
    sites = build_sites(CLIP, _tracked(boxes), [(0.0, 1.0)], [], duration=5.0)
    assert sites, "残す区間の中の穴が報告されていない"
    for site in sites:
        assert site.end <= 1.0 + 1e-6, f"捨てた区間まで報告している: {site}"


def test_voiced_raises_risk() -> None:
    """音が鳴っているのにマスクが無い箇所は、無音のときより危険度が高い。"""
    boxes = {f: [_box()] for f in range(50)}
    for f in range(20, 26):
        boxes[f] = []

    loud = build_sites(CLIP, _tracked(boxes), [(0.0, 5.0)], [], duration=5.0)
    quiet = build_sites(CLIP, _tracked(boxes), [(0.0, 5.0)],
                        [(0.0, 5.0)], duration=5.0)   # 全編無音

    loud_site = next(s for s in loud if s.kind == KIND_NO_MASK)
    quiet_site = next(s for s in quiet if s.kind == KIND_NO_MASK)
    assert loud_site.voiced and not quiet_site.voiced
    assert loud_site.risk > quiet_site.risk, \
        f"音の有無で危険度が変わっていない: {loud_site.risk} vs {quiet_site.risk}"


def test_kind_priority() -> None:
    """未被覆 > 位置不明 > 推定 の順で分類される。"""
    boxes = {
        0: [],                                            # 未被覆
        1: [_box(SOURCE_INTERP, uncertain=True)],         # 位置不明
        2: [_box(SOURCE_EXTRAP)],                         # 推定
        3: [_box(SOURCE_DETECTED)],                       # 問題なし
    }
    sites = build_sites(CLIP, _tracked(boxes), [(0.0, 0.4)], [],
                        duration=1.0, min_site_sec=0.0)
    kinds = {s.kind for s in sites}
    assert kinds == {KIND_NO_MASK, KIND_UNCERTAIN, KIND_ESTIMATED}, kinds
    # 検出できているフレームはサイトにならない
    assert all(not (s.start <= 0.35 < s.end) for s in sites)


def test_manual_boxes_are_not_estimates() -> None:
    """人が置いた箱は推定ではなく決定。サイトとして出し直さない。"""
    from tamako.tracks import SOURCE_MANUAL

    boxes = {f: [_box(SOURCE_MANUAL)] for f in range(20)}
    sites = build_sites(CLIP, _tracked(boxes), [(0.0, 2.0)], [], duration=2.0)
    assert not sites, f"人手の箱が要確認として出ている: {sites}"


def test_confirmed_ranges_are_skipped() -> None:
    """確認済みの区間は出し直さない。毎回同じ箇所を見せるのが破綻の原因。"""
    boxes = {f: [] for f in range(30)}
    all_sites = build_sites(CLIP, _tracked(boxes), [(0.0, 3.0)], [], duration=3.0)
    assert all_sites

    partial = build_sites(CLIP, _tracked(boxes), [(0.0, 3.0)], [],
                          duration=3.0, skip=[(0.0, 2.0)])
    for site in partial:
        assert site.start >= 2.0 - 1e-6, f"確認済みの区間が出ている: {site}"


def test_partial_detection_is_reported() -> None:
    """2 人写っていて 1 人だけ検出できているフレームが一覧に出る。

    フレームに箱が 1 つでもあれば no_mask にしないので、この最も起きやすい
    漏れが、以前は原理的にリストへ現れなかった（サイト 0 件）。
    """
    # b が 20〜29 のあいだ検出できず、30 で別トラックとして取り直される。
    # フレームには常に a の箱があるので、今の 3 種類では何も出ない。
    boxes = {}
    for f in range(50):
        row = [_box(track_id="a", x=10)]
        if not (20 <= f < 30):
            row.append(_box(track_id="b" if f < 20 else "b2", x=300))
        boxes[f] = row
    tracked = _tracked(boxes, tracks=[
        _track("a", range(50)), _track("b", range(20)), _track("b2", range(30, 50)),
    ])

    sites = build_sites(CLIP, tracked, [(0.0, 5.0)], [], duration=5.0)
    lost = [s for s in sites if s.kind == KIND_LOST_TRACK]
    assert lost, f"1 人だけ検出のフレームが報告されていない: {sites}"
    assert abs(lost[0].start - 2.0) < 1e-6 and abs(lost[0].end - 3.0) < 1e-6, lost[0]
    assert "もう 1 人" in lost[0].action()
    assert "b" in lost[0].track_ids


def test_lost_track_ignores_scene_breaks() -> None:
    """場面が変わって人が入れ替わっただけの箇所は出さない。"""
    boxes = {}
    for f in range(50):
        boxes[f] = [_box(track_id="a", x=10)] + (
            [_box(track_id="b", x=300)] if f < 20 else []
        )
    tracked = _tracked(boxes, scene_breaks=[20], tracks=[
        _track("a", range(50)), _track("b", range(20)),
    ])
    sites = build_sites(CLIP, tracked, [(0.0, 5.0)], [], duration=5.0)
    assert not [s for s in sites if s.kind == KIND_LOST_TRACK], sites


def test_lost_track_uses_detections_not_filled_boxes() -> None:
    """補間で埋めた区間は「人数が減った」にしない（それは推定で覆っている）。"""
    boxes = {f: [_box(track_id="a", x=10), _box(SOURCE_INTERP, track_id="b", x=300)]
             for f in range(50)}
    tracked = _tracked(boxes, tracks=[
        _track("a", range(50)),
        _track("b", list(range(20)) + list(range(30, 50))),  # 20〜29 は補間で埋まる
    ])
    sites = build_sites(CLIP, tracked, [(0.0, 5.0)], [], duration=5.0)
    assert not [s for s in sites if s.kind == KIND_LOST_TRACK], sites


def test_site_has_max_length() -> None:
    """長すぎるサイトは分割される。

    上限が無いと 20 秒の no_mask が 1 件でき、「サイト全体に効かせる」を
    既定にした瞬間、静止した箱を 20 秒に効かせることになる。
    """
    boxes = {f: [] for f in range(200)}
    sites = build_sites(CLIP, _tracked(boxes), [(0.0, 20.0)], [],
                        duration=20.0, max_site_sec=5.0)
    assert len(sites) >= 4, f"分割されていない: {[(s.start, s.end) for s in sites]}"
    for site in sites:
        assert site.duration <= 5.0 + 1e-6, f"上限を超えている: {site.duration}"
    # 区間としては隙間なく元の範囲を覆う。
    ordered = sorted(sites, key=lambda s: s.start)
    assert abs(ordered[0].start - 0.0) < 1e-6
    assert abs(ordered[-1].end - 20.0) < 1e-6
    for a, b in zip(ordered, ordered[1:]):
        assert abs(a.end - b.start) < 1e-6, f"隙間がある: {a.end} → {b.start}"


def test_shrunk_box_is_reported() -> None:
    """人が縮めた区間は必ず 1 度は出す。

    縮小した箱は source=manual / uncertain=False なので他のどの分岐にも
    入らない。確認済みを無効に戻しても、サイトが生まれなかった。
    """
    from tamako.tracks import SOURCE_MANUAL

    boxes = {f: [_box(SOURCE_MANUAL)] for f in range(30)}
    assert not build_sites(CLIP, _tracked(boxes), [(0.0, 3.0)], [], duration=3.0)

    sites = build_sites(CLIP, _tracked(boxes), [(0.0, 3.0)], [], duration=3.0,
                        shrunk=[(1.0, 2.0)])
    shrunk = [s for s in sites if s.kind == KIND_SHRUNK]
    assert shrunk, f"縮めた区間が報告されていない: {sites}"
    assert abs(shrunk[0].start - 1.0) < 1e-6 and abs(shrunk[0].end - 2.0) < 1e-6
    # 確認済みにすれば消える（永久に出続けはしない）。
    assert not [
        s for s in build_sites(CLIP, _tracked(boxes), [(0.0, 3.0)], [],
                               duration=3.0, shrunk=[(1.0, 2.0)],
                               skip=[(0.0, 3.0)])
        if s.kind == KIND_SHRUNK
    ]


def test_shrunk_intervals_from_operations() -> None:
    """scale < 1.0 と shrunk フラグの両方を拾う。形式は変えない。"""
    from tamako.manual import OP_ADJUST, Operation, shrunk_intervals

    ops = [
        Operation(op=OP_ADJUST, clip="a.mp4", start=0.0, end=1.0,
                  data={"scale": 0.8}),
        Operation(op=OP_ADJUST, clip="a.mp4", start=2.0, end=3.0,
                  data={"scale": 1.0, "shrunk": True}),
        Operation(op=OP_ADJUST, clip="a.mp4", start=4.0, end=5.0,
                  data={"scale": 1.2}),
    ]
    assert shrunk_intervals(ops) == [(0.0, 1.0), (2.0, 3.0)]


def test_listing_does_not_claim_completeness() -> None:
    """一覧を「網羅」と偽らない。"""
    from tamako.sites import summarize_sites

    boxes = {f: [] for f in range(30)}
    sites = build_sites(CLIP, _tracked(boxes), [(0.0, 3.0)], [], duration=3.0)
    for text in (summarize_sites(sites, out_duration=3.0),
                 summarize_sites([], out_duration=3.0)):
        assert "これで全部ではありません" in text, text


def test_cut_policy_removes_uncovered() -> None:
    """uncovered_policy=cut が、覆えない区間を残す区間から削る。"""
    boxes = {f: [_box()] for f in range(50)}
    for f in range(20, 30):
        boxes[f] = []
    sites = build_sites(CLIP, _tracked(boxes), [(0.0, 5.0)], [], duration=5.0)

    bad = uncovered_intervals(sites)
    assert bad and abs(bad[0][0] - 2.0) < 1e-6 and abs(bad[0][1] - 3.0) < 1e-6

    kept = apply_cut_policy([(0.0, 5.0)], sites, min_keep_sec=0.5)
    # 2.0〜3.0 が抜けて 2 本に割れる
    assert len(kept) == 2, f"削れていない: {kept}"
    assert abs(kept[0][1] - 2.0) < 1e-6 and abs(kept[1][0] - 3.0) < 1e-6
    # 削った区間はもうどの残す区間にも含まれない
    for s, e in kept:
        assert not (s < 3.0 and e > 2.0), f"覆えない区間が残っている: {(s, e)}"


def test_cut_policy_drops_too_short_remainders() -> None:
    """削った結果、短すぎる残りは捨てる（細切れ防止）。"""
    boxes = {f: [_box()] for f in range(50)}
    for f in range(2, 40):
        boxes[f] = []
    sites = build_sites(CLIP, _tracked(boxes), [(0.0, 5.0)], [], duration=5.0)
    kept = apply_cut_policy([(0.0, 5.0)], sites, min_keep_sec=1.0)
    for s, e in kept:
        assert e - s >= 1.0, f"短すぎる区間が残った: {(s, e)}"


def main() -> None:
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"  {name}: OK")
    print("sites の試験: OK")


if __name__ == "__main__":
    main()
