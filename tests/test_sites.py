"""サイト集約・危険度・C-2（覆えない区間の扱い）の試験。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tamako.sites import (
    KIND_ESTIMATED, KIND_NO_MASK, KIND_UNCERTAIN,
    apply_cut_policy, build_sites, uncovered_intervals,
)
from tamako.tracks import (
    SOURCE_DETECTED, SOURCE_EXTRAP, SOURCE_INTERP, PlacedBox, TrackedClip,
)

FPS = 10.0
CLIP = Path("a.mp4")


def _tracked(boxes_by_frame) -> TrackedClip:
    return TrackedClip(fps=FPS, width=640, height=360, frames_total=100,
                       boxes=dict(boxes_by_frame))


def _box(source=SOURCE_DETECTED, uncertain=False) -> PlacedBox:
    return PlacedBox(x=10, y=10, w=40, h=40, source=source,
                     track_id="t1", score=0.9, uncertain=uncertain)


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
