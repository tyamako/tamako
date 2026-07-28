"""区間演算とカット判定の単体試験。動画に触れないので単体で速く回る。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tamako.segments import (  # noqa: E402
    build_cut_plan, drop_short, intersect, invert, merge, pad, samples_to_intervals, total, union,
)


def approx(intervals, expected, tol=1e-6):
    assert len(intervals) == len(expected), f"{intervals} != {expected}"
    for (a, b), (c, d) in zip(intervals, expected):
        assert abs(a - c) < tol and abs(b - d) < tol, f"{intervals} != {expected}"


def test_merge_joins_overlaps_and_touching():
    approx(merge([(0, 1), (0.5, 2), (3, 4)]), [(0, 2), (3, 4)])
    approx(merge([(0, 1), (1, 2)]), [(0, 2)])
    approx(merge([]), [])
    # 長さゼロの区間は落とす。
    approx(merge([(1, 1), (2, 3)]), [(2, 3)])


def test_merge_bridges_gaps_within_tolerance():
    approx(merge([(0, 1), (1.4, 2)], gap=0.5), [(0, 2)])
    approx(merge([(0, 1), (1.6, 2)], gap=0.5), [(0, 1), (1.6, 2)])


def test_invert_is_complement_within_duration():
    approx(invert([(1, 2)], 5), [(0, 1), (2, 5)])
    approx(invert([], 5), [(0, 5)])
    approx(invert([(0, 5)], 5), [])
    # 尺をはみ出す区間を渡されても外に出ない。
    approx(invert([(4, 9)], 5), [(0, 4)])


def test_intersect_and_union():
    approx(intersect([(0, 3)], [(2, 5)]), [(2, 3)])
    approx(intersect([(0, 1)], [(2, 3)]), [])
    approx(union([(0, 1)], [(2, 3)]), [(0, 1), (2, 3)])
    approx(union([(0, 2)], [(1, 3)]), [(0, 3)])


def test_drop_short_and_pad():
    approx(drop_short([(0, 0.3), (1, 3)], 0.5), [(1, 3)])
    approx(pad([(1, 2)], 0.5, 10), [(0.5, 2.5)])
    # 端では 0 と尺で止まる。
    approx(pad([(0.2, 9.8)], 1.0, 10), [(0.0, 10.0)])
    # 広げた結果くっついたものは統合される。
    approx(pad([(0, 1), (1.5, 2)], 0.3, 5), [(0, 2.3)])


def test_samples_to_intervals_bridges_detection_dropouts():
    # 3fps 相当の標本。1.0 秒だけ抜けているが hold で繋がる。
    times = [0.0, 0.333, 0.666, 1.666, 2.0]
    approx(samples_to_intervals(times, step=0.333, hold=1.0),
           [(0.0, 2.333)])
    # hold が短ければ繋がらず 2 本になる。
    result = samples_to_intervals(times, step=0.333, hold=0.1)
    assert len(result) == 2


def test_cut_plan_mode_any_cuts_silence_or_faceless():
    plan = build_cut_plan(
        duration=10.0,
        silent=[(0.0, 2.0)],
        face_present=[(0.0, 6.0)],
        mode="any",
        min_cut_sec=0.5,
        min_keep_sec=0.5,
        padding_sec=0.0,
    )
    # 0-2 は無音、6-10 は顔なし。どちらも切られ 2-6 が残る。
    approx(plan.keep, [(2.0, 6.0)])
    assert abs(plan.kept_seconds - 4.0) < 1e-6


def test_cut_plan_mode_both_is_conservative():
    plan = build_cut_plan(
        duration=10.0,
        silent=[(0.0, 2.0)],
        face_present=[(0.0, 6.0)],
        mode="both",
        min_cut_sec=0.5,
        min_keep_sec=0.5,
        padding_sec=0.0,
    )
    # 無音かつ顔なしの区間は無いので、何も切られない。
    approx(plan.keep, [(0.0, 10.0)])


def test_cut_plan_ignores_too_short_cuts():
    plan = build_cut_plan(
        duration=10.0,
        silent=[(5.0, 5.2)],
        face_present=[(0.0, 10.0)],
        mode="any",
        min_cut_sec=0.5,
        min_keep_sec=0.5,
        padding_sec=0.0,
    )
    # 0.2 秒のカットは細切れになるだけなので実行しない。
    approx(plan.keep, [(0.0, 10.0)])


def test_cut_plan_drops_tiny_keeps():
    plan = build_cut_plan(
        duration=10.0,
        silent=[(0.0, 4.0), (4.3, 10.0)],
        face_present=[(0.0, 10.0)],
        mode="any",
        min_cut_sec=0.5,
        min_keep_sec=1.0,
        padding_sec=0.0,
    )
    # 残るのは 4.0-4.3 の 0.3 秒だけ。短すぎるので捨てられ、全体が空になる。
    approx(plan.keep, [])
    assert plan.kept_seconds == 0.0


def test_cut_plan_padding_restores_edges():
    plan = build_cut_plan(
        duration=10.0,
        silent=[(0.0, 3.0)],
        face_present=[(0.0, 10.0)],
        mode="any",
        min_cut_sec=0.5,
        min_keep_sec=0.5,
        padding_sec=0.5,
    )
    # 語頭が切れないよう 0.5 秒前から残す。
    approx(plan.keep, [(2.5, 10.0)])


def test_cut_plan_rejects_unknown_mode():
    try:
        build_cut_plan(duration=1.0, silent=[], face_present=[], mode="nope")
    except ValueError as exc:
        assert "nope" in str(exc)
    else:
        raise AssertionError("不正な mode が受理されてしまった")


def test_total():
    assert abs(total([(0, 1), (2, 4)]) - 3.0) < 1e-6


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  ok   {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {test.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
