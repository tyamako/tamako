"""ReviewSession と FrameCache の試験。ffmpeg もブラウザも要らない。

FrameCache は reader を差し替えられるので、素材を用意せずに
「終端の外」「幽霊フレーム」「窓の重なり」を直接叩ける。
"""

from __future__ import annotations

import datetime as _dt
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from tamako.ffmpeg import MediaInfo
from tamako.fix import FrameCache, ReviewSession
from tamako.manual import OP_ADJUST, ManualEdits, Operation
from tamako.ordering import Clip
from tamako.sites import KIND_NO_MASK, Site
from tamako.tracks import SOURCE_DETECTED, PlacedBox, TrackedClip

FPS = 10.0
W, H = 64, 36
FRAMES_TOTAL = 50
CLIP_PATH = Path("a.mp4")


def _clip(path: Path = CLIP_PATH) -> Clip:
    info = MediaInfo(path=path, duration=FRAMES_TOTAL / FPS, fps=FPS,
                     width=W, height=H, has_audio=False)
    return Clip(info=info, order=0, sort_time=_dt.datetime(2020, 1, 1),
                sort_basis="test")


def _tracked(boxes=None) -> TrackedClip:
    return TrackedClip(fps=FPS, width=W, height=H, frames_total=FRAMES_TOTAL,
                       boxes=dict(boxes or {}))


def _box(x=20.0, y=10.0, w=8.0, h=8.0) -> PlacedBox:
    return PlacedBox(x=x, y=y, w=w, h=h, source=SOURCE_DETECTED,
                     track_id="t1", score=0.9)


class _Reader:
    """フレーム番号をそのまま画素に書き込む偽の読み手。要求も記録する。"""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def __call__(self, clip: Clip, start_frame: int, count: int):
        self.calls.append((start_frame, count))
        return [
            np.full((H, W, 3), (start_frame + i) % 256, dtype=np.uint8)
            for i in range(count)
        ]


def _mask_file(tmp: Path) -> Path:
    """不透明な 16x16 の PNG。cv2 で書く（load_mask が読めればよい）。"""
    import cv2

    path = tmp / "mask.png"
    image = np.zeros((16, 16, 4), dtype=np.uint8)
    image[:, :, :3] = 255
    image[:, :, 3] = 255
    cv2.imwrite(str(path), image)
    return path


def test_no_cv2_window_entry_point() -> None:
    """cv2 の窓は残さない。**退路を断つための試験。**

    窓を残したままブラウザ UI を書くと、ReviewSession を座標明示 API に
    開いた時点で窓側の全メソッドのシグネチャが変わり、消す予定のコードを
    書き直すことになる。「途中で止めても壊れない」が成立しなくなる。
    """
    import tamako.fix as fix

    for name in ("run_window", "_draw_hud", "_HELP"):
        assert not hasattr(fix, name), f"{name} が復活している"


# ------------------------------------------------------------ FrameCache


def test_frame_at_clip_end_returns_none() -> None:
    """終端の外は例外ではなく None。人が普通にスクラブして踏む場所で
    工程ごと落ちてはいけない（以前は FFmpegError で終了コード 1）。"""
    reader = _Reader()
    cache = FrameCache([_clip()], reader=reader,
                       frames_total={CLIP_PATH: FRAMES_TOTAL})
    assert cache.get(CLIP_PATH, FRAMES_TOTAL - 1) is not None
    assert cache.get(CLIP_PATH, FRAMES_TOTAL) is None
    assert cache.get(CLIP_PATH, FRAMES_TOTAL + 100) is None
    assert cache.get(CLIP_PATH, -1) is None
    assert cache.get(Path("nosuch.mp4"), 0) is None


def test_frame_cache_clamps_to_frames_total() -> None:
    """存在しない番号を要求しない。要求すると expected_frames が最終フレームを
    複製するので「静止した最後の絵」が幽霊フレームとして生える。"""
    reader = _Reader()
    cache = FrameCache([_clip()], reader=reader,
                       frames_total={CLIP_PATH: FRAMES_TOTAL})
    cache.get(CLIP_PATH, FRAMES_TOTAL - 1)
    for start, count in reader.calls:
        assert start >= 0
        assert start + count <= FRAMES_TOTAL, \
            f"終端を越えて要求している: [{start}, {start + count})"


def test_frame_cache_keeps_frames_reloaded_by_newer_windows() -> None:
    """古い窓を捨てるとき、新しい窓が読み直した番号まで消さない。"""
    reader = _Reader()
    cache = FrameCache([_clip()], reader=reader, window_sec=1.0,
                       frames_total={CLIP_PATH: FRAMES_TOTAL})
    # 窓が 5 本になるよう、離れた位置を順に見る。最後にまた 0 付近へ戻る。
    for frame in (0, 12, 24, 36, 2):
        assert cache.get(CLIP_PATH, frame) is not None
    # 最後の窓が読み直した番号は、最初の窓を捨てても残っていなければならない。
    before = len(reader.calls)
    assert cache.get(CLIP_PATH, 2) is not None
    assert len(reader.calls) == before, "読み直しが起きている（窓が食い合った）"


def test_frame_cache_returns_the_requested_frame() -> None:
    """番号 → 画素の対応がずれていない。"""
    cache = FrameCache([_clip()], reader=_Reader(),
                       frames_total={CLIP_PATH: FRAMES_TOTAL})
    for frame in (0, 7, 33, FRAMES_TOTAL - 1):
        image = cache.get(CLIP_PATH, frame)
        assert image is not None and int(image[0, 0, 0]) == frame % 256


# ---------------------------------------------------------- ReviewSession


def _session(tmp: Path, *, boxes=None, ops=(), sites=None) -> ReviewSession:
    edits = ManualEdits(tmp / "faces_manual.jsonl")
    for op in ops:
        edits.append(op)
    session = ReviewSession(
        clips=[_clip()],
        tracked={CLIP_PATH: _tracked(boxes)},
        sites=list(sites if sites is not None else [
            Site(clip=CLIP_PATH, start=0.0, end=1.0, kind=KIND_NO_MASK, risk=0.5)
        ]),
        edits=edits,
        mask_path=_mask_file(tmp),
    )
    session._cache = FrameCache([_clip()], reader=_Reader(),
                                frames_total={CLIP_PATH: FRAMES_TOTAL})
    return session


def test_ops_are_not_double_applied() -> None:
    """dx=+6 はちょうど 6px。生の tracked を渡す契約が守られているか。"""
    with tempfile.TemporaryDirectory() as tmp:
        op = Operation(op=OP_ADJUST, clip="a.mp4", start=0.0, end=0.5,
                       data={"anchor": [24, 14], "dx": 6.0, "scale": 1.5})
        session = _session(Path(tmp), boxes={f: [_box()] for f in range(10)},
                           ops=[op])
        box = session._merged[CLIP_PATH].at_frame(0)[0]
        assert abs(box.center[0] - (24 + 6)) < 1e-6, f"dx が二重: {box}"
        assert abs(box.w - 8 * 1.5) < 1e-6, f"scale が二重: {box.w} (1.5²なら 18)"


def test_negative_regions_applied() -> None:
    """UI の絵からも恒常的な誤検出が落ちている（本番と同じ絵になる）。"""
    with tempfile.TemporaryDirectory() as tmp:
        session = _session(Path(tmp), boxes={f: [_box(20, 10), _box(50, 4)]
                                             for f in range(10)})
        assert len(session._merged[CLIP_PATH].at_frame(0)) == 2
        session.negative_regions = [{"rect": [45, 0, 64, 20]}]
        session.refresh()
        left = session._merged[CLIP_PATH].at_frame(0)
        assert len(left) == 1 and abs(left[0].center[0] - 24) < 1e-6, left


def test_cursor_uses_frame_numbers() -> None:
    """カーソルの内部表現はフレーム番号。pts の丸めが二重にかからない。"""
    with tempfile.TemporaryDirectory() as tmp:
        session = _session(Path(tmp), boxes={f: [_box()] for f in range(20)},
                           sites=[Site(clip=CLIP_PATH, start=0.7, end=1.2,
                                       kind=KIND_NO_MASK, risk=0.5)])
        assert session.current_frame == 7
        session.step(+3)
        assert session.current_frame == 10
        assert abs(session.current_pts - 1.0) < 1e-9


def test_composed_survives_missing_frame() -> None:
    """終端の外では None を返して黙って落ちない。"""
    with tempfile.TemporaryDirectory() as tmp:
        session = _session(Path(tmp), boxes={f: [_box()] for f in range(20)},
                           sites=[Site(clip=CLIP_PATH, start=0.0, end=1.0,
                                       kind=KIND_NO_MASK, risk=0.5)])
        assert session.composed() is not None
        session.frame_offset = FRAMES_TOTAL + 10
        assert session.frame_here() is None
        assert session.composed() is None


def main() -> None:
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"  {name}: OK")
    print("fix の試験: OK")


if __name__ == "__main__":
    main()
