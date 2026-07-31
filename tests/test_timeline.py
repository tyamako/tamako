"""時間軸の契約の試験。

前半は純粋な計算（フレーム数の確定と、出力フレーム ↔ 素材時刻の変換）。
後半は ffmpeg を実際に回し、read_frames の精密トリムが「フレーム番号単位で」
正しいことを確かめる。フレーム画素に番号を焼き込んだ動画を作り、
読み出した画素から番号を復元して照合する。
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from tamako.timeline import Segment, Timeline, build_timeline, segment_frame_count


@dataclass
class _StubClip:
    path: Path


@dataclass
class _StubPlan:
    keep: list


def test_frame_count() -> None:
    assert segment_frame_count(1.0, 30.0) == 30
    assert segment_frame_count(0.984, 30.0) == 30  # 29.52 → 30 に丸める
    assert segment_frame_count(0.01, 30.0) == 1    # 1 枚未満でも 1 枚は出す
    assert segment_frame_count(2.250, 24.0) == 54


def test_out_start_accumulates_in_frames() -> None:
    """out_start は秒の足し算ではなくフレーム数の足し算から導く。

    0.984 秒の区間を 100 個並べると、秒で積むと 98.4 秒だが、フレームで積むと
    30 枚 × 100 = 100.0 秒。この差が音ズレの原因だった。
    """
    clip = _StubClip(Path("a.mp4"))
    plans = [(clip, _StubPlan(keep=[(i * 1.0, i * 1.0 + 0.984) for i in range(100)]))]
    timeline = build_timeline(plans, fps=30.0)
    assert timeline.total_frames == 30 * 100
    assert abs(timeline.segments[-1].out_start - 99 * 1.0) < 1e-9
    # 各区間の out_start_frame は厳密に 30 の倍数
    for i, seg in enumerate(timeline.segments):
        assert seg.out_start_frame == 30 * i


def test_roundtrip() -> None:
    a, b = _StubClip(Path("a.mp4")), _StubClip(Path("b.mp4"))
    plans = [
        (a, _StubPlan(keep=[(2.0, 4.0), (10.0, 10.5)])),
        (b, _StubPlan(keep=[(0.0, 1.0)])),
    ]
    tl = build_timeline(plans, fps=30.0)
    assert tl.total_frames == 60 + 15 + 30

    # 出力フレーム → 素材時刻
    seg, pts = tl.to_source(0)
    assert seg.clip is a and abs(pts - 2.0) < 1e-9
    seg, pts = tl.to_source(59)
    assert seg.clip is a and abs(pts - (2.0 + 59 / 30.0)) < 1e-9
    seg, pts = tl.to_source(60)
    assert seg.clip is a and abs(pts - 10.0) < 1e-9
    seg, pts = tl.to_source(75)
    assert seg.clip is b and abs(pts - 0.0) < 1e-9

    # 素材時刻 → 出力フレーム（往復で一致する）
    for out_index in range(tl.total_frames):
        seg, pts = tl.to_source(out_index)
        back = tl.to_out_indices(seg.clip.path, pts)
        assert out_index in back, f"往復が壊れている: {out_index} → {pts} → {back}"

    # カットされた素材時刻は 0 件
    assert tl.to_out_indices(a.path, 6.0) == []

    # 出力の時刻（秒）からも引ける。2.4 秒は 2 区間目（素材 10.4 秒）、
    # 2.5 秒ちょうどは 3 区間目の先頭。
    seg, pts = tl.out_time_to_source(2.4)
    assert seg.clip is a and abs(pts - 10.4) < 1e-6
    seg, pts = tl.out_time_to_source(2.5)
    assert seg.clip is b and abs(pts - 0.0) < 1e-6
    assert tl.out_time_to_source(999.0) is None


def _write_indexed_video(path: Path, *, frames: int, fps: float, size: int = 64) -> None:
    """フレーム番号を画素に焼き込んだ動画を作る。

    番号を 8bit に量子化して全面ベタ塗りにする。crf を十分下げれば
    復号後も ±数階調に収まるので、丸めて番号を復元できる。
    """
    from tamako.frames import FrameWriter

    with FrameWriter(path, width=size, height=size, fps=fps, crf=10, preset="ultrafast") as w:
        for i in range(frames):
            assert i < 256
            frame = np.zeros((size, size, 3), dtype=np.uint8)
            # 左半分に上位 4bit、右半分に下位 4bit を 16 階調で塗る。
            # 16 刻みなら符号化の劣化（±数階調）で桁が化けない。
            frame[:, : size // 2] = (i // 16) * 16 + 8
            frame[:, size // 2 :] = (i % 16) * 16 + 8
            w.write(frame)


def _decode_index(frame: np.ndarray, total: int) -> int:
    """左右のベタ塗りからフレーム番号を復元する。"""
    size = frame.shape[1]
    high = int(np.median(frame[:, : size // 2]))
    low = int(np.median(frame[:, size // 2 :]))
    index = ((high - 8 + 8) // 16) * 16 + ((low - 8 + 8) // 16)
    assert 0 <= index < total, f"復元した番号が範囲外です: {index}"
    return index


def test_read_frames_precise_trim() -> None:
    """start/duration 指定の読み出しが、フレーム番号単位で正確であること。"""
    from tamako.frames import read_frames

    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "indexed.mp4"
        _write_indexed_video(video, frames=90, fps=30.0)

        # 1.0 秒目から 1.0 秒 = フレーム 30..59 がぴったり出るはず
        _, frames = read_frames(
            video, out_width=64, out_height=64, fps=30.0,
            filters=["fps=30.0"], start=1.0, duration=1.0, expected_frames=30,
        )
        indices = [_decode_index(f, 90) for f in frames]
        assert len(indices) == 30, f"枚数が違う: {len(indices)}"
        assert indices == list(range(30, 60)), f"開始位置がずれた: {indices[:5]}..."

        # 粗いシークが効く後半（start > 2 秒）でも同じ
        _, frames = read_frames(
            video, out_width=64, out_height=64, fps=30.0,
            filters=["fps=30.0"], start=2.5, duration=0.5, expected_frames=15,
        )
        indices = [_decode_index(f, 90) for f in frames]
        assert indices == list(range(75, 90)), f"粗シーク後の位置がずれた: {indices[:5]}..."


def test_read_frames_pads_to_expected() -> None:
    """素材の端で復号が足りなくても、expected_frames は必ず守られる。"""
    from tamako.frames import read_frames

    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "short.mp4"
        _write_indexed_video(video, frames=30, fps=30.0)

        # 素材は 1.0 秒しか無いのに 1.2 秒ぶん要求する → 最終フレームの複製で埋まる
        _, frames = read_frames(
            video, out_width=64, out_height=64, fps=30.0,
            filters=["fps=30.0"], start=0.0, duration=1.2, expected_frames=36,
        )
        got = list(frames)
        assert len(got) == 36, f"枚数が守られていない: {len(got)}"
        assert _decode_index(got[-1], 30) == 29  # 埋めたのは最終フレームの複製


def test_read_frames_rejects_garbage() -> None:
    """動画でないものを読ませたら、黙って 0 枚ではなく必ず例外になる。"""
    from tamako.ffmpeg import FFmpegError
    from tamako.frames import read_frames

    with tempfile.TemporaryDirectory() as tmp:
        bogus = Path(tmp) / "bogus.mp4"
        bogus.write_bytes(b"this is not a video file")
        _, frames = read_frames(bogus, out_width=64, out_height=64, fps=30.0)
        try:
            list(frames)
        except FFmpegError:
            pass
        else:
            raise AssertionError("壊れた入力が例外にならなかった")


def main() -> None:
    test_frame_count()
    test_out_start_accumulates_in_frames()
    test_roundtrip()
    print("純粋計算の試験: OK")
    test_read_frames_precise_trim()
    test_read_frames_pads_to_expected()
    test_read_frames_rejects_garbage()
    print("ffmpeg 読み出しの試験: OK")


if __name__ == "__main__":
    main()
