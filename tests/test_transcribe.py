"""字幕の組み立て（折り返し・分割・SRT 入出力）の単体試験。

文字起こしそのものはモデルの取得に通信が要るのでここでは扱わない。
モデルに依存しない部分は全部ここで固めておく。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tamako.transcribe import (  # noqa: E402
    Cue, format_srt, read_srt, split_long_cue, srt_timestamp, wrap_japanese, write_srt,
)


def test_srt_timestamp_format():
    assert srt_timestamp(0) == "00:00:00,000"
    assert srt_timestamp(1.5) == "00:00:01,500"
    assert srt_timestamp(61.25) == "00:01:01,250"
    assert srt_timestamp(3661.999) == "01:01:01,999"
    # 負の時刻は 0 に丸める。
    assert srt_timestamp(-3) == "00:00:00,000"


def test_wrap_japanese_splits_by_character_count():
    assert wrap_japanese("あいうえお", 10) == ["あいうえお"]
    assert wrap_japanese("あいうえおかきくけこさ", 10) == ["あいうえおかきくけこ", "さ"]


def test_wrap_japanese_avoids_leading_punctuation():
    # 10 文字目で切ると「、」が行頭に来るので 1 文字送る。
    lines = wrap_japanese("あいうえおかきくけ、こさしす", 9)
    assert lines[0] == "あいうえおかきくけ、"
    assert lines[1] == "こさしす"


def test_wrap_japanese_avoids_trailing_open_bracket():
    lines = wrap_japanese("あいうえおかきく「けこさしす", 9)
    assert not lines[0].endswith("「")


def test_split_long_cue_divides_time_proportionally():
    cue = Cue(start=0.0, end=10.0, text="あ" * 100)
    pieces = split_long_cue(cue, max_chars=20, max_lines=2)
    assert len(pieces) == 3  # 40 文字ずつ → 40/40/20
    assert abs(pieces[0].start - 0.0) < 1e-6
    assert abs(pieces[0].end - 4.0) < 1e-6
    assert abs(pieces[-1].end - 10.0) < 1e-6
    assert "".join(p.text for p in pieces) == cue.text


def test_split_long_cue_leaves_short_cue_alone():
    cue = Cue(start=1.0, end=2.0, text="短い")
    assert split_long_cue(cue, 20, 2) == [cue]


def test_format_srt_numbers_sequentially():
    cues = [Cue(0.0, 1.0, "ひとつめ"), Cue(1.0, 2.0, "ふたつめ")]
    text = format_srt(cues, max_chars_per_line=20)
    assert text.startswith("1\n00:00:00,000 --> 00:00:01,000\nひとつめ")
    assert "\n2\n00:00:01,000 --> 00:00:02,000\nふたつめ" in text


def test_format_srt_skips_empty_text():
    cues = [Cue(0.0, 1.0, "   "), Cue(1.0, 2.0, "本文")]
    text = format_srt(cues)
    assert text.count("-->") == 1
    assert "本文" in text


def test_write_and_read_srt_round_trip():
    cues = [Cue(0.0, 1.25, "こんにちは"), Cue(2.0, 3.5, "さようなら")]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.srt"
        write_srt(cues, path)
        restored = read_srt(path)
    assert len(restored) == 2
    assert restored[0].text == "こんにちは"
    assert abs(restored[0].end - 1.25) < 1e-3
    assert abs(restored[1].start - 2.0) < 1e-3


def test_read_srt_accepts_multi_line_cue():
    body = "1\n00:00:00,000 --> 00:00:02,000\n一行目\n二行目\n\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "in.srt"
        path.write_text(body, encoding="utf-8")
        cues = read_srt(path)
    assert cues[0].text == "一行目\n二行目"


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
