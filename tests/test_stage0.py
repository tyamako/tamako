"""段階 0（既存欠陥の修正）の試験。

- expanded() が画面端でクランプしないこと（クランプは顔を露出させるバグだった）
- mask.png の実効被覆と scale の自動補正
- 設定の未知キーがネストの中でもエラーになること
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np


def test_expanded_no_clamp() -> None:
    from tamako.faces import FaceBox

    # 左端の顔: 本来 -30〜130 を覆うべき。クランプがあると 0〜160 にずれていた。
    box = FaceBox(10, 400, 80, 80, 0.9).expanded(2.0)
    assert abs(box.x - (-30.0)) < 1e-9, f"左端で平行移動している: x={box.x}"
    assert abs(box.w - 160.0) < 1e-9

    # 右端の顔: クランプがあると幅が 90 に圧縮され、マスク画像が歪んでいた。
    box = FaceBox(1870, 400, 80, 80, 0.9).expanded(2.0)
    assert abs(box.w - 160.0) < 1e-9, f"右端で圧縮されている: w={box.w}"
    assert abs(box.x - 1830.0) < 1e-9

    # offset_y は維持
    box = FaceBox(100, 100, 80, 80, 0.9).expanded(2.0, offset_y=-0.1)
    assert abs((box.y + box.h / 2) - (140 - 8)) < 1e-9


def test_composite_at_edge() -> None:
    """画面外にはみ出す箱でも、画面内の部分は正しく合成される。"""
    from tamako.faces import FaceBox
    from tamako.overlay import composite

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.full((10, 10, 4), 255, dtype=np.uint8)  # 全面不透明の白
    composite(frame, mask, FaceBox(-30, 40, 60, 20, 1.0))
    # x: -30〜30 のうち画面内は 0〜30。左端の列まで塗られていること。
    assert frame[50, 0, 0] == 255, "画面左端が塗られていない"
    assert frame[50, 29, 0] == 255
    assert frame[50, 31, 0] == 0, "箱の外まで塗られている"


def test_effective_coverage() -> None:
    from tamako.overlay import effective_coverage, effective_scale

    # 全面不透明 → 被覆はほぼ 1.0、補正は等倍
    solid = np.full((100, 100, 4), 255, dtype=np.uint8)
    cov = effective_coverage(solid)
    assert cov > 0.97, f"全面不透明の被覆が低すぎる: {cov}"
    assert abs(effective_scale(2.0, solid) - 2.0 / cov) < 1e-9

    # 円形ステッカー → 内接矩形は約 0.70。補正で scale が約 1.4 倍になる。
    circle = np.zeros((200, 200, 4), dtype=np.uint8)
    yy, xx = np.mgrid[0:200, 0:200]
    inside = (xx - 100) ** 2 + (yy - 100) ** 2 <= 99**2
    circle[inside] = 255
    # 厳密な内接正方形は 0.707 だが、不透明率 99% の許容で少し大きめに出る。
    cov = effective_coverage(circle)
    assert 0.65 < cov < 0.80, f"円形の被覆が想定外: {cov}"
    scale = effective_scale(2.0, circle)
    assert 2.4 < scale < 3.1, f"円形の補正が想定外: {scale}"

    # ほぼ全部透明 → 下限で頭打ちし、暴走しない
    tiny = np.zeros((100, 100, 4), dtype=np.uint8)
    tiny[48:52, 48:52] = 255
    assert effective_scale(2.0, tiny) <= 2.0 / 0.35 + 1e-9


def test_config_rejects_nested_typo() -> None:
    import json
    import tempfile

    from tamako.config import ConfigError, load_config

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.json"

        # ネストした綴り間違いは必ずエラーになる（以前は黙って無視されていた）
        path.write_text(json.dumps({"mask": {"score_treshold": 0.25}}), encoding="utf-8")
        try:
            load_config(path)
        except ConfigError as exc:
            assert "mask.score_treshold" in str(exc)
        else:
            raise AssertionError("ネストした未知キーがエラーにならなかった")

        # 正しいキーは通り、cut.detect_width が独立して存在する
        path.write_text(json.dumps({"mask": {"score_threshold": 0.25}}), encoding="utf-8")
        config = load_config(path)
        assert config.section("mask")["score_threshold"] == 0.25
        assert config.section("cut")["detect_width"] == 640


def main() -> None:
    test_expanded_no_clamp()
    test_composite_at_edge()
    test_effective_coverage()
    test_config_rejects_nested_typo()
    print("段階 0 の試験: OK")


if __name__ == "__main__":
    main()
