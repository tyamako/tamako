"""解析パス（detect / faces.jsonl）の試験。

合成素材（実写の顔が動く）に対して:
- 検出結果が書け、メタ行・end 行・PTS の格子が正しいこと
- キャッシュが効くこと（素材と設定が同じなら再検出しない）
- 設定を変えたらキャッシュが無効になること
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _make_fixtures(target: Path) -> Path:
    subprocess.run(
        [sys.executable, str(ROOT / "tests" / "make_fixtures.py"), str(target)],
        check=True, capture_output=True,
    )
    return target / "input"


def main() -> None:
    from tamako.detect import (
        DetectParams, detect_clips, detections_path, is_fresh, load_detections,
    )
    from tamako.faces import MODEL_SHA256
    from tamako.ordering import order_clips

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        input_dir = _make_fixtures(tmpdir)
        clips, failures = order_clips(sorted(input_dir.glob("*.mp4")))
        assert clips and not failures

        work = tmpdir / "work"
        paths = detect_clips(
            clips, work_dir=work, score_threshold=0.5, detect_width=640, jobs=1,
        )
        assert set(paths) == {c.path for c in clips}

        # 読み戻し: 顔が写っている合成素材なので、記録があるはず
        det = load_detections(paths[clips[0].path])
        assert det.frames_total > 0
        assert det.records, "検出が 1 件も無い（素材か検出器がおかしい）"
        # PTS の格子: f / fps と一致する
        for index, rec in list(det.records.items())[:10]:
            assert abs(rec.pts - index / det.fps) < 1e-6
        # 座標は素材ピクセル（640x360 の範囲内）
        for rec in det.records.values():
            for x, y, w, h, score in rec.boxes:
                assert -50 <= x <= det.width + 50 and -50 <= y <= det.height + 50
                assert 0 < w <= det.width and 0 < h <= det.height
                assert 0.0 <= score <= 1.0

        # キャッシュ: 2 回目は一瞬で返る
        started = time.monotonic()
        detect_clips(clips, work_dir=work, score_threshold=0.5, detect_width=640, jobs=1)
        took = time.monotonic() - started
        assert took < 1.0, f"キャッシュが効いていない: {took:.2f}s"

        # 設定を変えると無効になる
        params_new = DetectParams(score_threshold=0.3, detect_width=640,
                                  model_sha=MODEL_SHA256)
        jsonl = detections_path(work, clips[0].path)
        assert not is_fresh(jsonl, clips[0].path, params_new), \
            "閾値を変えたのにキャッシュが有効のまま"

        # 書きかけ（end 行なし）は拒否される
        lines = jsonl.read_text(encoding="utf-8").splitlines()
        jsonl.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        try:
            load_detections(jsonl)
        except ValueError:
            pass
        else:
            raise AssertionError("書きかけの検出結果が読めてしまった")

    print("detect の試験: OK")


if __name__ == "__main__":
    main()
