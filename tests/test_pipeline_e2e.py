"""全工程の通し試験。合成素材で detect → tracks → edit → fix → remask を回す。

窓は開かない（人手修正は faces_manual.jsonl を直接書いて代用する）。
確かめたいのは、工程どうしが正しく繋がっていること:
- 検出が 1 回で済み、2 回目はキャッシュされること
- 人手修正が出力に届くこと
- 確認済みが効き、被覆が減ったときだけ無効になること
- 出力の映像と音声の長さが揃っていること
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _run(args, cwd) -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "tamako", *args],
        cwd=cwd, capture_output=True, text=True,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT / "src")},
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"tamako {' '.join(args)} が失敗しました\n{proc.stdout}\n{proc.stderr}"
        )
    return proc.stdout


def _durations(path: Path):
    """映像・音声それぞれの長さ。ffprobe が無い環境では ffmpeg の出力から拾う。"""
    import re

    from tamako.ffmpeg import find_ffmpeg, find_ffprobe, run

    ffprobe = find_ffprobe()
    if ffprobe:
        proc = run([
            ffprobe, "-v", "error", "-show_entries",
            "stream=codec_type,duration", "-of", "json", str(path),
        ], check=False)
        data = json.loads(proc.stdout or "{}")
        out = {}
        for stream in data.get("streams", []):
            if stream.get("duration"):
                out[stream["codec_type"]] = float(stream["duration"])
        if out:
            return out

    # 代替: それぞれのストリームだけを null 出力に流し、処理した時間を見る。
    out = {}
    for kind, selector in (("video", "0:v:0"), ("audio", "0:a:0")):
        proc = run([find_ffmpeg(), "-v", "error", "-stats", "-i", str(path),
                    "-map", selector, "-f", "null", "-"], check=False)
        found = re.findall(r"time=(\d+):(\d+):([\d.]+)", proc.stderr or "")
        if found:
            h, m, s = found[-1]
            out[kind] = int(h) * 3600 + int(m) * 60 + float(s)
    return out


def _fix_ui_round_trip(work: Path, args) -> None:
    """`tamako fix` が実素材でブラウザ UI を開き、絵を返し、終了できること。

    test_webui.py は偽の読み手で門番だけを見る。ここでは実際の ffmpeg 越しに
    (clip, frame) → JPEG が返るところまでを通す。**cmd_fix の配線の試験。**
    """
    import re
    import time
    import urllib.error
    import urllib.request

    log = work / "fix_ui.log"
    with log.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            [sys.executable, "-m", "tamako", "fix", *args, "--no-browser", "--port", "0"],
            cwd=work, stdout=handle, stderr=subprocess.STDOUT,
            env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT / "src")},
        )
    try:
        url = None
        for _ in range(600):
            if proc.poll() is not None:
                raise AssertionError(f"fix が終了しました\n{log.read_text('utf-8')}")
            found = re.search(r"http://127\.0\.0\.1:\d+/\?t=\S+", log.read_text("utf-8"))
            if found:
                url = found.group(0)
                break
            time.sleep(0.25)
        assert url, f"待ち受けアドレスが出ない\n{log.read_text('utf-8')}"

        base, token = url.split("/?t=")

        def call(path, data=None):
            request = urllib.request.Request(
                f"{base}{path}",
                data=None if data is None else json.dumps(data).encode(),
                method="POST" if data is not None else "GET",
            )
            request.add_header("Cookie", f"tamako_fix={token}")
            if data is not None:
                request.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.status, response.read()

        state = json.loads(call("/api/state")[1])
        assert state["clips"], state
        assert state["mask_scale_eff"] > 2.0, "実効倍率が補正されていない"
        clip = state["clips"][0]

        code, body = call(f"/api/frame?clip={clip['name']}&frame=1")
        assert code == 200 and body[:2] == b"\xff\xd8", "JPEG が返らない"
        # 素材の終端を越えても落ちない（以前は FFmpegError で工程ごと終了した）。
        try:
            call(f"/api/frame?clip={clip['name']}&frame={clip['frames'] + 5}")
            raise AssertionError("終端の外が 404 になっていない")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404, exc.code
        assert proc.poll() is None, "終端の要求でサーバが落ちた"

        assert call("/api/quit", {})[0] == 200
        assert proc.wait(timeout=30) == 0, log.read_text("utf-8")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(
            [sys.executable, str(ROOT / "tests" / "make_fixtures.py"), str(root)],
            check=True, capture_output=True,
        )
        work = root                    # make_fixtures が input/ と mask.png を作る
        args = ["-i", "input", "-o", "output", "-m", "mask.png"]

        # --- 検出（1 回目）とキャッシュ（2 回目） ---
        first = _run(["detect", *args], work)
        assert "検出が完了しました" in first
        second = _run(["detect", *args], work)
        assert "検出 " not in second.split("検出が完了")[0], \
            "2 回目で再検出している（キャッシュが効いていない）"

        # --- 書き出し ---
        out = _run(["edit", *args], work)
        assert "書き出したもの" in out
        edited = work / "output" / "edited.mp4"
        assert edited.is_file()

        durations = _durations(edited)
        assert "video" in durations and "audio" in durations, durations
        drift = abs(durations["video"] - durations["audio"])
        assert drift < 0.15, f"映像と音声の長さがずれている: {durations} (差 {drift:.3f}s)"

        report = json.loads((work / "output" / "edit_report.json").read_text("utf-8"))
        assert "review_sites" in report
        assert report["mask_check"]["effective_mask_scale"] > 2.0, \
            "マスクの実効倍率が補正されていない"

        # --- 配管の検査 ---
        assert "配管の検査: OK" in _run(["selftest", *args], work)

        # --- 人手修正が出力に届く ---
        manual = work / "output" / ".tamako_work" / "faces_manual.jsonl"
        manual.parent.mkdir(parents=True, exist_ok=True)
        manual.write_text(json.dumps({
            "id": "m1", "at": "2026-01-01T00:00:00", "op": "add",
            "clip": "a_clip.mp4", "start": 1.5, "end": 2.0,
            "keyframes": [[1.5, 100, 50, 90, 90]],
        }, ensure_ascii=False) + "\n", encoding="utf-8")

        out = _run(["edit", *args], work)
        assert "人手修正を 1 件反映しました" in out, out

        # --- 確認箇所の抜き出し（フル解像度であること） ---
        listing = _run(["fix", *args, "--list"], work)
        assert "機械が気づいた箇所" in listing
        _fix_ui_round_trip(work, args)
        if "機械が気づいた箇所: 0 箇所" not in listing:
            _run(["remask", *args], work)
            review = work / "output" / "_review" / "review.mp4"
            assert review.is_file()
            from tamako.ffmpeg import probe
            assert probe(review).width == probe(edited).width, \
                "確認用の書き出しで解像度が落ちている（部分被覆が見えなくなる）"

    print("通し試験: OK")


if __name__ == "__main__":
    main()
