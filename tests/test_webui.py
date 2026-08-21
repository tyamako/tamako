"""ブラウザ UI の HTTP 層。実サーバを立てて urllib で叩く。

ffmpeg は使わない（FrameCache に偽の読み手を差す）。ここで確かめたいのは
配管と門番——トークン・Host・Origin・Content-Type・no-store・素材名の解決——
であって、絵の中身ではない。
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from tamako.ffmpeg import MediaInfo
from tamako.fix import FrameCache, ReviewSession
from tamako.manual import ManualEdits
from tamako.ordering import Clip
from tamako.sites import KIND_NO_MASK, Site
from tamako.tracks import SOURCE_DETECTED, PlacedBox, TrackedClip
from tamako.webui import COOKIE_NAME, _App, _Handler, _Server

FPS = 10.0
W, H = 64, 36
FRAMES = 40
NAME = "a.mp4"
CLIP_PATH = Path(NAME)


def _clip() -> Clip:
    return Clip(
        info=MediaInfo(path=CLIP_PATH, duration=FRAMES / FPS, fps=FPS,
                       width=W, height=H, has_audio=False),
        order=0, sort_time=_dt.datetime(2020, 1, 1), sort_basis="test",
    )


def _reader(clip, start_frame, count):
    return [np.full((H, W, 3), (start_frame + i) % 256, dtype=np.uint8)
            for i in range(count)]


def _mask(tmp: Path) -> Path:
    import cv2

    path = tmp / "mask.png"
    image = np.zeros((16, 16, 4), dtype=np.uint8)
    image[:, :, :3] = 255
    image[:, :, 3] = 255
    cv2.imwrite(str(path), image)
    return path


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """303 をそのまま見たい。urllib は既定で追ってしまう。"""

    def redirect_request(self, *args, **kwargs):
        return None


class _Live:
    """立ち上げたサーバ。with で使う。"""

    def __init__(self, tmp: Path) -> None:
        boxes = {f: [PlacedBox(x=10, y=8, w=8, h=8, source=SOURCE_DETECTED,
                               track_id="t1", score=0.9)] for f in range(FRAMES)}
        self.jsonl = tmp / "faces_manual.jsonl"
        session = ReviewSession(
            clips=[_clip()],
            tracked={CLIP_PATH: TrackedClip(fps=FPS, width=W, height=H,
                                            frames_total=FRAMES, boxes=boxes)},
            sites=[Site(clip=CLIP_PATH, start=0.0, end=1.0,
                        kind=KIND_NO_MASK, risk=0.5)],
            edits=ManualEdits(self.jsonl),
            mask_path=_mask(tmp),
        )
        session._cache = FrameCache([_clip()], reader=_reader,
                                    frames_total={CLIP_PATH: FRAMES})
        self.app = _App(session, tmp / "webui.log",
                        Path(__file__).resolve().parent.parent
                        / "src" / "tamako" / "fixui.html")
        self.server = _Server(("127.0.0.1", 0), _Handler)
        self.server.app = self.app
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "_Live":
        self.thread.start()
        self.port = self.server.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()

    def request(self, path, *, data=None, token=True, headers=None, ctype=None,
                follow=True):
        url = f"{self.base}{path}"
        payload = None if data is None else json.dumps(data).encode()
        req = urllib.request.Request(url, data=payload,
                                     method="POST" if data is not None else "GET")
        if token:
            req.add_header("Cookie", f"{COOKIE_NAME}={self.app.token}")
        if payload is not None:
            req.add_header("Content-Type", ctype or "application/json")
        for key, value in (headers or {}).items():
            req.add_header(key, value)
        opener = (urllib.request.build_opener() if follow
                  else urllib.request.build_opener(_NoRedirect))
        try:
            with opener.open(req, timeout=10) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, dict(exc.headers), exc.read()


def _live(func):
    def wrapper():
        with tempfile.TemporaryDirectory() as tmp, _Live(Path(tmp)) as live:
            func(live)
    wrapper.__name__ = func.__name__
    return wrapper


# ---------------------------------------------------------------- 門番


@_live
def test_requires_token(live) -> None:
    assert live.request("/api/state", token=False)[0] == 401
    assert live.request("/api/state")[0] == 200
    # 別のトークンでも通らない。
    assert live.request("/api/state", token=False,
                        headers={"Cookie": f"{COOKIE_NAME}=deadbeef"})[0] == 401


@_live
def test_rejects_bad_host(live) -> None:
    """DNS リバインディング対策。攻撃者のドメインが 127.0.0.1 に解決されても
    同一オリジン扱いにさせない。"""
    assert live.request("/api/state",
                        headers={"Host": f"evil.example:{live.port}"})[0] == 403
    # ポートが違えば同じ機械でも別のサーバ。
    assert live.request("/api/state",
                        headers={"Host": "127.0.0.1:1"})[0] == 403


@_live
def test_rejects_foreign_origin(live) -> None:
    assert live.request("/api/state",
                        headers={"Origin": "http://evil.example"})[0] == 403
    assert live.request("/api/state",
                        headers={"Origin": f"http://127.0.0.1:{live.port}"})[0] == 200


@_live
def test_rejects_form_content_type(live) -> None:
    """フォーム投稿は CSRF の主経路。application/json 以外は受けない。"""
    code, _, _ = live.request("/api/op", data={"op": "add"},
                              ctype="application/x-www-form-urlencoded")
    assert code == 415


@_live
def test_no_cors_headers(live) -> None:
    _, headers, _ = live.request("/api/state")
    assert not any(k.lower().startswith("access-control-") for k in headers)


@_live
def test_clip_name_traversal_rejected(live) -> None:
    for name in ("../../etc/passwd", "a.mp4/../a.mp4", "A.MP4", "b.mp4"):
        code, _, _ = live.request(
            f"/api/frame?clip={urllib.parse.quote(name)}&frame=0")
        assert code == 404, f"{name} が通った"


@_live
def test_body_has_no_exception_detail(live) -> None:
    code, _, body = live.request("/api/frame?clip=a.mp4&frame=notanumber")
    assert code == 404 and body == b'{"error": "clip"}', body


# ---------------------------------------------------------------- 中身


@_live
def test_frames_are_no_store(live) -> None:
    """素顔のフレームをブラウザのディスクキャッシュに残さない。"""
    for mode in ("mask", "raw"):
        code, headers, body = live.request(f"/api/frame?clip={NAME}&frame=3&mode={mode}")
        assert code == 200 and headers["Content-Type"] == "image/jpeg"
        assert headers["Cache-Control"] == "no-store", headers
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert int(headers["Content-Length"]) == len(body)
        assert body[:2] == b"\xff\xd8", "JPEG ではない"


@_live
def test_frame_at_clip_end_is_404_not_crash(live) -> None:
    assert live.request(f"/api/frame?clip={NAME}&frame={FRAMES}")[0] == 404
    assert live.request(f"/api/frame?clip={NAME}&frame={FRAMES - 1}")[0] == 200


@_live
def test_boxes_are_integer_arrays(live) -> None:
    code, _, body = live.request(f"/api/boxes?clip={NAME}&frame=2")
    assert code == 200
    boxes = json.loads(body)
    assert boxes == [[10, 8, 8, 8, 0, 0]], boxes


@_live
def test_op_add_appends_a_line_and_changes_the_picture(live) -> None:
    """**縦の 1 本。** 箱を 1 個足して jsonl に行が増え、絵が変わる。"""
    before = live.request(f"/api/frame?clip={NAME}&frame=5")[2]
    assert not live.jsonl.exists()

    code, _, body = live.request("/api/op", data={
        "op": "add", "clip": NAME, "frame": 5,
        "start_frame": 3, "end_frame": 8, "rect": [30, 12, 16, 16],
    })
    assert code == 200, body
    result = json.loads(body)
    # 遠くの箱は数えない（_near をそのまま当てる）。
    assert result["affected"] == 1, result

    lines = live.jsonl.read_text("utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["op"] == "add" and record["clip"] == NAME
    # 時刻は秒で記録する（形式は変えない）。座標は素材ピクセルのまま。
    assert abs(record["start"] - 0.3) < 1e-6 and abs(record["end"] - 0.8) < 1e-6
    assert record["keyframes"] == [[0.5, 30.0, 12.0, 16.0, 16.0]]

    boxes = json.loads(live.request(f"/api/boxes?clip={NAME}&frame=5")[2])
    assert len(boxes) == 2 and [30, 12, 16, 16, 4, 0] in boxes, boxes
    assert live.request(f"/api/frame?clip={NAME}&frame=5")[2] != before, "絵が変わっていない"
    # 範囲の外は変わらない。
    assert len(json.loads(live.request(f"/api/boxes?clip={NAME}&frame=9")[2])) == 1


@_live
def test_op_reports_affected_neighbours(live) -> None:
    """隣の人物を巻き込む疑いを数で出す。突き合わせではなく _near で数える。"""
    far = json.loads(live.request("/api/op", data={
        "op": "add", "clip": NAME, "frame": 5, "start_frame": 5, "end_frame": 6,
        "rect": [40, 20, 10, 10],           # 既存の箱 (10,8,8,8) から遠い
    })[2])
    assert far["affected"] == 1, far
    near = json.loads(live.request("/api/op", data={
        "op": "add", "clip": NAME, "frame": 20, "start_frame": 20, "end_frame": 21,
        "rect": [12, 10, 10, 10],           # 既存の箱にかぶる
    })[2])
    assert near["affected"] == 2, near


@_live
def test_op_span_is_capped(live) -> None:
    """5 秒を超える指定は切り詰める。長すぎる範囲は漏れを隠す方向に働く。"""
    _, _, body = live.request("/api/op", data={
        "op": "add", "clip": NAME, "frame": 0,
        "start_frame": 0, "end_frame": 10_000, "rect": [1, 1, 8, 8],
    })
    result = json.loads(body)
    assert result["end"] - result["start"] == int(5.0 * FPS), result


@_live
def test_concurrent_ops_are_serialized(live) -> None:
    """8 スレッドで 8 行。追記が食い合わない。"""
    errors = []

    def push(i):
        try:
            code, _, _ = live.request("/api/op", data={
                "op": "add", "clip": NAME, "frame": i,
                "start_frame": i, "end_frame": i + 1, "rect": [i, 1, 8, 8],
            })
            if code != 200:
                errors.append(code)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=push, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    lines = live.jsonl.read_text("utf-8").strip().splitlines()
    assert len(lines) == 8, lines
    for line in lines:
        json.loads(line)          # 行が壊れていない


# ---------------------------------------------------------------- 入口


@_live
def test_bootstrap_moves_token_to_cookie(live) -> None:
    """?t= はブートストラップだけ。受け取ったら Cookie に移して URL から捨てる。"""
    assert live.request("/", token=False)[0] == 401
    code, headers, _ = live.request(f"/?t={live.app.token}", token=False,
                                    follow=False)
    assert code == 303 and headers["Location"] == "/"
    cookie = headers["Set-Cookie"]
    assert "SameSite=Strict" in cookie and "HttpOnly" in cookie
    assert live.app.token in cookie

    code, headers, body = live.request("/")
    assert code == 200 and headers["Content-Type"].startswith("text/html")
    assert "Content-Security-Policy" in headers
    assert b"getScreenCTM" in body, "座標変換を手計算していないか"


@_live
def test_quit_stops_the_server(live) -> None:
    assert not live.app.stopped.is_set()
    assert live.request("/api/quit", data={})[0] == 200
    assert live.app.stopped.wait(2.0)


def main() -> None:

    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"  {name}: OK")
    print("webui の試験: OK")


if __name__ == "__main__":
    main()
