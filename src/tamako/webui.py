"""ブラウザから顔隠しを直す。127.0.0.1 だけに開く HTTP の口。

**認証は Cookie。** `<img>` にリクエストヘッダは付けられないので、ヘッダ方式は
そもそも実装できない。クエリに倒すと `BaseHTTPRequestHandler.log_message` が
リクエスト行を stderr にそのまま出すため、.bat の黒い窓にトークンが流れる。
`SameSite=Strict` + Host 検査（DNS リバインディング対策）+ POST に
`Content-Type: application/json` 必須、で CSRF は塞げる。CORS ヘッダは一切返さない。

**画像は全モード no-store。** 版キー方式には skew の穴があり、`_merged` の
差し替えと版番号の更新の順序次第で「旧版の絵が新版のキーで長時間キャッシュされる」。
この道具でいちばん危険な誤解（隠れていないのに隠れて見える）を、応答性のために
作ることになる。速度はブラウザ側が Blob を持って稼ぐ（セッション中だけ）。

ハートビートは入れない。バックグラウンドタブでタイマーは間引かれ、スリープ復帰で
確実に切れる。作業中に勝手にサーバが死ぬ体験は、防いでいる脅威より遥かに高頻度。
"""

from __future__ import annotations

import hmac
import json
import secrets
import threading
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import cv2

from .fix import FixError, ReviewSession
from .manual import ANCHOR_RADIUS_RATIO, OP_ADD, Operation, _near
from .tracks import (
    SOURCE_DETECTED, SOURCE_DILATE, SOURCE_EXTRAP, SOURCE_INTERP, SOURCE_MANUAL,
)

HTML_PATH = Path(__file__).with_name("fixui.html")
COOKIE_NAME = "tamako_fix"
# add / adjust を効かせられる長さの上限。長すぎる既定値は「箱はあるが顔は
# もういない」を作り、しかも sites.py が要確認から外すので、漏れを隠す方向に働く。
MAX_SPAN_SEC = 5.0
# 箱の由来はコード番号で送る。辞書形式だと 4 人写った 300 フレームで 150KB を
# 超えるが、配列なら 16KB。
SOURCE_CODE = {SOURCE_DETECTED: 0, SOURCE_DILATE: 1, SOURCE_INTERP: 2,
               SOURCE_EXTRAP: 3, SOURCE_MANUAL: 4}


class _App:
    """サーバが持つ状態。編集は 1 本のロックで直列化する。"""

    def __init__(self, session: ReviewSession, log_path: Path,
                 html_path: Optional[Path] = None) -> None:
        self.session = session
        self.token = secrets.token_urlsafe(24)
        # 起動時に 1 回だけ読む。実行中にファイルが消えても動く。
        self.html = (html_path or HTML_PATH).read_text(encoding="utf-8").encode("utf-8")
        self.clips = {c.path.name: c for c in session.clips}
        self.lock = threading.Lock()
        self.log_path = log_path
        self.stopped = threading.Event()

    def log(self, message: str) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")
        except OSError:
            pass

    def fps(self, name: str) -> float:
        return self.session.fps_of(self.clips[name].path)

    def state(self) -> Dict:
        session = self.session
        return {
            "clips": [
                {"name": name, "fps": self.fps(name),
                 "width": clip.info.width, "height": clip.info.height,
                 "frames": session.tracked[clip.path].frames_total}
                for name, clip in self.clips.items()
            ],
            "mask_scale_eff": session.scale_eff,
            "offset_y": session.mask_offset_y,
            "sites": len(session.sites),
            "ops": len(session.edits.effective()),
            "max_span_sec": MAX_SPAN_SEC,
        }

    def boxes(self, name: str, frame: int) -> list:
        with self.lock:
            found = self.session.boxes_at(self.clips[name].path, frame)
        return [[round(b.x), round(b.y), round(b.w), round(b.h),
                 SOURCE_CODE.get(b.source, 0), int(b.uncertain)] for b in found]

    def image(self, name: str, frame: int, mode: str) -> Optional[bytes]:
        clip = self.clips[name]
        with self.lock:
            canvas = (self.session.frame_at(clip.path, frame) if mode == "raw"
                      else self.session.composed_at(clip.path, frame))
        if canvas is None:
            return None
        ok, buffer = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return buffer.tobytes() if ok else None

    def add_box(self, name: str, payload: Dict) -> Dict:
        """箱を足す。**時刻はフレーム番号で受け、記録の直前に秒へ直す。**

        faces_manual.jsonl の形式は変えない（tamako edit は無改造）。
        """
        fps = self.fps(name)
        rect = [float(v) for v in payload["rect"]]
        start = int(payload["start_frame"])
        end = max(start + 1, int(payload["end_frame"]))
        at = min(max(int(payload.get("frame", start)), start), end)
        end = min(end, start + int(round(MAX_SPAN_SEC * fps)))
        operation = Operation(
            op=OP_ADD, clip=name, start=start / fps, end=end / fps,
            data={"keyframes": [[at / fps, *rect]]},
        )
        # 効いた箱の数は突き合わせでなく _near を代表フレームに当てて数える。
        # _near が判定式そのものなので正確で、突き合わせは add や hold で
        # 意味のある数を出せない。2 個以上なら隣の人物を巻き込んでいる疑い。
        anchor = (rect[0] + rect[2] / 2, rect[1] + rect[3] / 2)
        radius = max(rect[2], rect[3]) * ANCHOR_RADIUS_RATIO
        with self.lock:
            self.session.edits.append(operation)
            self.session.refresh()
            here = self.session.boxes_at(self.clips[name].path, at)
        affected = sum(1 for b in here if _near(b, anchor, radius))
        return {"id": operation.id, "affected": affected,
                "start": start, "end": end}


class _Handler(BaseHTTPRequestHandler):
    # 既定の HTTP/1.0 では keep-alive が効かず、フレーム 1 枚ごとに TCP 接続と
    # スレッドが立つ。Content-Length は必ず送る。
    protocol_version = "HTTP/1.1"
    server_version = "tamako"
    sys_version = ""

    @property
    def app(self) -> _App:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args) -> None:
        # 黒い窓にリクエスト行を出さない（ブートストラップの ?t= が流れる）。
        self.app.log(fmt % args)

    # ------------------------------------------------------------ 応答

    def _send(self, code: int, body: bytes = b"",
              ctype: str = "application/json; charset=utf-8",
              extra: Tuple[Tuple[str, str], ...] = ()) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # 全モード no-store。素顔のフレームをブラウザのディスクに残さない。
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in extra:
            self.send_header(key, value)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionError):
            pass  # AbortController で切られただけ

    def _json(self, payload, code: int = HTTPStatus.OK) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def _fail(self, code: int, reason: str) -> None:
        # 応答本文に例外の内容は出さない。短い符丁だけ。
        self._json({"error": reason}, code)

    # ------------------------------------------------------------ 検査

    def _local_host(self, value: Optional[str]) -> bool:
        if not value:
            return False
        parsed = urlparse(value if "//" in value else f"//{value}")
        try:
            port = parsed.port
        except ValueError:
            return False
        return (parsed.hostname in ("127.0.0.1", "localhost")
                and port == self.server.server_address[1])

    def _guard(self, *, post: bool) -> bool:
        # Host 検査。攻撃者のドメインが 127.0.0.1 に解決された瞬間に
        # 同一オリジン扱いになるのを防ぐ（DNS リバインディング）。
        if not self._local_host(self.headers.get("Host")):
            self._fail(HTTPStatus.FORBIDDEN, "host")
            return False
        origin = self.headers.get("Origin")
        if origin is not None and not self._local_host(origin):
            self._fail(HTTPStatus.FORBIDDEN, "origin")
            return False
        if post:
            kind = (self.headers.get("Content-Type") or "").split(";")[0].strip()
            if kind != "application/json":
                self._fail(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "content_type")
                return False
        jar = SimpleCookie(self.headers.get("Cookie") or "")
        given = jar[COOKIE_NAME].value if COOKIE_NAME in jar else ""
        if not hmac.compare_digest(given, self.app.token):
            self._fail(HTTPStatus.UNAUTHORIZED, "token")
            return False
        return True

    def _clip_frame(self, query: Dict) -> Optional[Tuple[str, int]]:
        """clip はファイル名の完全一致でしか解決しない。"""
        name = (query.get("clip") or [""])[0]
        if name not in self.app.clips:
            return None
        try:
            return name, int((query.get("frame") or ["0"])[0])
        except ValueError:
            return None

    # ------------------------------------------------------------ 経路

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/":
            return self._index(query)
        if parsed.path == "/favicon.ico":
            # ブラウザが必ず取りに来る。404 を返すと黒い窓とログが濁る。
            return self._send(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
        if not self._guard(post=False):
            return
        if parsed.path == "/api/state":
            return self._json(self.app.state())
        target = self._clip_frame(query)
        if target is None:
            return self._fail(HTTPStatus.NOT_FOUND, "clip")
        name, frame = target
        if parsed.path == "/api/boxes":
            return self._json(self.app.boxes(name, frame))
        if parsed.path == "/api/frame":
            mode = (query.get("mode") or ["mask"])[0]
            body = self.app.image(name, frame, "raw" if mode == "raw" else "mask")
            if body is None:
                return self._fail(HTTPStatus.NOT_FOUND, "frame")
            return self._send(HTTPStatus.OK, body, "image/jpeg")
        self._fail(HTTPStatus.NOT_FOUND, "route")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if not self._guard(post=True):
            return
        try:
            size = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(size) or b"{}")
        except (ValueError, OSError):
            return self._fail(HTTPStatus.BAD_REQUEST, "body")
        if parsed.path == "/api/quit":
            self._json({"ok": True})
            self.app.stopped.set()
            return
        if parsed.path != "/api/op":
            return self._fail(HTTPStatus.NOT_FOUND, "route")
        name = payload.get("clip")
        if name not in self.app.clips or payload.get("op") != OP_ADD:
            return self._fail(HTTPStatus.BAD_REQUEST, "op")
        try:
            return self._json(self.app.add_box(name, payload))
        except (KeyError, TypeError, ValueError):
            return self._fail(HTTPStatus.BAD_REQUEST, "op")

    def _index(self, query: Dict) -> None:
        """`?t=` はブートストラップだけ。受け取ったら Cookie に移して URL から捨てる。"""
        app = self.app
        if not self._local_host(self.headers.get("Host")):
            return self._fail(HTTPStatus.FORBIDDEN, "host")
        given = (query.get("t") or [""])[0]
        if given and hmac.compare_digest(given, app.token):
            return self._send(HTTPStatus.SEE_OTHER, b"", "text/plain", extra=(
                ("Location", "/"),
                ("Set-Cookie",
                 f"{COOKIE_NAME}={app.token}; Path=/; SameSite=Strict; HttpOnly"),
            ))
        jar = SimpleCookie(self.headers.get("Cookie") or "")
        cookie = jar[COOKIE_NAME].value if COOKIE_NAME in jar else ""
        if not hmac.compare_digest(cookie, app.token):
            return self._fail(HTTPStatus.UNAUTHORIZED, "token")
        self._send(HTTPStatus.OK, app.html, "text/html; charset=utf-8", extra=(
            ("Content-Security-Policy",
             "default-src 'none'; img-src 'self' blob: data:; "
             "style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
             "connect-src 'self'; form-action 'none'; base-uri 'none'"),
        ))


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def handle_error(self, request, client_address) -> None:
        # AbortController で中断してもサーバ側は止まらないので、ffmpeg を
        # 回しきってから BrokenPipeError になる。黒い窓にトレースバックを吐かない。
        self.app.log(f"接続が切れました: {client_address}")  # type: ignore[attr-defined]


def serve(session: ReviewSession, *, port: int = 0, open_browser: bool = True,
          log_path: Optional[Path] = None, html_path: Optional[Path] = None,
          on_url=None) -> int:
    """127.0.0.1 だけに開く。--host は作らない（Windows のファイアウォールも出ない）。"""
    app = _App(session, log_path or Path(".tamako_work/webui.log"), html_path)
    try:
        server = _Server(("127.0.0.1", port), _Handler)
    except OSError as exc:
        raise FixError(f"画面を開けませんでした（ポート {port}）: {exc}") from exc
    server.app = app  # type: ignore[attr-defined]
    url = f"http://127.0.0.1:{server.server_address[1]}/?t={app.token}"
    if on_url:
        on_url(url)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        app.stopped.wait()
    except KeyboardInterrupt:
        pass
    server.shutdown()
    server.server_close()
    return 0
