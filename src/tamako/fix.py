"""人が仕上げる工程の窓と、その土台。

設計の要点は 2 つ。

1. **修正の確認に動画ファイルを焼かない。** 箱を動かしている最中に、その場で
   合成した結果を見せる。フレームはすでにメモリにあり、合成は数ミリ秒で終わる。
   1 箇所直すたびに動画 1 本を再符号化するのは、人 1 人の修正に対して
   18000 フレームの機械作業をさせることになる。
2. **cv2.selectROI は使わない。** 内部でモーダルなイベントループを回すため、
   コマ送りと同居できず、既存の箱を掴んで動かすこともできない（最頻の操作が
   「ずらす」なのに最悪）。setMouseCallback で自前に書くほうが短く済む。

窓が開けない環境（GUI 無しの OpenCV、SSH 越しなど）でも、選別と一覧は動く。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .frames import read_frames
from .manual import (
    OP_ADD, OP_ADJUST, OP_CONFIRM, OP_CUT, OP_DELETE, OP_HOLD,
    ManualEdits, Operation, apply_manual, coverage_signature,
)
from .ordering import Clip
from .overlay import composite, effective_scale, load_mask
from .sites import Site
from .tracks import TrackedClip


class FixError(RuntimeError):
    """窓を開けない、素材を読めない、など。"""


@dataclass
class _Rect:
    x: float
    y: float
    w: float
    h: float


class FrameCache:
    """素材の 1 フレームを取り出す。近傍をまとめて読んで持っておく。

    人はサイトの前後を行き来するので、1 フレームずつ ffmpeg を起動すると
    待ち時間がそのまま作業時間になる。
    """

    def __init__(self, clips: Sequence[Clip], *, window_sec: float = 2.0) -> None:
        self._clips = {c.path: c for c in clips}
        self._window = window_sec
        self._cache: Dict[Tuple[Path, int], np.ndarray] = {}
        self._loaded: List[Tuple[Path, float, float]] = []

    def get(self, clip_path: Path, pts: float) -> Optional[np.ndarray]:
        clip = self._clips.get(clip_path)
        if clip is None:
            return None
        index = int(round(pts * clip.info.fps))
        key = (clip_path, index)
        if key not in self._cache:
            self._load_around(clip, pts)
        return self._cache.get(key)

    def _load_around(self, clip: Clip, pts: float) -> None:
        start = max(0.0, pts - self._window / 2)
        duration = min(self._window, max(0.1, clip.info.duration - start))
        count = max(1, int(round(duration * clip.info.fps)))
        _, frames = read_frames(
            clip.path,
            out_width=clip.info.width, out_height=clip.info.height,
            fps=clip.info.fps, filters=[f"fps={clip.info.fps}"],
            start=start, duration=duration, expected_frames=count,
        )
        base = int(round(start * clip.info.fps))
        for offset, frame in enumerate(frames):
            self._cache[(clip.path, base + offset)] = frame.copy()
        # 古い窓は捨てる。長尺で全部持つとメモリが尽きる。
        self._loaded.append((clip.path, start, start + duration))
        if len(self._loaded) > 4:
            old_path, old_start, old_end = self._loaded.pop(0)
            lo = int(round(old_start * self._clips[old_path].info.fps))
            hi = int(round(old_end * self._clips[old_path].info.fps))
            for i in range(lo, hi + 1):
                self._cache.pop((old_path, i), None)


@dataclass
class ReviewSession:
    """確認と修正の状態。窓が無くても動く（試験もここを叩く）。"""

    clips: List[Clip]
    tracked: Dict[Path, TrackedClip]
    sites: List[Site]
    edits: ManualEdits
    mask_path: Path
    mask_scale: float = 2.0
    mask_offset_y: float = 0.0
    index: int = 0
    frame_offset: int = 0
    _cache: Optional[FrameCache] = None
    _mask: Optional[np.ndarray] = None
    _scale_eff: float = 0.0
    _merged: Dict[Path, TrackedClip] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._cache = FrameCache(self.clips)
        self._mask = load_mask(self.mask_path)
        self._scale_eff = effective_scale(self.mask_scale, self._mask)
        self.refresh()

    # ---------------------------------------------------------- 状態

    def refresh(self) -> None:
        """人手修正を反映し直す。1 操作ごとに呼ぶ（軽い）。"""
        self._merged = {}
        for path, track in self.tracked.items():
            ops = self.edits.effective(clip=path.name)
            self._merged[path] = apply_manual(track, ops)

    @property
    def site(self) -> Optional[Site]:
        if not self.sites:
            return None
        return self.sites[min(self.index, len(self.sites) - 1)]

    @property
    def current_pts(self) -> float:
        site = self.site
        if site is None:
            return 0.0
        track = self._merged.get(site.clip)
        fps = track.fps if track else 30.0
        return max(0.0, site.start + self.frame_offset / fps)

    def boxes_here(self) -> List:
        site = self.site
        if site is None:
            return []
        track = self._merged.get(site.clip)
        return track.at_pts(self.current_pts) if track else []

    def frame_here(self) -> Optional[np.ndarray]:
        site = self.site
        if site is None or self._cache is None:
            return None
        return self._cache.get(site.clip, self.current_pts)

    def composed(self) -> Optional[np.ndarray]:
        """今のフレームに、今の箱でマスクを合成した絵。窓に出すのはこれ。"""
        frame = self.frame_here()
        if frame is None:
            return None
        canvas = frame.copy()
        for box in self.boxes_here():
            cx, cy = box.center
            cy += box.h * self.mask_offset_y
            w = box.w * self._scale_eff
            h = box.h * self._scale_eff
            composite(canvas, self._mask, _Rect(cx - w / 2, cy - h / 2, w, h))
        return canvas

    # ---------------------------------------------------------- 移動

    def go(self, delta: int) -> None:
        if not self.sites:
            return
        self.index = max(0, min(len(self.sites) - 1, self.index + delta))
        self.frame_offset = 0

    def step(self, delta: int) -> None:
        site = self.site
        if site is None:
            return
        track = self._merged.get(site.clip)
        fps = track.fps if track else 30.0
        span = max(1, int(round((site.end - site.start) * fps)))
        # サイトの前後 1 秒までは行き来できるようにする。
        margin = int(round(fps))
        self.frame_offset = max(-margin, min(span + margin, self.frame_offset + delta))

    # ---------------------------------------------------------- 操作

    def _op(self, op: str, *, start: float, end: float, **data) -> Operation:
        site = self.site
        assert site is not None
        operation = self.edits.append(Operation(
            op=op, clip=site.clip.name, start=start, end=end, data=data,
        ))
        self.refresh()
        return operation

    def add_box(self, rect: Tuple[float, float, float, float],
                *, whole_site: bool = True) -> Optional[Operation]:
        """箱を足す。キーフレームは 1 点でよい（2 点要求は工数が 2 倍）。"""
        site = self.site
        if site is None:
            return None
        start, end = (site.start, site.end) if whole_site else (
            self.current_pts, self.current_pts + 1e-3)
        return self._op(OP_ADD, start=start, end=end,
                        keyframes=[[self.current_pts, *rect]])

    def delete_at(self, point: Tuple[float, float]) -> Optional[Operation]:
        site = self.site
        if site is None:
            return None
        return self._op(OP_DELETE, start=site.start, end=site.end, anchor=list(point))

    def adjust_at(self, point: Tuple[float, float], *, scale: float = 1.0,
                  dx: float = 0.0, dy: float = 0.0) -> Optional[Operation]:
        site = self.site
        if site is None:
            return None
        return self._op(OP_ADJUST, start=site.start, end=site.end,
                        anchor=list(point), scale=scale, dx=dx, dy=dy)

    def hold_from_here(self, seconds: float = 1.0) -> Optional[Operation]:
        site = self.site
        if site is None:
            return None
        return self._op(OP_HOLD, start=self.current_pts,
                        end=self.current_pts + seconds)

    def send_to_cut(self) -> Optional[Operation]:
        site = self.site
        if site is None:
            return None
        return self._op(OP_CUT, start=site.start, end=site.end)

    def confirm(self) -> Optional[Operation]:
        """見て問題なかった、を記録する。被覆の署名を一緒に残す。"""
        site = self.site
        if site is None:
            return None
        track = self._merged.get(site.clip)
        if track is None:
            return None
        signature = coverage_signature(
            track, site.start, site.end, self._mask[:, :, 3],
            mask_scale=self._scale_eff, offset_y=self.mask_offset_y,
        )
        operation = self._op(OP_CONFIRM, start=site.start, end=site.end,
                             signature=signature, kind=site.kind)
        self.go(+1)
        return operation

    def undo(self) -> Optional[Operation]:
        operation = self.edits.undo_last()
        if operation is not None:
            self.refresh()
        return operation


# ------------------------------------------------------------------ 窓

_HELP = [
    "左ドラッグ: 箱を足す   右クリック: その箱を消す",
    "+/-: 大きく/小さく   h j k l: 左下上右にずらす",
    ", .: コマ送り   n p: 次/前のサイト",
    "c: 確認済み   x: この区間を落とす   u: 元に戻す   q: 終了",
]


def run_window(session: ReviewSession, *, title: str = "tamako fix") -> int:
    """箱を引く窓。修正のたびに、その場で合成し直して見せる。"""
    import cv2

    state = {"drag": None, "cursor": (0, 0), "quit": False}

    def on_mouse(event, x, y, flags, _param):
        state["cursor"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drag"] = [x, y, x, y]
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"]:
            state["drag"][2:] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP and state["drag"]:
            x0, y0, x1, y1 = state["drag"]
            state["drag"] = None
            w, h = abs(x1 - x0), abs(y1 - y0)
            if w > 8 and h > 8:
                session.add_box((min(x0, x1), min(y0, y1), w, h))
        elif event == cv2.EVENT_RBUTTONDOWN:
            session.delete_at((x, y))

    try:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(title, on_mouse)
    except cv2.error as exc:
        raise FixError(
            "窓を開けませんでした。GUI の無い OpenCV が入っている可能性があります。\n"
            "  pip install opencv-python  （opencv-python-headless を置き換える）\n"
            f"（元の例外: {exc}）"
        ) from exc

    while not state["quit"]:
        canvas = session.composed()
        if canvas is None:
            break
        canvas = canvas.copy()
        _draw_hud(cv2, canvas, session, state)
        cv2.imshow(title, canvas)

        key = cv2.waitKey(30) & 0xFF
        if key == 255:
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            continue
        cursor = state["cursor"]
        if key == ord("q"):
            break
        elif key == ord("n"):
            session.go(+1)
        elif key == ord("p"):
            session.go(-1)
        elif key == ord(","):
            session.step(-1)
        elif key == ord("."):
            session.step(+1)
        elif key in (ord("+"), ord("=")):
            session.adjust_at(cursor, scale=1.15)
        elif key in (ord("-"), ord("_")):
            session.adjust_at(cursor, scale=1.0 / 1.15)
        elif key == ord("h"):
            session.adjust_at(cursor, dx=-6)
        elif key == ord("l"):
            session.adjust_at(cursor, dx=+6)
        elif key == ord("k"):
            session.adjust_at(cursor, dy=-6)
        elif key == ord("j"):
            session.adjust_at(cursor, dy=+6)
        elif key == ord("c"):
            session.confirm()
        elif key == ord("x"):
            session.send_to_cut()
        elif key == ord("u"):
            session.undo()

    cv2.destroyAllWindows()
    return 0


def _draw_hud(cv2, canvas, session: ReviewSession, state) -> None:
    """今どこを見ているか、何ができるかを画面内に出す。"""
    site = session.site
    height = canvas.shape[0]
    if site is not None:
        header = (
            f"[{session.index + 1}/{len(session.sites)}] {site.clip.name} "
            f"{site.start:.2f}-{site.end:.2f}s  risk={site.risk:.2f}  {site.kind}"
        )
        cv2.putText(canvas, header, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, header, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for i, line in enumerate(_HELP):
        y = height - 8 - (len(_HELP) - 1 - i) * 16
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (200, 255, 200), 1, cv2.LINE_AA)
    if state["drag"]:
        x0, y0, x1, y1 = state["drag"]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 255, 255), 2)
