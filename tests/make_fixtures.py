"""検証用の合成素材を作る。

実写の顔が写り、途中で顔が画面外に出て、音が有る区間と無音区間がある動画を
2 本作る。これで「撮影順の並べ替え」「無音カット」「顔なしカット」「顔マスク」
のすべてを実素材なしで検証できる。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tamako.ffmpeg import find_ffmpeg  # noqa: E402


def _face_patch(size: int = 160) -> np.ndarray:
    """実写の顔画像。検出器が本当に反応する素材でないと検証にならない。"""
    from skimage import data

    astronaut = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    # astronaut 画像の顔はおおよそ (170,55)-(275,180) の範囲にある。
    face = astronaut[40:200, 160:290]
    return cv2.resize(face, (size, size))


def make_clip(
    out_path: Path,
    *,
    seconds: float = 6.0,
    fps: int = 25,
    width: int = 640,
    height: int = 360,
    face_visible: tuple[float, float] = (0.0, 4.0),
    tone_spans: tuple[tuple[float, float], ...] = ((0.0, 2.0), (3.0, 6.0)),
    seed: int = 0,
) -> Path:
    """顔の出入りと無音区間を持つクリップを 1 本書き出す。"""
    rng = np.random.default_rng(seed)
    face = _face_patch()
    fh, fw = face.shape[:2]
    total_frames = int(seconds * fps)

    ffmpeg = find_ffmpeg()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 背景は毎フレーム微妙に変える。全フレーム同一だとエンコーダが潰してしまい
    # 顔検出の負荷検証にならないため。
    base = rng.integers(40, 90, size=(height, width, 3), dtype=np.uint8)

    silent_video = out_path.with_suffix(".video.mp4")
    proc = subprocess.Popen(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
            "-pix_fmt", "yuv420p", str(silent_video),
        ],
        stdin=subprocess.PIPE,
    )
    assert proc.stdin is not None
    try:
        for index in range(total_frames):
            t = index / fps
            frame = base.copy()
            frame = cv2.add(frame, rng.integers(0, 12, size=frame.shape, dtype=np.uint8))
            if face_visible[0] <= t < face_visible[1]:
                # 顔を左右に動かして、追従が効いているかを見えるようにする。
                progress = (t - face_visible[0]) / max(face_visible[1] - face_visible[0], 1e-6)
                x = int(40 + progress * (width - fw - 80))
                y = int(height / 2 - fh / 2 + 20 * np.sin(progress * 6.28))
                frame[y:y + fh, x:x + fw] = face
            proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
        proc.wait()

    # 音声: tone_spans の区間だけ 440Hz、それ以外は無音。
    #
    # adelay で頭に無音を足す方法は使えない。MP4 の edit list により先頭の
    # 遅延が再生時に詰められてしまい、「冒頭が無音」という肝心の場合を
    # 作れないため。無音と音を実体として作って concat で繋ぐ。
    parts: list[tuple[str, float]] = []
    cursor = 0.0
    for start, end in sorted(tone_spans):
        if start > cursor:
            parts.append(("silence", start - cursor))
        parts.append(("tone", end - start))
        cursor = end
    if cursor < seconds:
        parts.append(("silence", seconds - cursor))

    filters = []
    for i, (kind, length) in enumerate(parts):
        source = (
            f"anullsrc=r=44100:cl=mono,atrim=duration={length}"
            if kind == "silence"
            else f"sine=frequency=440:duration={length},aformat=channel_layouts=mono"
        )
        filters.append(f"{source},asetpts=N/SR/TB[s{i}]")
    concat_inputs = "".join(f"[s{i}]" for i in range(len(parts)))
    filter_complex = ";".join(filters) + f";{concat_inputs}concat=n={len(parts)}:v=0:a=1[a]"

    subprocess.run(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(silent_video),
            "-filter_complex", filter_complex,
            "-map", "0:v", "-map", "[a]",
            "-t", str(seconds),
            "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
            str(out_path),
        ],
        check=True,
    )
    silent_video.unlink(missing_ok=True)
    return out_path


def make_mask_png(path: Path, size: int = 256) -> Path:
    """顔に重ねる PNG。透過を含み、正方形でない場合の扱いも試せるようにする。"""
    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    cv2.circle(canvas, (size // 2, size // 2), size // 2 - 4, (40, 190, 250, 255), -1)
    cv2.circle(canvas, (size // 2 - 45, size // 2 - 30), 22, (30, 30, 30, 255), -1)
    cv2.circle(canvas, (size // 2 + 45, size // 2 - 30), 22, (30, 30, 30, 255), -1)
    cv2.ellipse(canvas, (size // 2, size // 2 + 40), (60, 35), 0, 0, 180, (30, 30, 30, 255), 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)
    return path


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("fixtures")
    raw = out_dir / "input"

    # わざと「ファイル名の順」と「撮影時刻の順」を食い違わせる。
    first = make_clip(
        raw / "b_clip.mp4",
        seconds=6.0,
        face_visible=(0.0, 4.0),
        tone_spans=((0.0, 2.0), (3.0, 6.0)),
        seed=1,
    )
    second = make_clip(
        raw / "a_clip.mp4",
        seconds=5.0,
        face_visible=(1.0, 5.0),
        tone_spans=((1.5, 5.0),),
        seed=2,
    )

    ffmpeg = find_ffmpeg()
    for path, stamp in ((first, "2024-05-01T10:00:00"), (second, "2024-05-01T11:30:00")):
        tagged = path.with_name(path.stem + ".tagged.mp4")
        subprocess.run(
            [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(path),
                "-c", "copy", "-metadata", f"creation_time={stamp}", str(tagged),
            ],
            check=True,
        )
        tagged.replace(path)

    make_mask_png(out_dir / "mask.png")
    print(f"fixtures written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
