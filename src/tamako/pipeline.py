"""解析の段取り。素材を読み、無音と顔を調べ、残す区間を決めるところまで。

書き出しは含めない。`check` で下見だけしたい場合と `edit` で本番を回す場合の
両方から同じ手順を呼べるようにしてある。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from .config import Config
from .detect import detect_clips, load_detections
from .ordering import Clip, find_videos, order_clips
from .segments import CutPlan, build_cut_plan, samples_to_intervals
from .silence import detect_silence

ProgressFn = Callable[[str], None]


class PipelineError(RuntimeError):
    """素材が見つからない、など解析を始められない状態。"""


def collect_clips(config: Config) -> Tuple[List[Clip], List[Tuple[Path, str]]]:
    """入力フォルダの素材を撮影順に並べる。"""
    input_dir = config.input_dir
    paths = find_videos(input_dir, config["video_extensions"])
    if not paths:
        raise PipelineError(
            f"入力フォルダに動画がありません: {input_dir}\n"
            f"  対象の拡張子: {', '.join(config['video_extensions'])}"
        )
    clips, failures = order_clips(paths)
    if not clips:
        raise PipelineError(
            f"読める動画がありませんでした（{len(failures)} 件が失敗）: {input_dir}"
        )
    return clips, failures


def detection_work_dir(config: Config) -> Path:
    """検出結果の置き場。output/ の外＝人に渡すものと混ざらない場所。"""
    return config.output_dir / ".tamako_work"


def run_detection(
    clips: Sequence[Clip],
    config: Config,
    *,
    face_model: Optional[str | Path] = None,
    on_progress: Optional[ProgressFn] = None,
) -> dict:
    """全素材の顔検出（重い工程）。済んでいる素材はメタ行の照合で飛ばす。

    保存する閾値はカット用とマスク用の低いほうに合わせる。消費側（カット判定・
    マスク描画）がそれぞれの閾値でフィルタして使うので、片方の設定を変えても
    再検出は不要になる。
    """
    cut = config.section("cut")
    mask_cfg = config.section("mask")
    store_threshold = min(
        float(cut["face_score_threshold"]), float(mask_cfg["score_threshold"])
    )
    return detect_clips(
        clips,
        work_dir=detection_work_dir(config),
        score_threshold=store_threshold,
        detect_width=int(mask_cfg["detect_width"]),
        face_model=face_model,
        on_progress=on_progress,
    )


def analyze_clips(
    clips: Sequence[Clip],
    config: Config,
    *,
    face_model: Optional[str | Path] = None,
    on_progress: Optional[ProgressFn] = None,
) -> List[Tuple[Clip, CutPlan]]:
    """素材ごとに無音と顔在を調べ、残す区間を決める。

    顔在は faces.jsonl（全フレーム検出）から導く。以前ここにあった 3fps の
    別走査は廃止した。検出は 1 回で、カット判定とマスク描画の両方が使う。
    """
    cut = config.section("cut")
    detections = run_detection(
        clips, config, face_model=face_model, on_progress=on_progress
    )

    results: List[Tuple[Clip, CutPlan]] = []
    for clip in clips:
        if on_progress:
            on_progress(f"解析中 [{clip.order + 1}/{len(clips)}] {clip.name}")

        silent = detect_silence(
            clip.path,
            duration=clip.info.duration,
            has_audio=clip.info.has_audio,
            noise_db=float(cut["silence_db"]),
            min_silence_sec=float(cut["silence_min_sec"]),
        )
        det = load_detections(detections[clip.path])
        threshold = float(cut["face_score_threshold"])
        times = sorted(
            rec.pts
            for rec in det.records.values()
            if any(box[4] >= threshold for box in rec.boxes)
        )
        face_present = samples_to_intervals(
            times, step=1.0 / det.fps, hold=float(cut["face_hold_sec"])
        )
        plan = build_cut_plan(
            duration=clip.info.duration,
            silent=silent,
            face_present=face_present,
            mode=str(cut["mode"]),
            min_cut_sec=float(cut["min_cut_sec"]),
            min_keep_sec=float(cut["min_keep_sec"]),
            padding_sec=float(cut["padding_sec"]),
        )
        results.append((clip, plan))
    return results


def describe_plans(clip_plans: Sequence[Tuple[Clip, CutPlan]]) -> str:
    """カット結果の下見。書き出す前に人が判断できるだけの情報を出す。"""
    from .report import timecode

    lines: List[str] = []
    for clip, plan in clip_plans:
        lines.append(
            f"  {clip.order + 1:3d}. {clip.name}  "
            f"{clip.info.duration:6.2f}s → {plan.kept_seconds:6.2f}s "
            f"({len(plan.keep)} 区間 / {plan.cut_seconds:.2f}s を削除)"
        )
        if not plan.keep:
            lines.append("        ※ 全部削除されます。条件が厳しすぎないか確認してください。")
            continue
        for start, end in plan.keep[:6]:
            lines.append(f"        残す: {timecode(start)} - {timecode(end)}")
        if len(plan.keep) > 6:
            lines.append(f"        … 他 {len(plan.keep) - 6} 区間")
    return "\n".join(lines)
