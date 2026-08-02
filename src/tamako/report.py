"""編集内容の書き出し。

自動で切った結果は、必ず人が検算できる形で残す。Filmora で手直しするときの
下敷きにもなるよう、タイムコード付きの一覧を CSV でも出す。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .ordering import Clip
from .render import RenderReport
from .segments import CutPlan, Interval


def timecode(seconds: float) -> str:
    """HH:MM:SS.mmm 形式。Filmora のタイムラインに手で入れられる形にする。"""
    if seconds < 0:
        seconds = 0.0
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"


def _intervals(intervals: Sequence[Interval]) -> List[Dict[str, object]]:
    return [
        {
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "start_tc": timecode(start),
            "end_tc": timecode(end),
        }
        for start, end in intervals
    ]


def write_json_report(
    path: str | Path,
    *,
    clip_plans: Sequence[Tuple[Clip, CutPlan]],
    render_report: RenderReport,
    settings: Dict[str, object],
    sites: Sequence = (),
) -> Path:
    """機械可読の全記録。あとから条件を変えて再現するための情報も含める。"""
    from .sites import sites_to_json

    payload = {
        "version": 2,
        "settings": settings,
        # 人が見るべき箇所。危険度順（時刻順ではない）。
        "review_sites": sites_to_json(sites),
        "output": {
            "path": str(render_report.output),
            "duration": round(render_report.duration, 3),
            "duration_tc": timecode(render_report.duration),
            "width": render_report.geometry.width,
            "height": render_report.geometry.height,
            "fps": render_report.geometry.fps,
        },
        "mask_check": {
            "frames_total": render_report.frames_total,
            "frames_with_mask": render_report.frames_with_mask,
            "frames_without_mask": render_report.frames_total - render_report.frames_with_mask,
            "coverage": round(render_report.mask_coverage, 4),
            "frames_drawn_from_estimate": render_report.frames_estimated,
            "effective_mask_scale": round(render_report.effective_mask_scale, 3),
            # フレームの時刻の羅列ではなく区間で持つ。人が見る単位は区間。
            "no_face_intervals": _intervals(render_report.no_face_intervals),
            "longest_no_face_sec": round(render_report.longest_no_face_sec, 3),
            "uncertain_intervals": _intervals(render_report.uncertain_intervals),
        },
        "clips": [
            {
                "order": clip.order + 1,
                "file": clip.path.name,
                "path": str(clip.path),
                "duration": round(clip.info.duration, 3),
                "width": clip.info.width,
                "height": clip.info.height,
                "fps": clip.info.fps,
                "has_audio": clip.info.has_audio,
                "shot_at": clip.sort_time.isoformat(),
                "shot_at_source": clip.sort_basis,
                "kept": _intervals(plan.keep),
                "cut": _intervals(plan.cut),
                "detected_silence": _intervals(plan.silent),
                "detected_no_face": _intervals(plan.faceless),
                "kept_seconds": round(plan.kept_seconds, 3),
                "cut_seconds": round(plan.cut_seconds, 3),
            }
            for clip, plan in clip_plans
        ],
        "timeline": [
            {
                "index": i + 1,
                "out_start": round(seg.out_start, 3),
                "out_end": round(seg.out_start + seg.duration, 3),
                "out_start_tc": timecode(seg.out_start),
                "out_end_tc": timecode(seg.out_start + seg.duration),
                "source_file": seg.clip.path.name,
                "source_start": round(seg.start, 3),
                "source_end": round(seg.end, 3),
                "source_start_tc": timecode(seg.start),
                "source_end_tc": timecode(seg.end),
            }
            for i, seg in enumerate(render_report.segments)
        ],
    }

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def write_cut_list_csv(path: str | Path, render_report: RenderReport) -> Path:
    """Filmora で並べ直すときに見る表。Excel で開けるよう BOM 付きにする。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "順番", "出力開始", "出力終了", "長さ(秒)",
            "元ファイル", "元の開始", "元の終了",
        ])
        for i, seg in enumerate(render_report.segments, start=1):
            writer.writerow([
                i,
                timecode(seg.out_start),
                timecode(seg.out_start + seg.duration),
                f"{seg.duration:.3f}",
                seg.clip.path.name,
                timecode(seg.start),
                timecode(seg.end),
            ])
    return target


def summarize(
    clip_plans: Sequence[Tuple[Clip, CutPlan]], render_report: RenderReport
) -> str:
    """端末に出す要約。数字は必ず見せて、鵜呑みにしないで済むようにする。"""
    original = sum(clip.info.duration for clip, _ in clip_plans)
    kept = render_report.duration
    ratio = (kept / original * 100.0) if original > 0 else 0.0

    # 「99.8% を覆った」という割合は必ず「安全」と読み替えられてしまう。
    # 割合ではなく、顔なし区間の件数と最長の長さ＝人が見に行くべき箇所を出す。
    lines = [
        "",
        "── 結果 ──────────────────────────────",
        f"  素材 {len(clip_plans)} 本 / 合計 {timecode(original)}",
        f"  出力 {timecode(kept)} ({ratio:.1f}% を残した / {len(render_report.segments)} 区間)",
        f"  解像度 {render_report.geometry.width}x{render_report.geometry.height} "
        f"{render_report.geometry.fps:g}fps",
        "",
        "── 顔隠しの確認 ──────────────────────",
        f"  何も隠していない区間: {len(render_report.no_face_intervals)} 箇所 "
        f"(最長 {render_report.longest_no_face_sec:.1f} 秒)",
        f"  推定で覆ったフレーム（補間・外挿・膨張）: {render_report.frames_estimated}",
        f"  位置を保証できない区間: {len(render_report.uncertain_intervals)} 箇所",
        f"  マスクの実効倍率: {render_report.effective_mask_scale:.2f} "
        "(mask.png の透明部分を考慮した補正後)",
    ]

    if render_report.no_face_intervals:
        preview = ", ".join(
            f"{timecode(s)}-{timecode(e)}" for s, e in render_report.no_face_intervals[:6]
        )
        lines.append(f"    最初の数箇所: {preview}")
        lines.append(
            "  ※ 上記の区間は何も隠れていません。顔が写っていないなら問題ありませんが、"
        )
        lines.append(
            "     写っているのに検出できていない場合は顔が出たままです。必ず目視してください。"
        )
    return "\n".join(line for line in lines if line != "")
