"""解析の段取り。素材を読み、無音と顔を調べ、残す区間を決めるところまで。

書き出しは含めない。`check` で下見だけしたい場合と `edit` で本番を回す場合の
両方から同じ手順を呼べるようにしてある。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from .tracks import TrackConfig

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


def track_config(config: Config) -> "TrackConfig":
    """設定から後処理の設定を組む。"""
    from .tracks import TrackConfig

    mask_cfg = config.section("mask")
    fps_guess = 30.0
    return TrackConfig(
        score_threshold=float(mask_cfg["score_threshold"]),
        dilate_frames=int(mask_cfg["dilate_frames"]),
        extrap_frames=max(1, int(round(float(mask_cfg["hold_sec"]) * fps_guess))),
        expand_per_velocity=float(mask_cfg["expand_per_velocity"]),
        expand_limit=float(mask_cfg["expand_limit"]),
    )


def track_clips(
    clips: Sequence[Clip],
    config: Config,
    detections: dict,
    *,
    on_progress: Optional[ProgressFn] = None,
) -> dict:
    """検出結果からトラックを組み、途切れを埋める。素材パス → TrackedClip。"""
    from .tracks import track_clip

    base = track_config(config)
    result = {}
    for clip in clips:
        if on_progress:
            on_progress(f"追従を計算中 [{clip.order + 1}/{len(clips)}] {clip.name}")
        det = load_detections(detections[clip.path])
        # hold_sec は秒で指定されるので、素材の実 fps でフレーム数に直す。
        cfg = replace(
            base,
            extrap_frames=max(1, int(round(
                float(config.section("mask")["hold_sec"]) * det.fps
            ))),
        )
        result[clip.path] = track_clip(det, cfg)
    return result


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


def negative_regions_for(config: Config) -> list:
    """恒常的な誤検出領域（ポスター・鏡・人形）。

    UI 側の refresh も同じものを見なければならない。ここで括り出しておかないと、
    UI 上だけ誤検出が「復活して見え」、利用者が無駄な delete を積む。
    """
    from .manual import load_negative_regions

    return load_negative_regions(config.output_dir.parent / "negative_regions.json")


def merge_manual(clips: Sequence[Clip], tracked: dict, config: Config) -> Tuple[dict, object]:
    """自動の結果に人手修正を重ねる。人手が常に勝つ。

    negative_regions（ポスター・鏡など恒常的な誤検出）もここで落とす。
    毎回同じ場所で「消す」を繰り返させないため。

    **返すのは新しい辞書で、引数の tracked は書き換えない。** 呼び出し側が
    「まだ人手を当てていない生の結果」を持ち続けられることが、二重適用を
    構造的に防ぐ唯一の手段になる。
    """
    from .manual import ManualEdits, apply_manual

    work = detection_work_dir(config)
    edits = ManualEdits(work / "faces_manual.jsonl")
    regions = negative_regions_for(config)
    merged = {}
    for clip in clips:
        track = tracked.get(clip.path)
        if track is None:
            continue
        merged[clip.path] = apply_manual(
            track, edits.effective(clip=clip.path.name), negative_regions=regions
        )
    return merged, edits


def confirmed_spans(clip: Clip, tracked_clip, edits, config: Config) -> List[Tuple[float, float]]:
    """まだ有効な「確認済み」区間。

    「下の検出結果が変わったら再確認」を箱の一致で判定すると、閾値を 0.01
    動かしただけで全部の確認が飛ぶ。しかも人が設定を変える動機の大半は
    安全側（scale を上げる・閾値を下げる）なので、最も安全な操作が最も確認を
    壊すことになる。**被覆が減っていなければ維持する**（上位集合なら、
    新たな漏れは原理的に発生しない）。
    """
    from .manual import OP_CONFIRM, confirmation_still_valid, coverage_signature
    from .overlay import effective_scale, load_mask

    ops = [op for op in edits.effective(clip=clip.path.name) if op.op == OP_CONFIRM]
    if not ops:
        return []

    mask_cfg = config.section("mask")
    mask = load_mask(config.mask_image)
    scale = effective_scale(float(mask_cfg["scale"]), mask)
    offset_y = float(mask_cfg.get("offset_y", 0.0))

    spans: List[Tuple[float, float]] = []
    for op in ops:
        old = op.data.get("signature", "")
        now = coverage_signature(
            tracked_clip, op.start, op.end, mask[:, :, 3],
            mask_scale=scale, offset_y=offset_y,
        )
        if confirmation_still_valid(old, now):
            spans.append((op.start, op.end))
    return spans


def collect_sites(
    clip_plans: Sequence[Tuple[Clip, CutPlan]],
    tracked: dict,
    *,
    min_site_sec: float = 0.05,
    edits: object = None,
    config: Optional[Config] = None,
) -> list:
    """全素材のサイトを危険度順に集める。確認済みの区間は出さない。"""
    from .sites import build_sites

    sites = []
    for clip, plan in clip_plans:
        track = tracked.get(clip.path)
        if track is None:
            continue
        skip: List[Tuple[float, float]] = []
        if edits is not None and config is not None:
            skip = confirmed_spans(clip, track, edits, config)
        sites.extend(build_sites(
            clip.path, track, plan.keep, plan.silent,
            duration=clip.info.duration, min_site_sec=min_site_sec, skip=skip,
        ))
    sites.sort(key=lambda s: s.risk, reverse=True)
    return sites


def apply_uncovered_policy(
    clip_plans: Sequence[Tuple[Clip, CutPlan]],
    tracked: dict,
    config: Config,
    edits: object = None,
) -> Tuple[List[Tuple[Clip, CutPlan]], list, float]:
    """C-2。覆えない区間の扱いを適用し、(計画, サイト, 削った秒数) を返す。

    cut は「フレームを間引く」ではなく「残す区間を削る」で実装する。
    フレームを個別に落とすと映像の枚数と音声の秒数が食い違い、
    時間軸の契約が壊れて全体が音ズレする。
    """
    from .manual import manual_cut_intervals
    from .segments import intersect, invert, total
    from .sites import apply_cut_policy

    policy = str(config.section("mask").get("uncovered_policy", "expand"))
    sites = collect_sites(clip_plans, tracked, edits=edits, config=config)

    # 人が「ここは落とす」と決めた区間は、policy によらず必ず落とす。
    manual_cuts: dict = {}
    if edits is not None:
        for clip, _ in clip_plans:
            spans = manual_cut_intervals(edits.effective(clip=clip.path.name))
            if spans:
                manual_cuts[clip.path] = spans

    if policy != "cut" and not manual_cuts:
        return list(clip_plans), sites, 0.0

    min_keep = float(config.section("cut")["min_keep_sec"])
    adjusted: List[Tuple[Clip, CutPlan]] = []
    removed = 0.0
    for clip, plan in clip_plans:
        keep = list(plan.keep)
        if policy == "cut":
            mine = [s for s in sites if s.clip == clip.path]
            keep = apply_cut_policy(keep, mine, min_keep_sec=min_keep)
        for span in manual_cuts.get(clip.path, []):
            keep = intersect(keep, invert([span], clip.info.duration))
        keep = [(s, e) for s, e in keep if e - s >= min_keep]
        removed += total(plan.keep) - total(keep)
        adjusted.append((clip, replace(
            plan, keep=keep, cut=invert(keep, clip.info.duration)
        )))
    # 削った後の状態でサイトを取り直す（削れた箇所はもう出力に出ない）。
    return adjusted, collect_sites(adjusted, tracked, edits=edits, config=config), removed


def subset_plans(
    clip_plans: Sequence[Tuple[Clip, CutPlan]],
    spans: dict,
) -> List[Tuple[Clip, CutPlan]]:
    """残す区間を、指定した素材時刻の範囲に絞った計画を作る。

    確認用の抜き出し（remask）と部分書き出し（edit --only）の共通の土台。
    素材から読み直すので、何度出しても画質は劣化しない。
    """
    from .segments import intersect, invert, merge

    result: List[Tuple[Clip, CutPlan]] = []
    for clip, plan in clip_plans:
        wanted = merge(spans.get(clip.path, []))
        if not wanted:
            continue
        keep = intersect(plan.keep, wanted)
        if not keep:
            continue
        result.append((clip, replace(
            plan, keep=keep, cut=invert(keep, clip.info.duration)
        )))
    return result


def site_spans(sites: Sequence, *, margin: float = 1.5) -> dict:
    """サイトの前後に余白を付けた素材時刻の範囲。

    **粗い画質で全編を焼くのではなく、フル解像度で危ない箇所だけを焼く。**
    実務で最も多い漏れは部分被覆（顎が数十 px 出ている等）で、720p に落とすと
    そこが原理的に判別できない。削ってよい次元は空間ではなく時間。
    """
    spans: dict = {}
    for site in sites:
        spans.setdefault(site.clip, []).append(
            (max(0.0, site.start - margin), site.end + margin)
        )
    return spans


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
