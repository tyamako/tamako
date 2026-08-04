"""コマンド行の入口。

前半（並べる・切る・顔を隠す）と後半（声を入れて字幕を付ける）を別の
コマンドに分けてある。あいだに「人が見て確認し、声を録る」という工程が
必ず入るため、一度に走らせる形にしても意味がない。
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from .config import Config, ConfigError, load_config


def _print(message: str = "") -> None:
    print(message, flush=True)


def _interactive() -> bool:
    """端末に直接出しているか。ファイルに落としているなら上書き表示はしない。"""
    return sys.stdout.isatty()


def _progress_line(message: str, fraction: float) -> None:
    if not _interactive():
        return
    width = 28
    filled = int(width * max(0.0, min(1.0, fraction)))
    bar = "#" * filled + "." * (width - filled)
    sys.stdout.write(f"\r  [{bar}] {fraction * 100:5.1f}%  {message}   ")
    sys.stdout.flush()
    if fraction >= 1.0:
        sys.stdout.write("\n")
        sys.stdout.flush()


def _status(message: str) -> None:
    """途中経過の 1 行。端末なら書き換え、そうでなければ普通に 1 行ずつ出す。"""
    if _interactive():
        sys.stdout.write("\r" + " " * 78 + f"\r  {message}")
        sys.stdout.flush()
    else:
        print(f"  {message}", flush=True)


def _clear_status() -> None:
    if _interactive():
        sys.stdout.write("\r" + " " * 78 + "\r")
        sys.stdout.flush()


CONFIG_TEMPLATE = """{
  // ── 入出力 ───────────────────────────────────────────
  // このファイルからの相対パスでも、C:/... のような絶対パスでも書けます。
  "input_dir": "input",      // 撮影した動画を入れるフォルダ
  "output_dir": "output",    // 書き出し先
  "mask_image": "mask.png",  // 顔に重ねる PNG（透過推奨）

  // ── どこを切るか ─────────────────────────────────────
  "cut": {
    // any  : 無音 または 顔が写っていない ところを切る（既定・最も短くなる）
    // both : 無音 かつ 顔が写っていない ところだけ切る（最も安全）
    // silence / face : どちらか一方だけを根拠にする
    "mode": "any",

    "silence_db": -32.0,        // これより静かなら無音とみなす（-40 で厳しく、-25 で緩く）
    "silence_min_sec": 0.8,     // この長さ以上続いた無音だけを対象にする
    "face_analysis_fps": 3.0,   // （互換のため残しています。今は使いません）
    "face_score_threshold": 0.6,// 顔と判定する確信度（カット判定用）
    "face_hold_sec": 1.0,       // この長さ以下の検出の途切れは無視する
    "padding_sec": 0.25,        // 残す区間の前後に足す余白（語頭語尾の切れ防止）
    "min_keep_sec": 0.6,        // これより短い残り区間は捨てる
    "min_cut_sec": 0.5,         // これより短いカットは行わない（細切れ防止）
    "detect_width": 640         // カット判定用の検出解像度（マスク用とは独立）
  },

  // ── 顔の隠し方 ───────────────────────────────────────
  "mask": {
    "score_threshold": 0.5,   // 隠すときは低めにする（見逃すより多めに隠す）
    "scale": 2.0,             // 検出枠の何倍を隠すか。1.6 だと髪が出ます
                              // ※ mask.png の透明部分に応じて自動で補正されます
    "offset_y": 0.0,          // 上下の微調整（顔の高さに対する割合、負で上）
    "hold_sec": 0.7,          // 検出が切れたあと、外挿で覆い続ける長さ
    "detect_width": 640,      // 検出用に縮小する幅（大きいほど小さい顔に強い・遅い）
    "dilate_frames": 2,       // 前後これだけのフレームの箱も取り込む（保険）
    "expand_per_velocity": 0.5, // 推定の不確かさを箱の大きさで吸収する強さ
    "expand_limit": 4.0,      // 広げる上限（倍）。超えたら「位置不明」として報告
    "uncovered_policy": "expand" // 覆えない箇所: expand / cut / warn
  },

  // ── 書き出し設定 ─────────────────────────────────────
  "encode": {
    "crf": 20,            // 小さいほど高画質・大容量（18〜24 が実用範囲）
    "preset": "medium",   // ultrafast〜veryslow。遅いほど圧縮が良い
    "audio_bitrate": "192k",
    "pix_fmt": "yuv420p"
  },

  // ── 字幕と音声（後半の工程）─────────────────────────
  "subtitle": {
    "model": "small",          // tiny / base / small / medium / large-v3
    "language": "ja",
    "font": "Yu Gothic UI",    // 端末に入っているフォント名
    "font_size": 42,
    "margin_v": 60,
    "outline": 3,
    "max_chars_per_line": 20,
    "audio_mode": "replace",   // replace: 元音声を置換 / mix: 元音声を下げて重ねる
    "original_volume": 0.15
  }
}
"""


def cmd_init(args: argparse.Namespace) -> int:
    """設定ファイルとフォルダの雛形を作る。"""
    target_dir = Path(args.directory).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    config_path = target_dir / "config.json"

    if config_path.exists() and not args.force:
        _print(f"すでにあります: {config_path}")
        _print("上書きする場合は --force を付けてください。")
        return 1

    config_path.write_text(CONFIG_TEMPLATE, encoding="utf-8")
    (target_dir / "input").mkdir(exist_ok=True)
    (target_dir / "output").mkdir(exist_ok=True)

    _print(f"用意しました: {target_dir}")
    _print("  config.json  設定ファイル（テキストエディタで編集できます）")
    _print("  input/       ここに撮影した動画を入れてください")
    _print("  output/      ここに書き出されます")
    _print("")
    _print("次に、顔に重ねる PNG を mask.png という名前でこのフォルダに置いてください。")
    _print(f"準備ができたら:  tamako check --config \"{config_path}\"")
    return 0


def _load(args: argparse.Namespace) -> Config:
    config = load_config(args.config)
    if getattr(args, "input", None):
        config.data["input_dir"] = str(Path(args.input).expanduser().resolve())
    if getattr(args, "output", None):
        config.data["output_dir"] = str(Path(args.output).expanduser().resolve())
    if getattr(args, "mask", None):
        config.data["mask_image"] = str(Path(args.mask).expanduser().resolve())
    if getattr(args, "mode", None):
        config.data["cut"]["mode"] = args.mode
    return config


def cmd_detect(args: argparse.Namespace) -> int:
    """顔検出だけを先に走らせる（重い工程。結果はキャッシュされる）。"""
    import time

    from .pipeline import collect_clips, detection_work_dir, run_detection

    config = _load(args)
    clips, failures = collect_clips(config)
    for path, reason in failures:
        _print(f"  読み飛ばし: {path.name} ({reason})")

    started = time.monotonic()
    run_detection(clips, config, face_model=args.face_model, on_progress=_status)
    _clear_status()
    took = time.monotonic() - started
    total = sum(c.info.duration for c in clips)
    _print(f"検出が完了しました: {len(clips)} 本 / 素材 {total:.1f}s / 所要 {took:.1f}s")
    _print(f"  結果の置き場: {detection_work_dir(config) / 'detect'}")
    _print("  ※ 素材や検出設定が変わらない限り、次回からここは飛ばされます。")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """書き出さずに、並び順とカット結果だけを見る。"""
    from .pipeline import analyze_clips, collect_clips, describe_plans
    from .ordering import describe_order

    config = _load(args)
    clips, failures = collect_clips(config)

    _print("── 撮影順 ────────────────────────────")
    _print(describe_order(clips))
    if failures:
        _print("")
        _print("読めなかったファイル:")
        for path, reason in failures:
            _print(f"  {path.name}: {reason}")

    _print("")
    _print("── カット結果（下見）────────────────")
    plans = analyze_clips(clips, config, face_model=args.face_model, on_progress=_status)
    _clear_status()
    _print(describe_plans(plans))

    original = sum(c.info.duration for c in clips)
    kept = sum(p.kept_seconds for _, p in plans)
    _print("")
    _print(f"  合計 {original:.2f}s → {kept:.2f}s "
           f"({kept / original * 100 if original else 0:.1f}% を残す)")
    _print("")
    _print("この内容でよければ `tamako edit` を実行してください。")
    _print("切りすぎ・切らなすぎの場合は config.json の cut を調整してください。")
    return 0


def cmd_edit(args: argparse.Namespace) -> int:
    """並べる・切る・顔を隠す（前半の工程）。"""
    from .ordering import describe_order
    from .pipeline import (
        analyze_clips, apply_uncovered_policy, collect_clips, describe_plans,
        detection_work_dir, merge_manual, run_detection, subset_plans, track_clips,
    )
    from .render import render
    from .report import summarize, write_cut_list_csv, write_json_report
    from .sites import describe_sites, summarize_sites

    config = _load(args)
    mask_path = config.mask_image
    if not mask_path.is_file():
        _print(f"顔に重ねる画像が見つかりません: {mask_path}")
        _print("透過 PNG を用意して、config.json の mask_image に指定してください。")
        return 1

    clips, failures = collect_clips(config)
    _print("── 撮影順 ────────────────────────────")
    _print(describe_order(clips))
    if failures:
        _print("")
        for path, reason in failures:
            _print(f"  読み飛ばし: {path.name} ({reason})")

    _print("")
    plans = analyze_clips(clips, config, face_model=args.face_model, on_progress=_status)
    _clear_status()
    _print(describe_plans(plans))
    _print("")

    mask_cfg = config.section("mask")
    encode_cfg = config.section("encode")
    output_dir = config.output_dir
    output_path = output_dir / args.name

    detections = run_detection(clips, config, face_model=args.face_model)
    tracked = track_clips(clips, config, detections, on_progress=_status)
    tracked, edits = merge_manual(clips, tracked, config)
    _clear_status()

    manual_count = len(edits.effective())
    if manual_count:
        _print(f"  人手修正を {manual_count} 件反映しました")

    plans, sites, removed = apply_uncovered_policy(plans, tracked, config, edits)
    if removed > 0:
        _print(f"  覆えない区間を {removed:.2f}s 削りました")
    if manual_count or removed > 0:
        _print("")

    work = detection_work_dir(config)
    if getattr(args, "only", None):
        # 出力時刻で指定された範囲を、素材時刻に直して絞り込む。
        # 「直したものが本番画質で本当に隠れているか」を全編待たずに見るため。
        from .render import choose_geometry
        from .timeline import build_timeline

        try:
            low, high = (_parse_timecode(p) for p in args.only.split("-", 1))
        except ValueError as exc:
            _print(f"--only の指定が読めません: {exc}")
            return 2
        geometry = choose_geometry([clip for clip, _ in plans])
        timeline = build_timeline(plans, geometry.fps)
        spans: dict = {}
        for index in range(int(low * geometry.fps), int(high * geometry.fps) + 1):
            if not (0 <= index < timeline.total_frames):
                continue
            segment, pts = timeline.to_source(index)
            spans.setdefault(segment.clip.path, []).append((pts, pts + 1.0 / geometry.fps))
        plans = subset_plans(plans, spans)
        if not plans:
            _print(f"指定の範囲に書き出すものがありません: {args.only}")
            return 1
        output_path = output_path.with_name(f"{output_path.stem}_part{output_path.suffix}")
        _print(f"  部分書き出し: {args.only} → {output_path.name}")
        _print("")

    report = render(
        plans,
        tracked,
        output_path=output_path,
        mask_path=mask_path,
        mask_scale=float(mask_cfg["scale"]),
        mask_offset_y=float(mask_cfg.get("offset_y", 0.0)),
        crf=int(encode_cfg["crf"]),
        preset=str(encode_cfg["preset"]),
        pix_fmt=str(encode_cfg["pix_fmt"]),
        audio_bitrate=str(encode_cfg["audio_bitrate"]),
        diagnostic=getattr(args, "diagnostic", False),
        draw_log=work / f"draw_{output_path.stem}.tsv",
        work_dir=work,
        on_progress=_progress_line,
    )

    json_path = write_json_report(
        output_dir / "edit_report.json",
        clip_plans=plans,
        render_report=report,
        settings={"cut": config.section("cut"), "mask": mask_cfg},
        sites=sites,
    )
    csv_path = write_cut_list_csv(output_dir / "cut_list.csv", report)

    _print(summarize(plans, report))
    _print("")
    _print(summarize_sites(sites, out_duration=report.duration))
    if sites:
        _print("")
        _print("── 危険度の高い順 ────────────────────")
        _print(describe_sites(sites))
    _print("")
    _print("── 書き出したもの ────────────────────")
    _print(f"  {report.output}")
    _print(f"  {csv_path}   （カット位置の一覧）")
    _print(f"  {json_path}  （全記録・サイト一覧）")
    _print("")
    if sites:
        _print("次の工程: 上の箇所を確認し、直すところがあれば")
        _print("  tamako fix --review")
    else:
        _print("次の工程: 動画を見ながら声を録音し、その音声ファイルを用意してから")
        _print(f"  tamako finish --video \"{report.output}\" --audio \"収録音声.wav\"")
    return 0


def _parse_timecode(text: str) -> float:
    """HH:MM:SS.mmm / MM:SS / 秒 のどれでも受ける。"""
    parts = text.strip().split(":")
    try:
        values = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"時刻として読めません: {text}") from exc
    seconds = 0.0
    for value in values:
        seconds = seconds * 60.0 + value
    return seconds


def _prepare(args: argparse.Namespace, config: Config):
    """検出 → 追従 → 人手修正 → サイト、までをまとめて行う。"""
    from .pipeline import (
        analyze_clips, apply_uncovered_policy, collect_clips, merge_manual,
        run_detection, track_clips,
    )

    clips, _ = collect_clips(config)
    plans = analyze_clips(clips, config, face_model=args.face_model, on_progress=_status)
    detections = run_detection(clips, config, face_model=args.face_model)
    tracked = track_clips(clips, config, detections, on_progress=_status)
    tracked, edits = merge_manual(clips, tracked, config)
    plans, sites, _removed = apply_uncovered_policy(plans, tracked, config, edits)
    _clear_status()
    return clips, plans, tracked, edits, sites


def cmd_remask(args: argparse.Namespace) -> int:
    """要確認の箇所だけを、フル解像度で短く書き出す（確認用）。"""
    from .pipeline import site_spans, subset_plans
    from .render import render
    from .report import timecode

    config = _load(args)
    mask_path = config.mask_image
    if not mask_path.is_file():
        _print(f"顔に重ねる画像が見つかりません: {mask_path}")
        return 1

    clips, plans, tracked, edits, sites = _prepare(args, config)
    if not sites:
        _print("要確認の箇所はありません。")
        return 0

    spans = site_spans(sites, margin=args.margin)
    subset = subset_plans(plans, spans)
    if not subset:
        _print("書き出す区間がありません。")
        return 1

    review_dir = config.output_dir / "_review"
    output_path = review_dir / "review.mp4"
    mask_cfg = config.section("mask")
    encode_cfg = config.section("encode")

    _print(f"要確認 {len(sites)} 箇所の前後 ±{args.margin:g} 秒を書き出します。")
    report = render(
        subset, tracked,
        output_path=output_path,
        mask_path=mask_path,
        mask_scale=float(mask_cfg["scale"]),
        mask_offset_y=float(mask_cfg.get("offset_y", 0.0)),
        # 解像度は落とさない。落とすと部分被覆（最頻の漏れ）が見えなくなる。
        # 速くするのは符号化の手間と尺のほうで落とす。
        crf=30, preset="ultrafast",
        pix_fmt=str(encode_cfg["pix_fmt"]),
        audio_bitrate=str(encode_cfg["audio_bitrate"]),
        diagnostic=args.diagnostic,
        work_dir=config.output_dir / ".tamako_work",
    )
    _print("")
    _print(f"  {report.output}  ({timecode(report.duration)})")
    _print("  ※ 素材と同じ解像度です。マスクの角から顔が出ていないか見てください。")
    _print("")
    _print("直すところがあれば:  tamako fix --review")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """確認用の書き出しと本番の書き出しが、同じものを描いているか検査する。

    出力を再検出して顔を探す方式は、検査器と本番の検出器が同じなら原理的に
    空振りするうえ、最頻の漏れ（部分被覆）は検出器では捕まらない。ここで
    保証できるのは配管の正しさ——座標変換の取り違え、レターボックスのずれ、
    フレーム同期のずれ——であって、検出の再現率ではない。そう割り切って、
    描いた箱の記録どうしを突き合わせる。
    """
    from .pipeline import site_spans, subset_plans
    from .render import render

    config = _load(args)
    mask_path = config.mask_image
    if not mask_path.is_file():
        _print(f"顔に重ねる画像が見つかりません: {mask_path}")
        return 1

    clips, plans, tracked, edits, sites = _prepare(args, config)
    spans = site_spans(sites, margin=1.0) if sites else {
        clip.path: list(plan.keep)[:1] for clip, plan in plans
    }
    subset = subset_plans(plans, spans) or list(plans)[:1]

    work = config.output_dir / ".tamako_work"
    mask_cfg = config.section("mask")
    logs = []
    for name, crf, preset in (("preview", 30, "ultrafast"), ("final", 20, "medium")):
        log_path = work / f"selftest_{name}.tsv"
        render(
            subset, tracked,
            output_path=work / f"selftest_{name}.mp4",
            mask_path=mask_path,
            mask_scale=float(mask_cfg["scale"]),
            mask_offset_y=float(mask_cfg.get("offset_y", 0.0)),
            crf=crf, preset=preset,
            draw_log=log_path, work_dir=work,
        )
        logs.append(log_path)

    a = logs[0].read_text(encoding="utf-8").splitlines()
    b = logs[1].read_text(encoding="utf-8").splitlines()
    if a == b:
        _print(f"配管の検査: OK （{len(a)} フレームで一致）")
        for path in logs:
            path.unlink(missing_ok=True)
            path.with_suffix(".mp4").unlink(missing_ok=True)
        return 0

    _print(f"配管の検査: 不一致 （{len(a)} 行 vs {len(b)} 行）")
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            _print(f"  最初の相違 {i} 行目:")
            _print(f"    確認用: {x}")
            _print(f"    本番:   {y}")
            break
    _print(f"  記録: {logs[0]} / {logs[1]}")
    return 1


def cmd_fix(args: argparse.Namespace) -> int:
    """要確認の箇所を出す（人が仕上げる工程）。

    cv2 の窓は削除した。置き換え先のブラウザ UI が入るまで、このコマンドは
    一覧までを受け持つ。
    """
    from .manual import OP_ADJUST
    from .pipeline import (
        analyze_clips, collect_clips, collect_sites, merge_manual,
        run_detection, track_clips,
    )
    from .sites import describe_sites, summarize_sites

    config = _load(args)
    mask_path = config.mask_image
    if not mask_path.is_file():
        _print(f"顔に重ねる画像が見つかりません: {mask_path}")
        return 1

    clips, _ = collect_clips(config)
    plans = analyze_clips(clips, config, face_model=args.face_model, on_progress=_status)
    detections = run_detection(clips, config, face_model=args.face_model)
    tracked = track_clips(clips, config, detections, on_progress=_status)
    # merge_manual の結果はサイト算出にだけ使う。ReviewSession には生の
    # tracked を渡す（refresh が人手修正を当てるので、渡すと二重に当たる）。
    merged, edits = merge_manual(clips, tracked, config)
    _clear_status()

    sites = collect_sites(plans, merged, edits=edits, config=config)

    _print(summarize_sites(sites, out_duration=sum(p.kept_seconds for _, p in plans)))
    _print("")
    _print("── 危険度の高い順 ────────────────────")
    _print(describe_sites(sites, limit=30))
    _print("")
    _print(f"  これまでの修正: {len(edits.effective())} 件  ({edits.path})")
    if any(op.op == OP_ADJUST for op in edits.effective()):
        _print("")
        _print("  ※ 位置調整が二重に当たっていた不具合を直しました。過去に記録した")
        _print("     scale / dx / dy は効きが以前の半分になり、被覆が減ったぶん")
        _print("     確認済みが失効して要確認が増えて見えます。調整済みの箇所は")
        _print("     一度見直してください。")

    if args.list_only or not sites:
        return 0

    _print("")
    _print("箱を直す画面は今 OpenCV の窓からブラウザに移している最中です。")
    _print("それまでは、直したい箇所を確認用に短く書き出せます:")
    _print("  tamako remask")
    return 0


def cmd_transcribe(args: argparse.Namespace) -> int:
    """収録音声から字幕ファイル（SRT）だけを作る。"""
    from .transcribe import transcribe, write_srt

    config = _load(args)
    sub = config.section("subtitle")
    audio = Path(args.audio).expanduser().resolve()
    srt_path = Path(args.srt).expanduser().resolve() if args.srt else audio.with_suffix(".srt")

    cues = transcribe(
        audio,
        model_size=str(sub["model"]),
        language=str(sub["language"]) or None,
        on_progress=_progress_line,
    )
    write_srt(cues, srt_path, max_chars_per_line=int(sub["max_chars_per_line"]))
    _print("")
    _print(f"字幕を書き出しました: {srt_path}  ({len(cues)} 枚)")
    _print("内容を直したい場合はテキストエディタで編集してから `tamako subtitle` を実行してください。")
    return 0


def cmd_subtitle(args: argparse.Namespace) -> int:
    """既にある SRT を焼き込み、音声を差し替える。"""
    from .subtitle import SubtitleStyle, burn_subtitles, check_audio_length

    config = _load(args)
    sub = config.section("subtitle")
    encode_cfg = config.section("encode")

    video = Path(args.video).expanduser().resolve()
    srt_path = Path(args.srt).expanduser().resolve()
    audio = Path(args.audio).expanduser().resolve() if args.audio else None
    output = (
        Path(args.out).expanduser().resolve()
        if args.out
        else config.output_dir / "final.mp4"
    )

    if audio is not None and (warning := check_audio_length(video, audio)):
        _print(f"注意: {warning}")
        _print("")

    _print("字幕を焼き込んでいます…")
    burn_subtitles(
        video, srt_path, output,
        audio=audio,
        audio_mode=str(sub["audio_mode"]),
        original_volume=float(sub["original_volume"]),
        style=SubtitleStyle(
            font=str(sub["font"]),
            font_size=int(sub["font_size"]),
            outline=int(sub["outline"]),
            margin_v=int(sub["margin_v"]),
        ),
        crf=int(encode_cfg["crf"]),
        preset=str(encode_cfg["preset"]),
        audio_bitrate=str(encode_cfg["audio_bitrate"]),
    )
    _print(f"完成しました: {output}")
    return 0


def cmd_finish(args: argparse.Namespace) -> int:
    """文字起こしと焼き込みを続けて行う（後半の工程）。"""
    from .transcribe import transcribe, write_srt
    from .subtitle import SubtitleStyle, burn_subtitles, check_audio_length

    config = _load(args)
    sub = config.section("subtitle")
    encode_cfg = config.section("encode")

    video = Path(args.video).expanduser().resolve()
    audio = Path(args.audio).expanduser().resolve()
    output_dir = config.output_dir
    srt_path = Path(args.srt).expanduser().resolve() if args.srt else output_dir / "subtitles.srt"
    output = Path(args.out).expanduser().resolve() if args.out else output_dir / "final.mp4"

    if (warning := check_audio_length(video, audio)):
        _print(f"注意: {warning}")
        _print("")

    cues = transcribe(
        audio,
        model_size=str(sub["model"]),
        language=str(sub["language"]) or None,
        on_progress=_progress_line,
    )
    write_srt(cues, srt_path, max_chars_per_line=int(sub["max_chars_per_line"]))
    _print("")
    _print(f"字幕: {srt_path}  ({len(cues)} 枚)")

    _print("字幕を焼き込んでいます…")
    burn_subtitles(
        video, srt_path, output,
        audio=audio,
        audio_mode=str(sub["audio_mode"]),
        original_volume=float(sub["original_volume"]),
        style=SubtitleStyle(
            font=str(sub["font"]),
            font_size=int(sub["font_size"]),
            outline=int(sub["outline"]),
            margin_v=int(sub["margin_v"]),
        ),
        crf=int(encode_cfg["crf"]),
        preset=str(encode_cfg["preset"]),
        audio_bitrate=str(encode_cfg["audio_bitrate"]),
    )
    _print("")
    _print(f"完成しました: {output}")
    _print(f"字幕を直したくなったら {srt_path} を編集して:")
    _print(f"  tamako subtitle --video \"{video}\" --srt \"{srt_path}\" --audio \"{audio}\"")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tamako",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            撮影素材を撮影順に並べ、不要な区間を切り、顔を PNG で隠し、
            あとから収録した声で字幕を付けるまでを自動化します。

            進め方:
              1) tamako init      設定ファイルとフォルダを用意する
              2) tamako check     切り方の下見（書き出さない）
              3) tamako edit      並べる・切る・顔を隠す
                 → 出来た動画を見ながら声を録音する
              4) tamako finish    文字起こしして字幕を焼き込む
        """),
    )
    parser.add_argument("--version", action="version", version=f"tamako {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", "-c", help="設定ファイル (config.json)")
    common.add_argument("--face-model", help="顔検出モデル (.onnx) の場所を明示する")

    folders = argparse.ArgumentParser(add_help=False)
    folders.add_argument("--input", "-i", help="素材フォルダ（設定より優先）")
    folders.add_argument("--output", "-o", help="書き出し先フォルダ（設定より優先）")
    folders.add_argument("--mask", "-m", help="顔に重ねる PNG（設定より優先）")
    folders.add_argument("--mode", choices=["any", "both", "silence", "face"],
                         help="カットの基準（設定より優先）")

    p_init = subparsers.add_parser("init", help="設定ファイルとフォルダの雛形を作る")
    p_init.add_argument("directory", nargs="?", default=".", help="作る場所（既定: 今のフォルダ）")
    p_init.add_argument("--force", action="store_true", help="既存の config.json を上書きする")
    p_init.set_defaults(func=cmd_init)

    p_detect = subparsers.add_parser("detect", parents=[common, folders],
                                     help="顔検出だけを先に走らせる（結果はキャッシュされる）")
    p_detect.set_defaults(func=cmd_detect)

    p_check = subparsers.add_parser("check", parents=[common, folders],
                                    help="書き出さずに並び順とカット結果を見る")
    p_check.set_defaults(func=cmd_check)

    p_edit = subparsers.add_parser("edit", parents=[common, folders],
                                   help="並べる・切る・顔を隠す")
    p_edit.add_argument("--name", default="edited.mp4", help="出力ファイル名（既定: edited.mp4）")
    p_edit.add_argument("--diagnostic", action="store_true",
                        help="マスクの代わりに覆う範囲と由来を描く（位置合わせの確認用）")
    p_edit.add_argument("--only", metavar="開始-終了",
                        help="出力時刻のこの範囲だけを本番画質で書き出す（例 00:03:10-00:03:20）")
    p_edit.set_defaults(func=cmd_edit)

    p_remask = subparsers.add_parser("remask", parents=[common, folders],
                                     help="要確認の箇所だけをフル解像度で短く書き出す")
    p_remask.add_argument("--margin", type=float, default=1.5,
                          help="前後に付ける余白（秒。既定: 1.5）")
    p_remask.add_argument("--diagnostic", action="store_true",
                          help="マスクの代わりに覆う範囲と由来を描く")
    p_remask.set_defaults(func=cmd_remask)

    p_self = subparsers.add_parser("selftest", parents=[common, folders],
                                   help="確認用と本番で同じものを描いているか検査する")
    p_self.set_defaults(func=cmd_selftest)

    p_fix = subparsers.add_parser("fix", parents=[common, folders],
                                  help="要確認の箇所を危険度順に出す")
    p_fix.add_argument("--review", action="store_true",
                       help="（互換のため残しています。今は既定の動作です）")
    p_fix.add_argument("--list", dest="list_only", action="store_true",
                       help="一覧だけを出す")
    p_fix.set_defaults(func=cmd_fix)

    p_tr = subparsers.add_parser("transcribe", parents=[common],
                                 help="収録音声から字幕ファイルだけを作る")
    p_tr.add_argument("--audio", "-a", required=True, help="収録した音声ファイル")
    p_tr.add_argument("--srt", help="書き出す字幕の場所")
    p_tr.set_defaults(func=cmd_transcribe)

    p_sub = subparsers.add_parser("subtitle", parents=[common],
                                  help="既にある字幕を焼き込む")
    p_sub.add_argument("--video", "-v", required=True, help="対象の動画")
    p_sub.add_argument("--srt", required=True, help="焼き込む字幕 (SRT)")
    p_sub.add_argument("--audio", "-a", help="差し替える音声")
    p_sub.add_argument("--out", help="出力先")
    p_sub.set_defaults(func=cmd_subtitle)

    p_fin = subparsers.add_parser("finish", parents=[common],
                                  help="文字起こしと字幕の焼き込みをまとめて行う")
    p_fin.add_argument("--video", "-v", required=True, help="前半で書き出した動画")
    p_fin.add_argument("--audio", "-a", required=True, help="収録した音声ファイル")
    p_fin.add_argument("--srt", help="字幕の書き出し先")
    p_fin.add_argument("--out", help="完成品の書き出し先")
    p_fin.set_defaults(func=cmd_finish)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # 設定の指定が無ければ、今のフォルダの config.json を自動で拾う。
    if getattr(args, "config", None) is None and hasattr(args, "func") and args.func is not cmd_init:
        default_config = Path.cwd() / "config.json"
        if default_config.is_file():
            args.config = str(default_config)

    try:
        return int(args.func(args))
    except ConfigError as exc:
        _print(f"設定の誤りです: {exc}")
        return 2
    except KeyboardInterrupt:
        _print("\n中断しました。")
        return 130
    except Exception as exc:  # noqa: BLE001 — 端末には読める形だけ出す
        _print("")
        _print(f"エラー: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
