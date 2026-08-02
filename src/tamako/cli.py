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
    "face_analysis_fps": 3.0,   // 顔の有無を調べる細かさ（大きいほど正確・遅い）
    "face_score_threshold": 0.6,// 顔と判定する確信度
    "face_hold_sec": 1.0,       // この長さ以下の検出の途切れは無視する
    "padding_sec": 0.25,        // 残す区間の前後に足す余白（語頭語尾の切れ防止）
    "min_keep_sec": 0.6,        // これより短い残り区間は捨てる
    "min_cut_sec": 0.5          // これより短いカットは行わない（細切れ防止）
  },

  // ── 顔の隠し方 ───────────────────────────────────────
  "mask": {
    "score_threshold": 0.5,   // 隠すときは低めにする（見逃すより多めに隠す）
    "scale": 2.0,             // 検出枠の何倍を隠すか。1.6 だと髪が出ます
    "offset_y": 0.0,          // 上下の微調整（顔の高さに対する割合、負で上）
    "hold_sec": 0.7,          // 見失っても直前の位置に出し続ける長さ
    "detect_width": 640,      // 検出用に縮小する幅（小さいほど速い）
    "detect_every_n_frames": 1, // 1 なら毎フレーム検出（最も安全）
    "low_score_warn": 0.75    // これ未満の確信度は報告書に記録する
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
    from .faces import FaceDetector, ensure_model
    from .ordering import describe_order
    from .pipeline import analyze_clips, collect_clips, describe_plans
    from .render import render
    from .report import summarize, write_cut_list_csv, write_json_report

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

    detector = FaceDetector(
        ensure_model(args.face_model), score_threshold=float(mask_cfg["score_threshold"])
    )
    report = render(
        plans,
        output_path=output_path,
        mask_path=mask_path,
        detector=detector,
        mask_scale=float(mask_cfg["scale"]),
        mask_offset_y=float(mask_cfg.get("offset_y", 0.0)),
        hold_sec=float(mask_cfg["hold_sec"]),
        detect_width=int(mask_cfg["detect_width"]) or None,
        detect_every_n_frames=int(mask_cfg["detect_every_n_frames"]),
        low_score_warn=float(mask_cfg["low_score_warn"]),
        crf=int(encode_cfg["crf"]),
        preset=str(encode_cfg["preset"]),
        pix_fmt=str(encode_cfg["pix_fmt"]),
        audio_bitrate=str(encode_cfg["audio_bitrate"]),
        on_progress=_progress_line,
    )

    json_path = write_json_report(
        output_dir / "edit_report.json",
        clip_plans=plans,
        render_report=report,
        settings={"cut": config.section("cut"), "mask": mask_cfg},
    )
    csv_path = write_cut_list_csv(output_dir / "cut_list.csv", report)

    _print(summarize(plans, report))
    _print("")
    _print("── 書き出したもの ────────────────────")
    _print(f"  {report.output}")
    _print(f"  {csv_path}   （カット位置の一覧。Filmora で手直しする際の下敷き）")
    _print(f"  {json_path}  （全記録）")
    _print("")
    _print("次の工程: 動画を見ながら声を録音し、その音声ファイルを用意してから")
    _print(f"  tamako finish --video \"{report.output}\" --audio \"収録音声.wav\"")
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
    p_edit.set_defaults(func=cmd_edit)

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
