"""設定ファイルの読み込み。

設定は JSON で書くが、`//` と `/* */` のコメントを許可する。
GUI を持たない道具なので、設定ファイル自体が説明書を兼ねられたほうがよい。
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

# 既定値。設定ファイルに書かれた値だけがこの上に重ねられる。
DEFAULTS: Dict[str, Any] = {
    "input_dir": "input",
    "output_dir": "output",
    "mask_image": "mask.png",
    "video_extensions": [".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mts", ".m2ts", ".wmv"],
    "cut": {
        # any  : 無音 または 顔なし を切る（既定・最も短くなる）
        # both : 無音 かつ 顔なし のときだけ切る（最も安全）
        # silence / face : どちらか一方の基準だけで切る
        "mode": "any",
        "silence_db": -32.0,
        "silence_min_sec": 0.8,
        "face_analysis_fps": 3.0,
        "face_score_threshold": 0.6,
        "face_hold_sec": 1.0,
        "padding_sec": 0.25,
        "min_keep_sec": 0.6,
        "min_cut_sec": 0.5,
        # カット判定に使う検出解像度。mask.detect_width とは独立させてある。
        # カットは「誤検出で残りすぎる」のが困り、マスクは「見逃して顔が出る」のが
        # 困る、という非対称があるため、マスク側の解像度を上げてもここは変えない。
        "detect_width": 640,
    },
    "mask": {
        # 顔隠しは検出漏れがそのまま顔バレになる。閾値は低め、箱は大きめ、
        # 見失っても hold_sec のあいだは直前の位置に出し続ける、が既定方針。
        "score_threshold": 0.5,
        # 検出器が返す枠は目・鼻・口だけの狭いもので、額・髪・顎を含まない。
        # 実測では 1.6 倍だと髪が角から出る。2.0 倍で頭部が隠れる。
        "scale": 2.0,
        # 重ねる位置の上下微調整（顔の高さに対する割合、負で上）。
        # 髪が多い人は少し上げるとよいが、下げすぎると顎が出る。
        "offset_y": 0.0,
        "hold_sec": 0.7,
        "detect_width": 640,
        "detect_every_n_frames": 1,
        "low_score_warn": 0.75,
    },
    "encode": {
        "crf": 20,
        "preset": "medium",
        "audio_bitrate": "192k",
        "pix_fmt": "yuv420p",
    },
    "subtitle": {
        "model": "small",
        "language": "ja",
        "font": "Yu Gothic UI",
        "font_size": 42,
        "margin_v": 60,
        "outline": 3,
        "max_chars_per_line": 20,
        # replace : 収録音声で元音声を置き換える
        # mix     : 元音声を下げて収録音声を重ねる
        "audio_mode": "replace",
        "original_volume": 0.15,
    },
}

_LINE_COMMENT = re.compile(r"(?<!:)//[^\n\r]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)


def _strip_comments(text: str) -> str:
    """JSON からコメントを除去する。文字列リテラル内の // は残す。"""
    out = []
    in_string = False
    escaped = False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt == "/":
                while i < len(text) and text[i] not in "\r\n":
                    i += 1
                continue
            if nxt == "*":
                end = text.find("*/", i + 2)
                i = len(text) if end == -1 else end + 2
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _unknown_keys(loaded: Dict[str, Any], defaults: Dict[str, Any], prefix: str = "") -> list[str]:
    """既定に無いキーを、ネストの中まで再帰的に探す。

    最上位しか見ないと "mask": {"score_treshold": ...} のような綴り間違いが
    黙って無視される。この設定は「隠し漏れがあったら閾値を下げろ」と README が
    指示する、最も打鍵される項目なので、間違いは必ずエラーにする。
    """
    result: list[str] = []
    for key, value in loaded.items():
        path = f"{prefix}{key}"
        if key not in defaults:
            result.append(path)
        elif isinstance(value, dict) and isinstance(defaults[key], dict):
            result.extend(_unknown_keys(value, defaults[key], prefix=f"{path}."))
    return result


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@dataclass
class Config:
    """解決済みの設定。相対パスは設定ファイルの位置を基準に絶対化してある。"""

    data: Dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULTS))
    source: Path | None = None
    base_dir: Path = field(default_factory=Path.cwd)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def section(self, name: str) -> Dict[str, Any]:
        value = self.data.get(name, {})
        if not isinstance(value, dict):
            raise ConfigError(f"設定 '{name}' はオブジェクトである必要があります")
        return value

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (self.base_dir / path).resolve()

    @property
    def input_dir(self) -> Path:
        return self.resolve_path(self.data["input_dir"])

    @property
    def output_dir(self) -> Path:
        return self.resolve_path(self.data["output_dir"])

    @property
    def mask_image(self) -> Path:
        return self.resolve_path(self.data["mask_image"])


class ConfigError(Exception):
    """設定ファイルが読めない、または内容が不正。"""


def load_config(path: str | Path | None) -> Config:
    """設定ファイルを読み、既定値に重ねて返す。path が None なら既定値のみ。"""
    if path is None:
        return Config()

    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise ConfigError(f"設定ファイルが見つかりません: {config_path}")

    raw = config_path.read_text(encoding="utf-8-sig")
    try:
        loaded = json.loads(_strip_comments(raw))
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"設定ファイルの書式が不正です ({config_path}): {exc.lineno}行目 {exc.msg}"
        ) from exc

    if not isinstance(loaded, dict):
        raise ConfigError(f"設定ファイルの最上位はオブジェクトである必要があります: {config_path}")

    unknown = _unknown_keys(loaded, DEFAULTS)
    if unknown:
        raise ConfigError(
            "設定ファイルに未知のキーがあります: "
            + ", ".join(sorted(unknown))
            + "（綴り間違いの可能性があります）"
        )

    return Config(
        data=_deep_merge(DEFAULTS, loaded),
        source=config_path,
        base_dir=config_path.parent.resolve(),
    )
