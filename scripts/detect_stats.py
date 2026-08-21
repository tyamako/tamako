# -*- coding: utf-8 -*-
r"""faces.jsonl を読んで、顔が写らない数字だけを要約する。

使い方:  python stats.py .tamako_work\detect\xxxx.jsonl
引数を省くと .tamako_work\detect\ の中の *.jsonl を全部見る。
"""
import glob
import json
import sys


def summarize(path):
    meta = None
    total = -1
    hits = {}          # フレーム番号 -> そのフレームの箱
    scenes = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            kind = row.get("type")
            if kind == "meta":
                meta = row
                continue
            if kind == "end":
                total = int(row["frames_total"])
                continue
            boxes = row.get("boxes") or []
            if boxes:
                hits[int(row["f"])] = boxes
            if row.get("scene"):
                scenes += 1

    if meta is None or total < 0:
        print("  この jsonl は書きかけです（end 行がありません）")
        return

    fps = float(meta["fps"]) or 30.0
    width = int(meta["width"])
    height = int(meta["height"])

    print("  素材        : %dx%d / %.3f fps / %d フレーム (%.1fs)"
          % (width, height, fps, total, total / fps))
    print("  検出の設定  : detect_width=%s / model=%s"
          % (meta.get("detect_width"), str(meta.get("model_sha"))[:8]))
    print("  場面変化    : %d 箇所" % scenes)

    if total == 0:
        return
    print("  顔が取れた  : %d / %d フレーム (%.1f%%)"
          % (len(hits), total, len(hits) * 100.0 / total))
    if not hits:
        print("  → 1 フレームも取れていません。config.json の mask.detect_width を"
              " 1280 に、mask.score_threshold を 0.3 に下げて試してください。")
        return

    # 1 フレームに何個の箱が出たか
    counts = {}
    for boxes in hits.values():
        counts[len(boxes)] = counts.get(len(boxes), 0) + 1
    print("  箱の個数    : " + " / ".join(
        "%d個=%dフレーム" % (n, counts[n]) for n in sorted(counts)))

    # スコアの分布。低いほど「自信のない検出」で、閾値を下げれば拾える余地がある。
    scores = sorted(box[4] for boxes in hits.values() for box in boxes)
    marks = [("最小", 0.0), ("下位10%", 0.10), ("中央", 0.50), ("上位10%", 0.90), ("最大", 1.0)]
    print("  スコア分布  : " + " / ".join(
        "%s=%.3f" % (name, scores[min(len(scores) - 1, int(q * (len(scores) - 1)))])
        for name, q in marks))

    # 顔の大きさ。画面幅に対する比。小さすぎると detect_width を上げる価値がある。
    sizes = sorted(box[2] / float(width) for boxes in hits.values() for box in boxes)
    print("  顔の幅      : " + " / ".join(
        "%s=%.1f%%" % (name, 100.0 * sizes[min(len(sizes) - 1, int(q * (len(sizes) - 1)))])
        for name, q in marks))
    small = 1.0 / max(1.0, float(meta.get("detect_width") or width))
    px = sizes[0] * float(meta.get("detect_width") or width)
    print("                (最小の顔は検出解像度で約 %.0f ピクセル幅。"
          "20 を切ると取りこぼしが増えます)" % px)

    # ここが本題。連続して取れなかった区間＝マスクが推定に頼る区間。
    gaps = []
    run = 0
    for index in range(total):
        if index in hits:
            if run:
                gaps.append((index - run, run))
            run = 0
        else:
            run += 1
    if run:
        gaps.append((total - run, run))

    if not gaps:
        print("  途切れ      : なし（全フレームで取れています）")
        return
    gaps.sort(key=lambda g: g[1], reverse=True)
    long_gaps = [g for g in gaps if g[1] / fps >= 0.5]
    print("  途切れ      : %d 箇所 / うち 0.5 秒以上が %d 箇所"
          % (len(gaps), len(long_gaps)))
    print("  長い順に 5 つ:")
    for start, length in gaps[:5]:
        print("    %6.2fs から %5.2fs ぶん (%d フレーム)"
              % (start / fps, length / fps, length))


def main(argv):
    paths = argv[1:] or sorted(glob.glob(".tamako_work/detect/*.jsonl"))
    if not paths:
        print("faces.jsonl が見つかりません。先に tamako check を実行してください。")
        return 1
    for path in paths:
        print("── %s" % path)
        try:
            summarize(path)
        except Exception as exc:  # noqa: BLE001 — 診断用なので原因を出して続ける
            print("  読めませんでした: %r" % (exc,))
        print("")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
