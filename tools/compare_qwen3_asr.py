#!/usr/bin/env python3
"""Compare original Qwen and native stage exports, preserving numerical and decision differences."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def native_array(root, record):
    raw = np.fromfile(root / record["file"], dtype="<u2" if record["dtype"] == "bf16" else "<f4")
    if record["dtype"] == "bf16":
        raw = (raw.astype(np.uint32) << 16).view(np.float32)
    return raw.reshape(record["rows"], record["width"])


def metrics(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {actual.shape} != {expected.shape}")
    a, e = actual.astype(np.float64), expected.astype(np.float64)
    if not np.isfinite(a).all() or not np.isfinite(e).all():
        raise ValueError("non-finite stage output")
    delta = a - e
    norms = np.linalg.norm(a, axis=-1) * np.linalg.norm(e, axis=-1)
    cosine = np.divide((a * e).sum(-1), norms, out=np.ones_like(norms), where=norms != 0)
    return {
        "rows": a.shape[0], "width": a.shape[1], "max_abs": float(abs(delta).max()),
        "rmse": float(np.sqrt(np.mean(delta ** 2))),
        "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(e), 1e-30)),
        "min_row_cosine": float(cosine.min()), "mean_row_cosine": float(cosine.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    ref = json.loads((args.reference / "manifest.json").read_text())
    got = json.loads((args.native / "manifest.json").read_text())
    groups = defaultdict(list)
    for record in got["records"]:
        groups[record["name"]].append(native_array(args.native, record))
    position = defaultdict(int)
    comparisons = []
    for record in ref["records"]:
        name = record["name"]
        index = position[name]
        expected = np.load(args.reference / record["file"])
        expected = expected.reshape(-1, expected.shape[-1])
        actual = groups[name][index]
        position[name] += 1
        item = {"name": name, "occurrence": index, **metrics(actual, expected)}
        if name == "decoder.logits":
            e, a = expected[-1], actual[-1]
            best = np.argsort(e)[-2:][::-1]
            item.update(reference_top1=int(best[0]), native_top1=int(a.argmax()),
                        reference_margin=float(e[best[0]] - e[best[1]]),
                        native_margin_at_reference=float(a[best[0]] - a[best[1]]))
        comparisons.append(item)
    mel = np.load(args.reference / "mel.npy").reshape(128, -1)
    comparisons.insert(0, {"name": "audio.mel", "occurrence": 0, **metrics(groups["audio.mel"][0], mel)})
    logits = [v for v in comparisons if v["name"] == "decoder.logits"]
    result = {"comparisons": comparisons, "logit_rows": len(logits),
              "top1_equal": sum(v["native_top1"] == v["reference_top1"] for v in logits),
              "reference_generated_text": ref["generated_text"]}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "comparisons"}, ensure_ascii=False))
    for item in comparisons[:9]:
        print(json.dumps(item))


if __name__ == "__main__":
    main()
