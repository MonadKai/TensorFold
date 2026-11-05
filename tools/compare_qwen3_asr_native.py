#!/usr/bin/env python3
"""Keep exact native regression and preregistered throughput error/decision gates separate."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_qwen3_asr import metrics, native_array


def bf16_ulp(values):
    bits = np.abs(np.asarray(values, dtype=np.float32)).view(np.uint32)
    exponent = ((bits >> 23) & 255).astype(np.int32)
    return np.ldexp(np.ones(exponent.shape), np.where(exponent == 0, -133, exponent - 134))


def decisions(left, right):
    rows = []
    for old, new in zip(left, right):
        indices = np.arange(len(old))
        a = np.lexsort((indices, -old))[:2]
        b = np.lexsort((indices, -new))[:2]
        old_margin = float(old[a[0]] - old[a[1]])
        new_margin = float(new[b[0]] - new[b[1]])
        scale = max(abs(float(old[a[0]])), abs(float(old[a[1]])),
                    abs(float(new[b[0]])), abs(float(new[b[1]])))
        tie_bound = float(2 * bf16_ulp(scale))
        equal = int(a[0]) == int(b[0])
        near_tie = not equal and set(a.tolist()) == set(b.tolist()) and max(old_margin, new_margin) <= tie_bound
        rows.append(dict(reference_top2=a.tolist(), candidate_top2=b.tolist(), reference_margin=old_margin,
                         candidate_margin=new_margin, top1_equal=equal, near_tie=near_tie,
                         near_tie_bound=tie_bound, reference_contenders=old[a].tolist(),
                         candidate_at_reference_contenders=new[a].tolist()))
    return rows


def compare(reference, candidate, exact):
    left = json.loads((reference / "manifest.json").read_text())
    right = json.loads((candidate / "manifest.json").read_text())
    if left["prompt_ids"] != right["prompt_ids"] or left["teacher_tokens"] != right["teacher_tokens"]:
        raise ValueError("prompt or teacher-forcing input differs")
    if len(left["records"]) != len(right["records"]):
        raise ValueError("stage record counts differ")
    records = []
    logits = []
    before_head = True
    for old, new in zip(left["records"], right["records"]):
        if any(old[key] != new[key] for key in ("name", "rows", "width", "dtype")):
            raise ValueError("stage identity differs")
        a, b = native_array(reference, old), native_array(candidate, new)
        same = (reference / old["file"]).read_bytes() == (candidate / new["file"]).read_bytes()
        if old["name"] == "decoder.logits":
            before_head = False
        item = dict(name=old["name"], byte_equal=same, **metrics(b, a))
        required_exact = exact or before_head or old["name"] == "decoder.embeddings" or not old["name"].startswith("decoder.")
        item["requires_exact"] = required_exact
        item["passed"] = same if required_exact else True
        if old["name"] == "decoder.logits":
            delta = np.abs(b.astype(np.float64) - a)
            allowance = np.maximum(0.5, 4 * bf16_ulp(a))
            item["error_limit_exceeded_columns"] = int((delta > allowance).sum())
            item["maximum_normalized_delta"] = float((delta / allowance).max())
            item["decisions"] = decisions(a, b)
            item["bounded_logits_passed"] = item["relative_l2"] <= 0.005 and item["error_limit_exceeded_columns"] == 0
            item["passed"] = item["passed"] and item["bounded_logits_passed"]
            logits.extend(item["decisions"])
        records.append(item)
    top1 = sum(x["top1_equal"] for x in logits)
    nontie = sum(not x["top1_equal"] and not x["near_tie"] for x in logits)
    return dict(schema=1, exact_mode=exact, passed=bool(logits) and all(r["passed"] for r in records),
                reference_manifest_sha256=hashlib.sha256((reference / "manifest.json").read_bytes()).hexdigest(),
                candidate_manifest_sha256=hashlib.sha256((candidate / "manifest.json").read_bytes()).hexdigest(),
                stage_records=len(records), byte_equal_records=sum(r["byte_equal"] for r in records),
                logit_rows=len(logits), top1_equal=top1, top1_difference_rows=len(logits) - top1,
                near_tie_difference_rows=sum(x["near_tie"] for x in logits), nontie_difference_rows=nontie,
                default_decision_gate=bool(logits) and top1 / len(logits) >= 0.999 and nontie == 0,
                records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate, args.exact)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}))
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
