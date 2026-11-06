#!/usr/bin/env python3
"""Pair the frozen native long-audio and mixed-load ablations before attributing performance gains."""
import argparse
import hashlib
import json
from pathlib import Path

from check_qwen3_asr_determinism import fingerprint


def ordinals(rows, expected):
    actual = [row.get("index") for row in rows]
    if not all(type(value) is int for value in actual) or actual != expected:
        raise ValueError("missing, duplicate or reordered request ordinal")


def groups(report):
    if not report.get("complete"):
        raise ValueError("incomplete scheduler cell")
    if report.get("schema") == "asr-performance-v1":
        work = report["workload"]
        expected = dict(mode="transcription", kind="natural", concurrency=1, reps=1,
                        temperature=0, seed=0, output_cap=1024, require_natural_stop=True)
        if any(work.get(key) != value for key, value in expected.items()):
            raise ValueError("the fixed single-long natural workload is required")
        if (not report.get("warmup_work_matches") or len(report["requests"]) != 1
                or len(report["warmup_requests"]) != 1 or report["runtime"].get("context") != 8192):
            raise ValueError("single-long requests, warmup or context differ")
        ordinals(report["requests"], [0])
        ordinals(report["warmup_requests"], [0])
        if report["requests"][0]["sha256"] != report["warmup_requests"][0]["sha256"]:
            raise ValueError("single-long warmup uses another input")
        return dict(measured=report["requests"], warmup=report["warmup_requests"])
    expected = dict(short_requests=64, short_concurrency=4, context=8192, temperature=0,
                    seed=0, output_cap=1024, long_arrivals_seconds=[0, 0.5], short_start_seconds=0.25)
    if (report.get("settings") != expected or not report.get("live_overlap_observed")
            or not report.get("short_identity_equal") or len(report["long"]) != 2
            or len(report["short"]) != 64 or len(report["warmup"]) != 2):
        raise ValueError("the frozen two-long/64-short workload and live overlap are required")
    long = [row["raw"] for row in report["long"]]
    short = [row["raw"] for row in report["short"]]
    ordinals(long, [0, 1])
    ordinals(short, list(range(64)))
    ordinals(report["warmup"], [0, 0])
    waves = [row["sha256"] for row in report["warmup"]]
    if len(set(waves)) != 2 or any(row["sha256"] != waves[i % 2] for i, row in enumerate(short)):
        raise ValueError("short requests do not follow the fixed two-input cycle")
    return dict(long=long, short=short, warmup=report["warmup"])


def compare(reference, candidate):
    left, right = groups(reference), groups(candidate)
    if left.keys() != right.keys():
        raise ValueError("different scheduler workload kinds")
    if "workload" in reference:
        if reference["workload"] != candidate["workload"] or reference["model"] != candidate["model"]:
            raise ValueError("different input or sampler configuration")
    differences, paired_groups = [], []
    for phase in left:
        if len(left[phase]) != len(right[phase]):
            raise ValueError("different request counts")
        old_tokens, new_tokens = 0, 0
        for index, (old, new) in enumerate(zip(left[phase], right[phase])):
            if old["sha256"] != new["sha256"] or old.get("index") != new.get("index"):
                raise ValueError("different input identity or request order")
            a, b = fingerprint(old), fingerprint(new)
            valid = a is not None and b is not None and all(reason == "stop" for reason in
                                                            a["finish_reason"] + b["finish_reason"])
            if not valid or a != b:
                differences.append(dict(phase=phase, index=index, waveform_sha256=old["sha256"],
                                        reference=a, candidate=b, reason="invalid or unequal native output work"))
            if a is not None:
                old_tokens += sum(a["completion_tokens"])
            if b is not None:
                new_tokens += sum(b["completion_tokens"])
        paired_groups.append(dict(phase=phase, requests=len(left[phase]),
                                  reference_tokens=old_tokens, candidate_tokens=new_tokens))
    return dict(schema=1, passed=not differences, equal_output_work=not differences,
                groups=paired_groups, differences=differences,
                note="Pairs request ordinals, waveform SHA, ordered chunk/token fingerprints, text and stopping; duplicate short inputs are intentional.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare(json.loads(args.reference.read_text()), json.loads(args.candidate.read_text()))
    except (ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
        result = dict(schema=1, passed=False, equal_output_work=False, error=str(exc))
    result["input_sha256"] = {"reference": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                              "candidate": hashlib.sha256(args.candidate.read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
