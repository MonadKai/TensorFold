#!/usr/bin/env python3
"""Pair 64 fixed-input performance receipts and distinguish native token equality from cross-engine differences."""
import argparse
import hashlib
import json
from pathlib import Path


def paired(reference, candidate, cross_engine=False):
    for report in (reference, candidate):
        if not report["complete"] or not report["warmup_work_matches"]:
            raise ValueError("incomplete performance cell")
        rows = report["requests"]
        if len(rows) != 64 or len({row["sha256"] for row in rows}) != 64:
            raise ValueError("64 unique fixed inputs are required")
        if any(row["status"] != 200 for row in rows):
            raise ValueError("failed request cannot enter a paired comparison")
    for key in ("kind", "mode", "temperature", "seed", "output_cap", "concurrency"):
        if reference["workload"][key] != candidate["workload"][key]:
            raise ValueError("workload differs: " + key)
    for key in ("context", "model_revision", "dtype", "cache_condition"):
        if reference["runtime"].get(key) is None or reference["runtime"].get(key) != candidate["runtime"].get(key):
            raise ValueError("runtime differs or lacks identity: " + key)
    if reference["workload"]["mode"] != "chat":
        raise ValueError("paired token work requires chat response usage")
    left = {row["sha256"]: row for row in reference["requests"]}
    right = {row["sha256"]: row for row in candidate["requests"]}
    if left.keys() != right.keys():
        raise ValueError("fixed waveform identities differ")
    differences = []
    token_counts = []
    for sha, old in left.items():
        new = right[sha]
        responses = [old["response"], new["response"]]
        identities = []
        counts = []
        for value in responses:
            choice = value["choices"][0]
            count = value.get("usage", {}).get("completion_tokens")
            ids = choice.get("token_ids")
            if ids is None:
                ids = value.get("tensorfold", {}).get("token_ids")
            if (not isinstance(ids, list) or type(count) is not int or count <= 0 or len(ids) != count
                    or any(type(token) is not int or token < 0 for token in ids)):
                raise ValueError("token identities or actual work are missing")
            if reference["workload"]["kind"] == "natural" and choice["finish_reason"] != "stop":
                raise ValueError("natural workload was truncated")
            if reference["workload"]["kind"] == "equal_tokens" and count != reference["workload"]["output_cap"]:
                raise ValueError("equal-token work differs from the fixed cap")
            identities.append(dict(text=choice["message"]["content"], token_ids=ids))
            counts.append(count)
        token_counts.append(dict(sha256=sha, reference=counts[0], candidate=counts[1]))
        if identities[0] != identities[1]:
            differences.append(dict(sha256=sha, reference=identities[0], candidate=identities[1]))
    equal_work = all(row["reference"] == row["candidate"] for row in token_counts)
    result = dict(schema=1, waveform_count=64, equal_output_work=equal_work,
                  actual_token_counts=token_counts, differing_waveforms=differences, cross_engine=cross_engine,
                  passed=cross_engine or not differences,
                  limitation="Cross-engine differences are reported; this comparison does not establish ASR quality.")
    if not cross_engine and not equal_work:
        result["passed"] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--cross-engine", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = paired(json.loads(args.reference.read_text()), json.loads(args.candidate.read_text()), args.cross_engine)
    result.update(reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                  candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("passed", "waveform_count", "equal_output_work")}))
    return not result["passed"]


if __name__ == "__main__":
    raise SystemExit(main())
