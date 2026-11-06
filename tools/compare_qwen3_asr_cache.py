#!/usr/bin/env python3
"""Compare repeated-input eager reference runs without hiding output-work differences."""
import argparse
import hashlib
import json
from pathlib import Path


def compare(disabled, enabled):
    for report in (disabled, enabled):
        if not report["complete"] or not report["warmup_work_matches"]:
            raise ValueError("incomplete cache control")
        if len(report["requests"]) != 64 or len(report["warmup_requests"]) != 32:
            raise ValueError("cache controls require 64 measured and 32 warmup requests")
        if report["runtime"]["engine"] != "vllm":
            raise ValueError("reference engine required")
        if report["runtime"].get("enforce_eager") is not True:
            raise ValueError("both cache controls must use eager execution")
    for key in ("kind", "mode", "temperature", "seed", "output_cap", "concurrency"):
        if disabled["workload"][key] != enabled["workload"][key]:
            raise ValueError("workload differs: " + key)
    for key in ("context", "model_revision", "dtype", "image_id"):
        if disabled["runtime"].get(key) is None or disabled["runtime"][key] != enabled["runtime"].get(key):
            raise ValueError("runtime identity differs or is missing: " + key)
    if disabled["runtime"]["cache_condition"] != "disabled" or enabled["runtime"]["cache_condition"] != "enabled":
        raise ValueError("expected disabled/enabled cache conditions")
    if disabled["workload"]["kind"] != "natural" or disabled["workload"]["mode"] != "chat":
        raise ValueError("natural-stop chat controls required")
    differences = []
    equal_counts = True
    for phase, count in (("requests", 64), ("warmup_requests", 32)):
        expected_indices = list(range(count)) if phase == "requests" else [0] * count
        for report in (disabled, enabled):
            indices = [row["index"] for row in report[phase]]
            if any(type(index) is not int for index in indices) or indices != expected_indices:
                raise ValueError("request ordinals are incomplete or repeated")
        waves = [row["sha256"] for row in disabled[phase]]
        if len(set(waves)) != 4 or waves != [waves[i % 4] for i in range(count)]:
            raise ValueError("four repeated inputs required in fixed order")
        for index, (left, right) in enumerate(zip(disabled[phase], enabled[phase])):
            if left["sha256"] != right["sha256"] or left["index"] != right["index"]:
                raise ValueError("ordered request identities differ")
            identities = []
            for row in (left, right):
                if row["status"] != 200:
                    raise ValueError("failed cache request")
                response = row["response"]
                choice = response["choices"][0]
                tokens = choice.get("token_ids", response.get("tensorfold", {}).get("token_ids"))
                actual = response["usage"]["completion_tokens"]
                if type(actual) is not int or actual <= 0 or not isinstance(tokens, list) or len(tokens) != actual:
                    raise ValueError("actual token work is missing")
                if choice["finish_reason"] != "stop":
                    raise ValueError("natural cache request was truncated")
                identities.append(dict(text=choice["message"]["content"], tokens=tokens, count=actual))
            equal_counts &= identities[0]["count"] == identities[1]["count"]
            if identities[0] != identities[1]:
                differences.append(dict(phase=phase, index=index, sha256=left["sha256"],
                                        disabled=identities[0], enabled=identities[1]))
    old = disabled["measurements"]["batch_wall_seconds"]
    new = enabled["measurements"]["batch_wall_seconds"]
    if old <= 0 or new <= 0:
        raise ValueError("invalid makespan")
    return dict(schema=1, measured_requests=64, warmup_requests=32, unique_waveforms=4,
                equal_output_work=equal_counts, output_differences=differences,
                disabled_over_enabled_makespan=old / new if equal_counts else None,
                cache_metrics=dict(disabled=disabled["cache_metric_deltas"], enabled=enabled["cache_metric_deltas"]),
                limitation="Joint observed cache configuration; this does not isolate prefix, processor, or encoder cache savings or establish recognition quality.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("disabled", type=Path)
    parser.add_argument("enabled", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.disabled.read_text()), json.loads(args.enabled.read_text()))
    result.update(disabled_sha256=hashlib.sha256(args.disabled.read_bytes()).hexdigest(),
                  enabled_sha256=hashlib.sha256(args.enabled.read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
