#!/usr/bin/env python3
"""Apply the preregistered paired bilingual quality gate without changing input selection."""
import argparse
import hashlib
import json
from pathlib import Path
import random

from bench_qwen3_asr_accuracy import distance, normalize

FROZEN_MANIFEST_SHA256 = "0d8a427d232e76df2dbfa0d6bdc314799db47d4006050b1d0fe7bbf75f5a552a"


def validate_report(report):
    expected = dict(model="Qwen/Qwen3-ASR-1.7B", seed=0, temperature=0, max_completion_tokens=1024)
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("model or sampling differs from the preregistered contract")
    manifest = report["manifest"]
    canonical = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    if report.get("manifest_sha256") != FROZEN_MANIFEST_SHA256 or hashlib.sha256(canonical).hexdigest() != FROZEN_MANIFEST_SHA256:
        raise ValueError("report does not bind the frozen manifest bytes")
    if not report.get("complete") or not report.get("requires_observed_natural_stop"):
        raise ValueError("incomplete reports cannot pass native qualification")
    clips = manifest["clips"]
    if manifest.get("schema") != "qwen3-asr-bilingual-eval-v1" or len(clips) != 256:
        raise ValueError("qualification requires the frozen bilingual 256-utterance manifest")
    if manifest["mandarin"]["revision"] != "2a509ef3a7a88d3234205eaeecba39fdd9d50518":
        raise ValueError("dataset revision differs from the preregistered contract")
    by_id = {clip["id"]: clip for clip in clips}
    records = report["records"]
    ids = [r["id"] for r in records]
    if len(by_id) != 256 or len(ids) != 256 or len(set(ids)) != 256 or set(ids) != by_id.keys():
        raise ValueError("records must uniquely and exactly cover all 256 frozen utterances")
    for language in ("en", "zh"):
        rows = [r for r in clips if r["language"] == language]
        speakers = {r["speaker"] for r in rows}
        if len(rows) != 128 or len(speakers) != 16 or any(sum(r["speaker"] == s for r in rows) != 8 for s in speakers):
            raise ValueError("language/speaker coverage differs from the preregistered contract")
    for row in records:
        clip = by_id[row["id"]]
        raw = row["raw"]
        response = raw["response"]
        if row["language"] != clip["language"] or row["reference"] != clip["reference"]:
            raise ValueError("record language or label differs from the frozen utterance")
        if not row["valid"] or not row["identity_matches"] or raw["sha256"] != clip["sha256"] or raw["status"] != 200:
            raise ValueError("failed request or waveform identity mismatch")
        chunks = response.get("tensorfold", {}).get("chunks", [])
        if row.get("natural_stop_observed") is not True or not chunks or any(c.get("finish_reason") != "stop" for c in chunks):
            raise ValueError("natural stopping lacks per-request evidence")
        expected = normalize(clip["reference"])
        actual = normalize(response["text"])
        if row["normalized_reference"] != expected or row["normalized_hypothesis"] != actual:
            raise ValueError("normalized text differs from the frozen label or raw response")
        counts = dict(reference_words=len(expected.split()), reference_characters=len(expected.replace(" ", "")),
                      word_edits=distance(expected.split(), actual.split()),
                      character_edits=distance(expected.replace(" ", ""), actual.replace(" ", "")))
        if any(row[name] != value for name, value in counts.items()):
            raise ValueError("edit counts or reference denominators differ from raw text")


def paired_gate(reference, candidate, language, iterations=10000):
    validate_report(reference)
    validate_report(candidate)
    left = {r["id"]: r for r in reference["records"] if r["language"] == language}
    right = {r["id"]: r for r in candidate["records"] if r["language"] == language}
    if not left or left.keys() != right.keys():
        raise ValueError("paired utterance identities differ")
    if reference["manifest_sha256"] != candidate["manifest_sha256"]:
        raise ValueError("manifest identity differs")
    pairs = []
    for key, old in left.items():
        new = right[key]
        if not old["valid"] or not new["valid"]:
            return {"language": language, "complete": False, "passed": False}
        if old["normalized_reference"] != new["normalized_reference"]:
            raise ValueError("paired labels differ")
        pairs.append((old, new))
    primary = "word" if language == "en" else "character"
    details = {}
    for unit in ("word", "character"):
        denominator = sum(x["reference_" + ("words" if unit == "word" else "characters")] for x, _ in pairs)
        old_edits = sum(x[unit + "_edits"] for x, _ in pairs)
        new_edits = sum(y[unit + "_edits"] for _, y in pairs)
        details[unit] = dict(reference_edits=old_edits, candidate_edits=new_edits, denominator=denominator,
                             reference_rate=old_edits / denominator, candidate_rate=new_edits / denominator,
                             degradation=(new_edits - old_edits) / denominator)
    differences = [{"id": old["id"], "reference_text": old["raw"]["response"]["text"],
                    "candidate_text": new["raw"]["response"]["text"]}
                   for old, new in pairs if old["raw"]["response"]["text"] != new["raw"]["response"]["text"]]
    rng = random.Random(20261106)
    counts = [(new[primary + "_edits"] - old[primary + "_edits"],
               old["reference_" + ("words" if primary == "word" else "characters")]) for old, new in pairs]
    boot = []
    for _ in range(iterations):
        selected = rng.choices(counts, k=len(counts))
        boot.append(sum(x for x, _ in selected) / sum(y for _, y in selected))
    boot.sort()
    upper = boot[int(0.95 * (iterations - 1))]
    passed = details[primary]["degradation"] <= 0.002 and upper <= 0.005
    if language == "en":
        passed = passed and details["character"]["degradation"] <= 0.001
    return dict(language=language, complete=True, passed=passed, utterances=len(pairs), primary=primary,
                edits=details, paired_bootstrap=dict(iterations=iterations, seed=20261106, upper_95=upper,
                                                    unit="utterance", quantile="floor(0.95*(n-1))"),
                differing_utterances=differences)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    candidate = json.loads(args.candidate.read_text())
    results = [paired_gate(reference, candidate, language) for language in ("en", "zh")]
    result = dict(schema=1, reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                  candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(), languages=results,
                  passed=all(x["passed"] for x in results))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "languages": [{k: r[k] for k in ("language", "passed")} for r in results]}))
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
