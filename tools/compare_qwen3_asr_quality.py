#!/usr/bin/env python3
"""Apply the preregistered paired bilingual quality gate without changing input selection."""
import argparse
import hashlib
import json
from pathlib import Path
import random


def paired_gate(reference, candidate, language, iterations=10000):
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
    manifest = reference["manifest"]
    if manifest.get("schema") != "qwen3-asr-bilingual-eval-v1" or len(manifest["clips"]) != 256:
        raise ValueError("qualification requires the frozen bilingual 256-utterance manifest")
    if manifest["mandarin"]["revision"] != "2a509ef3a7a88d3234205eaeecba39fdd9d50518":
        raise ValueError("dataset revision differs from the preregistered contract")
    for language in ("en", "zh"):
        clips = [r for r in manifest["clips"] if r["language"] == language]
        speakers = {r["speaker"] for r in clips}
        if len(clips) != 128 or len(speakers) != 16 or any(sum(r["speaker"] == s for r in clips) != 8 for s in speakers):
            raise ValueError("language/speaker coverage differs from the preregistered contract")
    if not all(r.get("requires_observed_natural_stop") for r in (reference, candidate)):
        raise ValueError("native qualification requires observed natural stops for both paths")
    results = [paired_gate(reference, candidate, language) for language in ("en", "zh")]
    result = dict(schema=1, reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                  candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(), languages=results,
                  passed=all(x["passed"] for x in results))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "languages": [{k: r[k] for k in ("language", "passed")} for r in results]}))
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
