#!/usr/bin/env python3
"""Score a pinned, locally supplied labelled ASR manifest; retain each raw response and edit counts."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import re
from types import SimpleNamespace
import unicodedata

from bench_qwen3_asr import run_one


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.sub(r"[^\w\s]", "", text).split())


def distance(left, right):
    previous = list(range(len(right) + 1))
    for i, x in enumerate(left, 1):
        current = [i]
        for j, y in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("model")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("concurrency must be positive")
    manifest = json.loads(args.manifest.read_text())
    options = SimpleNamespace(model=args.model, base_url=args.base_url, temperature=0, seed=None, tokens=1024, language="en")
    clips = manifest["clips"]

    def one(clip):
        path = args.manifest.parent / clip["file"]
        result = run_one(options, path, "transcription")
        if result["sha256"] != clip["sha256"]:
            raise ValueError("audio identity differs from manifest")
        response = result["response"]
        text = response.get("text") if isinstance(response, dict) else None
        valid = result["status"] == 200 and isinstance(text, str)
        reference = normalize(clip["reference"])
        hypothesis = normalize(text) if valid else None
        return {"id": clip["id"], "reference": clip["reference"], "normalized_reference": reference,
                "normalized_hypothesis": hypothesis, "valid": valid, "raw": result,
                "word_edits": distance(reference.split(), hypothesis.split()) if valid else None,
                "reference_words": len(reference.split()),
                "character_edits": distance(reference.replace(" ", ""), hypothesis.replace(" ", "")) if valid else None,
                "reference_characters": len(reference.replace(" ", ""))}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        records = list(pool.map(one, clips))
    valid = all(r["valid"] for r in records)
    words = sum(r["reference_words"] for r in records)
    characters = sum(r["reference_characters"] for r in records)
    result = {"manifest": manifest, "model": args.model, "concurrency": args.concurrency,
              "normalization": "NFKC, casefold, remove punctuation, collapse whitespace; CER excludes spaces",
              "records": records, "complete": valid,
              "wer": sum(r["word_edits"] for r in records) / words if valid and words else None,
              "cer": sum(r["character_edits"] for r in records) / characters if valid and characters else None}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("complete", "wer", "cer")}))
    return int(not valid)


if __name__ == "__main__":
    raise SystemExit(main())
