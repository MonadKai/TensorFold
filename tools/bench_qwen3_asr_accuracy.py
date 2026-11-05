#!/usr/bin/env python3
"""Score a pinned, locally supplied labelled ASR manifest; retain each raw response and edit counts."""
import argparse
import concurrent.futures
import hashlib
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require-natural-stop", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("concurrency must be positive")
    manifest = json.loads(args.manifest.read_text())
    clips = manifest["clips"]

    def one(clip):
        path = args.manifest.parent / clip["file"]
        language = clip.get("language", manifest.get("language", "en"))
        options = SimpleNamespace(model=args.model, base_url=args.base_url, temperature=0,
                                  seed=args.seed, tokens=1024, language=language)
        result = run_one(options, path, "transcription")
        identity_matches = result["sha256"] == clip["sha256"]
        response = result["response"]
        text = response.get("text") if isinstance(response, dict) else None
        chunks = response.get("tensorfold", {}).get("chunks", []) if isinstance(response, dict) else []
        stopped = all(chunk.get("finish_reason") == "stop" for chunk in chunks) if chunks else None
        valid = result["status"] == 200 and isinstance(text, str) and identity_matches and stopped is not False
        if args.require_natural_stop and stopped is not True:
            valid = False
        reference = normalize(clip["reference"])
        hypothesis = normalize(text) if valid else None
        return {"id": clip["id"], "language": language, "identity_matches": identity_matches,
                "natural_stop_observed": stopped,
                "reference": clip["reference"], "normalized_reference": reference,
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
    by_language = {}
    for language in sorted({r["language"] for r in records}):
        rows = [r for r in records if r["language"] == language]
        complete = all(r["valid"] for r in rows)
        word_count = sum(r["reference_words"] for r in rows)
        char_count = sum(r["reference_characters"] for r in rows)
        by_language[language] = dict(utterances=len(rows), complete=complete,
                                    wer=sum(r["word_edits"] for r in rows) / word_count if complete and word_count else None,
                                    cer=sum(r["character_edits"] for r in rows) / char_count if complete and char_count else None,
                                    primary="wer" if language == "en" else "cer")
    result = {"manifest": manifest, "model": args.model, "concurrency": args.concurrency,
              "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "seed": args.seed, "temperature": 0, "max_completion_tokens": 1024,
              "requires_observed_natural_stop": args.require_natural_stop,
              "normalization": r"NFKC, casefold, re.sub(r'[^\w\s]', '', text), collapse whitespace; CER excludes spaces",
              "records": records, "complete": valid, "by_language": by_language,
              "wer": sum(r["word_edits"] for r in records) / words if valid and words else None,
              "cer": sum(r["character_edits"] for r in records) / characters if valid and characters else None}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("complete", "wer", "cer")}))
    return int(not valid)


if __name__ == "__main__":
    raise SystemExit(main())
