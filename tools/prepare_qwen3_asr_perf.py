#!/usr/bin/env python3
"""Freeze 4096 distinct real recordings from the already pinned evaluation source assets."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import random
import tarfile

from prepare_qwen3_asr_eval import aishell_rows, digest, libri_index, normalize


def select(records):
    eligible = sorted((r for r in records if 32000 <= r["frames"] <= 320000), key=lambda r: r["id"])
    if len(eligible) < 2048:
        raise ValueError("2048 real 2..20 second recordings required per language")
    return [eligible[round(i * (len(eligible) - 1) / 2047)] for i in range(2048)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        raise ValueError("the performance manifest is already frozen")
    corpus = json.loads((args.evaluation / "manifest.json").read_text())
    if corpus["mandarin"]["revision"] != "2a509ef3a7a88d3234205eaeecba39fdd9d50518":
        raise ValueError("source revision differs")
    assets = args.evaluation / "assets"
    for asset in corpus["source_assets"]:
        if digest(assets / asset["file"]) != asset["sha256"]:
            raise ValueError("source asset identity differs")
    english = select(libri_index(assets / "test-clean.tar.gz", select=False))
    paths = sorted(assets.glob("*.parquet"))
    chinese = select([{k: v for k, v in r.items() if k != "data"} for r in aishell_rows(paths)])
    members = {r["member"]: r for r in english}
    targets = {r["id"] for r in chinese}
    selected = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending = []
        with tarfile.open(assets / "test-clean.tar.gz", "r:gz") as archive:
            for member in archive:
                if member.name in members:
                    pending.append(pool.submit(normalize, members[member.name], archive.extractfile(member).read(), "en", root))
        for row in aishell_rows(paths):
            if row["id"] in targets:
                pending.append(pool.submit(normalize, row, row["data"], "zh", root))
        for i, future in enumerate(pending, 1):
            selected.append(future.result())
            if i % 256 == 0:
                print(json.dumps({"normalized": i}), flush=True)
    selected.sort(key=lambda r: (r["language"], r["id"]))
    random.Random(20261106).shuffle(selected)
    if len(selected) != 4096 or len({r["sha256"] for r in selected}) != 4096:
        raise ValueError("4096 unique waveform identities required")
    result = dict(schema="qwen3-asr-performance-pool-v1", recognition_not_yet_run=True,
                  evaluation_manifest_sha256=digest(args.evaluation / "manifest.json"),
                  english=corpus["english"], mandarin=corpus["mandarin"], source_assets=corpus["source_assets"],
                  selection="per language: filter 2..20 seconds; sort IDs; 2048 nearest-even rank quantiles; shuffle seed 20261106",
                  ffmpeg_version=corpus["ffmpeg_version"], clips=selected)
    destination = root / "manifest.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(clips=len(selected), audio_seconds=sum(r["seconds"] for r in selected),
                          manifest_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
