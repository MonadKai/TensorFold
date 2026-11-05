#!/usr/bin/env python3
"""Freeze a deterministic bilingual, multi-speaker ASR evaluation set before recognition."""
import argparse
from collections import defaultdict
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import time
import urllib.request
import wave

REVISION = "2a509ef3a7a88d3234205eaeecba39fdd9d50518"
REPOSITORY = "TwinkStart/AISHELL-1"
LIBRI_MD5 = "32fa31d27d2e1cad72775fee3f4849a9"
LIBRI_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"


def digest(path, algorithm="sha256"):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def download(url, path, size=None, sha256=None):
    if path.exists() and (size is None or path.stat().st_size == size):
        if sha256 is None or digest(path) == sha256:
            return
    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                append = offset > 0 and response.status == 206
                with partial.open("ab" if append else "wb") as target:
                    shutil.copyfileobj(response, target, 1 << 20)
            if size is not None and partial.stat().st_size != size:
                raise ValueError("download length does not match fixed metadata")
            if sha256 is not None and digest(partial) != sha256:
                partial.unlink()
                raise ValueError("download SHA256 does not match fixed metadata")
            partial.replace(path)
            print(json.dumps({"downloaded": path.name, "bytes": path.stat().st_size}), flush=True)
            return
        except Exception as error:
            print(json.dumps({"download": path.name, "attempt": attempt + 1,
                              "error_type": type(error).__name__}), flush=True)
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def audio_info(data):
    if data[:4] == b"fLaC":
        if data[4] & 127 or int.from_bytes(data[5:8], "big") != 34:
            raise ValueError("FLAC STREAMINFO must be first")
        packed = int.from_bytes(data[18:26], "big")
        return packed & ((1 << 36) - 1), packed >> 44, ((packed >> 41) & 7) + 1
    with wave.open(io.BytesIO(data)) as stream:
        return stream.getnframes(), stream.getframerate(), stream.getnchannels()


def pick(records):
    groups = defaultdict(list)
    for record in records:
        if record["rate"] != 16000 or record["channels"] != 1 or record["frames"] <= 200:
            raise ValueError("unexpected source audio geometry")
        groups[record["speaker"]].append(record)
    speakers = sorted(groups, key=lambda x: int(x.lstrip("S")))[:16]
    if len(speakers) != 16:
        raise ValueError("sixteen test speakers required")
    selected = []
    for speaker in speakers:
        rows = sorted(groups[speaker], key=lambda r: (r["frames"], r["id"]))
        if len(rows) < 8:
            raise ValueError("eight distinct utterances per speaker required")
        indices = [round(i * (len(rows) - 1) / 7) for i in range(8)]
        if len(set(indices)) != 8:
            raise ValueError("selection indices must be unique")
        selected.extend(rows[i] for i in indices)
    return selected


def libri_index(path):
    records, labels = [], {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if member.name.endswith(".trans.txt"):
                for line in archive.extractfile(member).read().decode().splitlines():
                    identity, text = line.split(" ", 1)
                    labels[identity] = text
            elif member.name.endswith(".flac"):
                identity = Path(member.name).stem
                if not re.fullmatch(r"\d+-\d+-\d+", identity):
                    raise ValueError("unexpected LibriSpeech identity")
                frames, rate, channels = audio_info(archive.extractfile(member).read(42))
                records.append({"id": identity, "speaker": identity.split("-")[0], "frames": frames,
                                "rate": rate, "channels": channels, "member": member.name})
    selected = pick(records)
    for row in selected:
        row["reference"] = labels[row["id"]]
    return selected


def aishell_rows(paths):
    import pyarrow.parquet as pq
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                identity = row["name"]
                match = re.fullmatch(r"BAC009(S\d+)W\d+", identity)
                if not match:
                    raise ValueError("unexpected AISHELL identity")
                data = row["audio"]["bytes"]
                frames, rate, channels = audio_info(data)
                yield {"id": identity, "speaker": match[1], "frames": frames, "rate": rate,
                       "channels": channels, "reference": row["text"], "data": data,
                       "parquet": path.name, "source_path": row["WavPath"]}


def normalize(row, data, language, root):
    directory = root / language
    directory.mkdir(exist_ok=True)
    extension = ".flac" if data[:4] == b"fLaC" else ".wav"
    original = directory / (row["id"] + ".source" + extension)
    original.write_bytes(data)
    target = directory / (row["id"] + ".wav")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(original),
                    "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target)], check=True)
    with wave.open(str(target)) as stream:
        frames = stream.getnframes()
        if (stream.getnchannels(), stream.getframerate(), stream.getsampwidth()) != (1, 16000, 2):
            raise ValueError("normalization geometry differs")
    if frames != row["frames"]:
        raise ValueError("lossless normalization changed sample count")
    return {"id": row["id"], "speaker": row["speaker"], "language": language,
            "file": str(target.relative_to(root)), "sha256": digest(target),
            "source_sha256": hashlib.sha256(data).hexdigest(), "seconds": frames / 16000,
            "reference": row["reference"],
            "reference_sha256": hashlib.sha256(row["reference"].encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        parser.error("a frozen manifest already exists; choose a new task directory")
    assets = root / "assets"
    assets.mkdir(exist_ok=True)
    tree = get_json(f"https://huggingface.co/api/datasets/{REPOSITORY}/tree/{REVISION}/data?expand=true")
    files = [row for row in tree if row["path"].endswith(".parquet")]
    if len(files) != 3:
        raise ValueError("expected three pinned AISHELL test shards")
    jobs = [(LIBRI_URL, assets / "test-clean.tar.gz", None, None)]
    for row in files:
        sha256 = row["lfs"]["oid"]
        jobs.append((f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{row['path']}",
                     assets / Path(row["path"]).name, row["size"], sha256))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda args: download(*args), jobs))
    if digest(assets / "test-clean.tar.gz", "md5") != LIBRI_MD5:
        raise ValueError("LibriSpeech official archive checksum differs")
    paths = sorted(assets.glob("*.parquet"))
    english = libri_index(assets / "test-clean.tar.gz")
    chinese = pick([{k: v for k, v in row.items() if k != "data"} for row in aishell_rows(paths)])
    selected = []
    members = {row["member"]: row for row in english}
    with tarfile.open(assets / "test-clean.tar.gz", "r:gz") as archive:
        for member in archive:
            if member.name in members:
                selected.append(normalize(members[member.name], archive.extractfile(member).read(), "en", root))
    by_id = {row["id"]: row for row in chinese}
    for row in aishell_rows(paths):
        if row["id"] in by_id:
            selected.append(normalize(row, row["data"], "zh", root))
    selected.sort(key=lambda r: (r["language"], int(r["speaker"].lstrip("S")), r["seconds"], r["id"]))
    if len(selected) != 256 or len({r["id"] for r in selected}) != 256:
        raise ValueError("the frozen sample must contain 256 unique utterances")
    manifest = {"schema": "qwen3-asr-bilingual-eval-v1", "recognition_not_yet_run": True,
                "selection": "first16 numeric test speakers/language; eight nearest-even duration quantiles/speaker",
                "english": {"dataset": "OpenSLR12 LibriSpeech test-clean", "url": LIBRI_URL,
                            "official_md5": LIBRI_MD5, "license": "CC BY 4.0"},
                "mandarin": {"dataset": REPOSITORY, "revision": REVISION, "split": "test",
                             "original": "OpenSLR33 AISHELL-1", "license": "Apache-2.0"},
                "source_assets": [{"file": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
                                  for path in sorted(assets.iterdir()) if path.is_file() and not path.name.endswith(".part")],
                "ffmpeg_version": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
                "clips": selected}
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    for language in ["en", "zh"]:
        subset = {**manifest, "language": language, "clips": [r for r in selected if r["language"] == language]}
        (root / (language + ".json")).write_text(json.dumps(subset, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"clips": len(selected), "speakers": len({(r['language'], r['speaker']) for r in selected}),
                      "audio_seconds": sum(r["seconds"] for r in selected),
                      "manifest_sha256": digest(root / "manifest.json")}), flush=True)


if __name__ == "__main__":
    main()
