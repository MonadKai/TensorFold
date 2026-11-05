#!/usr/bin/env python3
"""Compare all fixed seeded probes while queued and decoding neighbors are actually cancelled."""
import argparse
import base64
from collections import defaultdict
import concurrent.futures
import contextlib
import hashlib
import json
from pathlib import Path
import socket
import threading
import time
from types import SimpleNamespace
import urllib.parse
import urllib.request

from bench_qwen3_asr import run_one
from check_qwen3_asr_determinism import fingerprint
from compare_qwen3_asr_quality import FROZEN_MANIFEST_SHA256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("model")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("hold_audio", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    if not reference["passed"] or reference["manifest_sha256"] != FROZEN_MANIFEST_SHA256 or hashlib.sha256(args.manifest.read_bytes()).hexdigest() != FROZEN_MANIFEST_SHA256:
        raise ValueError("valid matching deterministic reference required")
    by_id = {row["id"]: row for row in json.loads(args.manifest.read_text())["clips"]}
    selected = [by_id[key] for key in reference["selected_ids"]]
    if len(selected) != 64 or len({row["id"] for row in selected}) != 64:
        raise ValueError("all 64 fixed seeded probes required")
    groups = defaultdict(list)
    for clip in by_id.values():
        groups[(clip["language"], clip["speaker"])].append(clip)
    prescribed = []
    for key in sorted(groups):
        clips = sorted(groups[key], key=lambda row: (row["seconds"], row["id"]))
        prescribed += [clips[0]["id"], clips[7]["id"]]
    if reference["selected_ids"] != prescribed:
        raise ValueError("probe selection differs from the frozen duration ranks")
    baseline = reference["baselines"]["sampled"]
    address = urllib.parse.urlsplit(args.base_url)
    body = json.dumps(dict(model=args.model, temperature=0, seed=0, max_tokens=1024, ignore_eos=True,
                           messages=[dict(role="user", content=[dict(type="input_audio", input_audio=dict(
                               data=base64.b64encode(args.hold_audio.read_bytes()).decode(), format="wav"))])])).encode()
    observations, results, mismatches = [], [], []

    def metrics():
        with urllib.request.urlopen(args.base_url + "/metrics", timeout=10) as response:
            raw = response.read().decode()
        return {line.split()[0]: float(line.split()[-1]) for line in raw.splitlines() if line and not line.startswith("#")}

    def count(values, name):
        return values["tensorfold:" + name]

    def decode8(values):
        return sum(n for key, n in values.items() if key.startswith("tensorfold:asr_decode_batches_total{")
                   and 'mode="throughput",rows="8"' in key)

    def wait(label, predicate):
        deadline = time.monotonic() + 120
        while True:
            values = metrics()
            if predicate(values):
                observations.append(dict(label=label, metrics=values))
                return values
            if time.monotonic() >= deadline:
                raise TimeoutError(label)
            time.sleep(0.005)

    @contextlib.contextmanager
    def held():
        with socket.create_connection((address.hostname, address.port), timeout=30) as client:
            client.sendall((f"POST /v1/chat/completions HTTP/1.1\r\nHost: {address.netloc}\r\n"
                            f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
                            "Connection: close\r\n\r\n").encode() + body)
            yield

    passed = False
    failure = None
    try:
        for first in range(0, 64, 8):
            before = wait(f"wave{first//8}_drained", lambda m: count(m, "num_requests_running") == 0
                          and count(m, "num_requests_waiting") == 0)
            with contextlib.ExitStack() as active:
                for _ in range(8):
                    active.enter_context(held())
                wait(f"wave{first//8}_eight_decoding", lambda m: count(m, "num_requests_running") == 8
                     and decode8(m) > decode8(before))
                barrier = threading.Barrier(9)

                def one(clip):
                    barrier.wait(timeout=30)
                    options = SimpleNamespace(base_url=args.base_url, model=args.model, temperature=0.7,
                                              seed=314159, top_k=50, top_p=0.9, min_p=0, tokens=128,
                                              language=clip["language"])
                    raw = run_one(options, args.manifest.parent / clip["file"], "transcription")
                    return dict(id=clip["id"], raw=raw, fingerprint=fingerprint(raw)
                                if raw["sha256"] == clip["sha256"] else None)

                pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
                try:
                    jobs = [pool.submit(one, clip) for clip in selected[first:first+8]]
                    barrier.wait(timeout=30)
                    wait(f"wave{first//8}_probes_waiting", lambda m: count(m, "num_requests_running") == 8
                         and count(m, "num_requests_waiting") >= 4)
                    queued_before = metrics()
                    with held():
                        wait(f"wave{first//8}_neighbor_queued", lambda m: count(m, "num_requests_running") == 8
                             and count(m, "num_requests_waiting") > count(queued_before, "num_requests_waiting"))
                        time.sleep(2)
                    wait(f"wave{first//8}_queued_cancelled", lambda m: count(m, "client_disconnections_total")
                         == count(before, "client_disconnections_total") + 1)
                    active.close()
                    wait(f"wave{first//8}_eight_active_cancelled", lambda m: count(m, "client_disconnections_total")
                         == count(before, "client_disconnections_total") + 9)
                    for job in jobs:
                        row = job.result()
                        results.append(row)
                        if row["fingerprint"] is None or row["fingerprint"] != baseline[row["id"]]:
                            mismatches.append(row["id"])
                finally:
                    active.close()
                    pool.shutdown(wait=True)
        wait("all_probes_released", lambda m: count(m, "num_requests_running") == count(m, "num_requests_waiting") == 0)
        passed = len(results) == 64 and not mismatches
    except Exception as exc:
        failure = dict(type=type(exc).__name__, detail=str(exc))
    args.output.write_text(json.dumps(dict(schema=1, passed=passed, failure=failure, observations=observations,
                                         manifest_sha256=reference["manifest_sha256"], settings=reference["settings"],
                                         selected_ids=reference["selected_ids"], responses=results, mismatches=mismatches,
                                         limitation="Capped seeded identity diagnostics; not complete ASR or throughput."),
                                    ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(passed=passed, responses=len(results), mismatches=mismatches, failure=failure)))
    return not passed


if __name__ == "__main__":
    raise SystemExit(main())
