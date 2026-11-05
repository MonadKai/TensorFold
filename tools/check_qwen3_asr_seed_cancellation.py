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

    def decodes(values, rows):
        return sum(n for key, n in values.items() if key.startswith("tensorfold:asr_decode_batches_total{")
                   and f'mode="throughput",rows="{rows}"' in key)

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
            yield client

    def fingerprint(raw):
        if raw["status"] != 200:
            return None
        value = raw["response"]
        ids = value.get("tensorfold", {}).get("token_ids")
        if not isinstance(ids, list) or len(ids) != 128 or value.get("usage", {}).get("completion_tokens") != 128:
            return None
        return dict(text=value["choices"][0]["message"]["content"], token_ids=ids,
                    finish_reason=value["choices"][0]["finish_reason"])

    def one(clip, mode, barrier=None):
        if barrier is not None:
            barrier.wait(timeout=30)
        options = SimpleNamespace(base_url=args.base_url, model=args.model, temperature=0 if mode == "greedy" else 0.7,
                                  seed=0 if mode == "greedy" else 314159, top_k=50, top_p=0.9, min_p=0,
                                  tokens=128, ignore_eos=True, language=None)
        raw = run_one(options, args.manifest.parent / clip["file"], "chat")
        return dict(id=clip["id"], raw=raw, fingerprint=fingerprint(raw)
                    if raw["sha256"] == clip["sha256"] else None)

    passed = False
    failure = None
    baselines = {}
    try:
        ordered = sorted(selected, key=lambda row: (row["seconds"], row["id"]))
        groups = [ordered[first:first+7] for first in range(0, 64, 7)]
        for mode in ("greedy", "sampled"):
            rows = [one(clip, mode) for clip in selected]
            baselines[mode] = rows
            baseline = {row["id"]: row["fingerprint"] for row in rows}
            if any(value is None for value in baseline.values()):
                raise ValueError("baseline failed or lacks equal token work")
            for schedule in range(3):
                waves = groups if schedule == 0 else list(reversed(groups)) if schedule == 1 else groups[3:] + groups[:3]
                for wave, clips in enumerate(waves):
                    clips = clips if schedule == 0 else list(reversed(clips)) if schedule == 1 else clips[3:] + clips[:3]
                    size = len(clips)
                    label = f"{mode}_s{schedule}_wave{wave}"
                    before = wait(label+"_drained", lambda m: count(m, "num_requests_running") == 0
                                  and count(m, "num_requests_waiting") == 0)
                    with contextlib.ExitStack() as active:
                        sockets = [active.enter_context(held()) for _ in range(8)]
                        wait(label+"_eight_decoding", lambda m: count(m, "num_requests_running") == 8
                             and decodes(m, 8) > decodes(before, 8))
                        barrier = threading.Barrier(size + 1)
                        pool = concurrent.futures.ThreadPoolExecutor(max_workers=size)
                        try:
                            jobs = [pool.submit(one, clip, mode, barrier) for clip in clips]
                            barrier.wait(timeout=30)
                            wait(label+"_all_probes_queued", lambda m: count(m, "num_requests_running") == 8
                                 and count(m, "num_requests_waiting") == size)
                            with held():
                                wait(label+"_neighbor_queued", lambda m: count(m, "num_requests_running") == 8
                                     and count(m, "num_requests_waiting") == size + 1)
                            wait(label+"_queued_cancelled", lambda m: count(m, "client_disconnections_total")
                                 == count(before, "client_disconnections_total") + 1)
                            for client in sockets[:7]:
                                client.close()
                            released = wait(label+"_seven_holders_cancelled", lambda m: count(m, "client_disconnections_total")
                                            == count(before, "client_disconnections_total") + 8)
                            running = wait(label+"_all_probes_decoding_with_neighbor", lambda m:
                                           count(m, "num_requests_running") == size + 1
                                           and decodes(m, size + 1) > decodes(released, size + 1))
                            sockets[7].close()
                            cancelled = wait(label+"_decoding_neighbor_cancelled", lambda m: count(m, "client_disconnections_total")
                                             == count(before, "client_disconnections_total") + 9)
                            wait(label+"_all_probes_decode_after_cancel", lambda m: count(m, "num_requests_running") == size
                                 and decodes(m, size) > decodes(cancelled, size))
                            for job in jobs:
                                row = job.result()
                                row.update(mode=mode, schedule=schedule, wave=wave)
                                results.append(row)
                                if row["fingerprint"] is None or row["fingerprint"] != baseline[row["id"]]:
                                    mismatches.append(dict(mode=mode, schedule=schedule, id=row["id"]))
                        finally:
                            active.close()
                            pool.shutdown(wait=True)
        wait("all_probes_released", lambda m: count(m, "num_requests_running") == count(m, "num_requests_waiting") == 0)
        passed = len(results) == 384 and not mismatches
    except Exception as exc:
        failure = dict(type=type(exc).__name__, detail=str(exc))
    args.output.write_text(json.dumps(dict(schema=1, passed=passed, failure=failure, observations=observations,
                                         manifest_sha256=reference["manifest_sha256"], baselines=baselines,
                                         settings=dict(context=8192, output_cap=128, ignore_eos=True, input_mode="chat",
                                                       language="auto in both baseline and disturbed probes", greedy_seed=0,
                                                       sampled_seed=314159, sampled_temperature=0.7, top_k=50, top_p=0.9, min_p=0),
                                         schedules="duration buckets of seven; original, reverse, rotate3 waves and rows",
                                         selected_ids=reference["selected_ids"], responses=results, mismatches=mismatches,
                                         limitation="Capped seeded identity diagnostics; not complete ASR or throughput."),
                                    ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(passed=passed, responses=len(results), mismatches=mismatches, failure=failure)))
    return not passed


if __name__ == "__main__":
    raise SystemExit(main())
