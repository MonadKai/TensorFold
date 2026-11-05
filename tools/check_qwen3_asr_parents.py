#!/usr/bin/env python3
"""Observe occupied child leases during quota shrink, cancellation and ordered long-audio completion."""
import argparse
import concurrent.futures
import contextlib
import json
from pathlib import Path
import re
import socket
import time
from types import SimpleNamespace
import urllib.parse
import urllib.request

from bench_qwen3_asr import run_one
from check_qwen3_asr_determinism import fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("model")
    parser.add_argument("short", type=Path)
    parser.add_argument("long", nargs=2, type=Path, help="real and repeated long audio, each with at least four chunks")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    options = SimpleNamespace(base_url=args.base_url, model=args.model, temperature=0, seed=0, tokens=1024, language="en")
    sampled = SimpleNamespace(**(vars(options) | dict(temperature=0.7, seed=314159, top_k=50, top_p=0.9, min_p=0)))
    snapshots = []
    address = urllib.parse.urlsplit(args.base_url)

    def metrics():
        with urllib.request.urlopen(args.base_url + "/metrics", timeout=10) as response:
            raw = response.read().decode()
        return {line.split()[0]: float(line.split()[-1]) for line in raw.splitlines() if line and not line.startswith("#")}

    def wait(label, predicate, timeout=120):
        started = time.monotonic()
        while True:
            values = metrics()
            if predicate(values):
                snapshots.append(dict(label=label, seconds=time.monotonic() - started, metrics=values))
                return values
            if time.monotonic() - started >= timeout:
                raise TimeoutError(label + ": required live scheduler state not observed")
            time.sleep(0.005)

    def value(values, name):
        return values["tensorfold:" + name]

    def drained(values):
        return all(value(values, name) == 0 for name in
                   ("audio_chunks_active", "audio_chunks_waiting", "audio_chunks_parents",
                    "num_requests_running", "num_requests_waiting"))

    def rows4(values):
        return sum(n for key, n in values.items() if key.startswith("tensorfold:asr_decode_batches_total{")
                   and 'mode="throughput"' in key and re.search(r'(?<!physical_)rows="4"', key))

    @contextlib.contextmanager
    def parent(path):
        boundary = "asr-parent-cancellation-9cbf7a12"
        body = b"".join(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{val}\r\n'.encode()
                        for key, val in dict(model=args.model, language="en", temperature=0, seed=0,
                                             max_completion_tokens=1024).items())
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
                 f'Content-Type: audio/wav\r\n\r\n').encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        with socket.create_connection((address.hostname, address.port), timeout=60) as client:
            header = (f"POST /v1/audio/transcriptions HTTP/1.1\r\nHost: {address.netloc}\r\n"
                      f"Content-Type: multipart/form-data; boundary={boundary}\r\n"
                      f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode()
            client.sendall(header + body)
            yield

    result = dict(schema=1, passed=False, observations=snapshots,
                  limitation="Live admission plus distinct-slot decode evidence; counters alone are not KV byte measurements.")
    try:
        wait("initially_drained", drained)
        baseline = run_one(sampled, args.short, "transcription")
        with parent(args.long[0]):
            first = wait("single_parent_four_children", lambda m: value(m, "audio_chunks_parents") == 1
                         and value(m, "audio_parent_active_max") == 4 and value(m, "num_requests_running") >= 4)
            with parent(args.long[1]):
                shrink = wait("two_parents_existing_leases_retained", lambda m: value(m, "audio_chunks_parents") == 2
                              and value(m, "audio_parent_active_max") > 2 and value(m, "audio_chunks_waiting") > 0)
                work = wait("four_distinct_slots_still_decode", lambda m: rows4(m) > rows4(shrink)
                            and value(m, "audio_chunks_parents") == 2 and value(m, "audio_parent_active_max") > 2)
                sibling = run_one(sampled, args.short, "transcription")
                before_queued_cancel = metrics()
            cancelled = wait("queued_parent_cancelled", lambda m: value(m, "audio_chunks_parents") == 1
                             and value(m, "audio_parents_cancelled_total")
                             == value(before_queued_cancel, "audio_parents_cancelled_total") + 1)
        wait("all_children_released", lambda m: drained(m) and value(m, "audio_parents_cancelled_total")
             == value(cancelled, "audio_parents_cancelled_total") + 1)
        recovery = run_one(sampled, args.short, "transcription")
        result.update(short_baseline=baseline, short_sibling=sibling, short_recovery=recovery,
                      cancellation_seed_equal=fingerprint(baseline) is not None
                      and fingerprint(baseline) == fingerprint(sibling) == fingerprint(recovery),
                      occupied_after_shrink=dict(before_rows4=rows4(shrink), after_rows4=rows4(work),
                                                 observed_max_parent_active=value(shrink, "audio_parent_active_max")))
        solo = [run_one(options, path, "transcription") for path in args.long]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            mixed = list(pool.map(lambda path: run_one(options, path, "transcription"), args.long))
        ordered = all(fingerprint(a) is not None and fingerprint(a) == fingerprint(b) for a, b in zip(solo, mixed))
        natural = all(r["status"] == 200 and len(r["response"].get("tensorfold", {}).get("chunks", [])) >= 4
                      and all(c.get("finish_reason") == "stop" for c in r["response"]["tensorfold"]["chunks"])
                      for r in solo + mixed)
        wait("completed_parents_released", drained)
        result.update(solo=solo, concurrent=mixed, ordered_results_equal=ordered, natural_stop=natural,
                      passed=ordered and natural and result["cancellation_seed_equal"])
    except Exception as exc:
        result["failure"] = dict(type=type(exc).__name__, detail=str(exc))
    finally:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result.get(key) for key in ("passed", "failure", "ordered_results_equal", "cancellation_seed_equal")}))
    return not result["passed"]


if __name__ == "__main__":
    raise SystemExit(main())
