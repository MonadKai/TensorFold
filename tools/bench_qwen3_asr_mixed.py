#!/usr/bin/env python3
"""Measure fixed native long/short ASR arrivals with observed natural stopping and live admission."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import urllib.request

from bench_qwen3_asr import run_one
from bench_qwen3_asr_perf import percentile


def complete_response(raw):
    if raw["status"] != 200 or not isinstance(raw["response"], dict):
        return False
    if raw["mode"] == "transcription":
        chunks = raw["response"].get("tensorfold", {}).get("chunks", [])
        return isinstance(raw["response"].get("text"), str) and bool(chunks) and all(
            chunk.get("finish_reason") == "stop" for chunk in chunks)
    choices = raw["response"].get("choices", [])
    return bool(choices) and choices[0].get("finish_reason") == "stop"


def overlap_observed(snapshots):
    return any(0 < snap.get("values", {}).get("tensorfold:audio_chunks_active", 0)
               < snap["values"].get("tensorfold:num_requests_running", 0) for snap in snapshots)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("model")
    parser.add_argument("--long", nargs=2, type=Path, required=True)
    parser.add_argument("--short", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--short-requests", type=int, default=64)
    parser.add_argument("--short-concurrency", type=int, default=4)
    args = parser.parse_args()
    if min(args.short_requests, args.short_concurrency) < 1:
        parser.error("request count and concurrency must be positive")
    options = SimpleNamespace(base_url=args.base_url, model=args.model, temperature=0, seed=0, tokens=1024, language="en")
    warmup = [run_one(options, path, "chat") for path in args.short]
    start = time.monotonic()
    snapshots = []
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                with urllib.request.urlopen(args.base_url + "/metrics", timeout=10) as response:
                    raw = response.read().decode()
                values = {line.split()[0]: float(line.split()[-1]) for line in raw.splitlines()
                          if line and not line.startswith("#") and any(s in line for s in
                          ("audio_chunks_", "audio_parent_active_max", "num_requests_", "asr_decode_batches_total"))}
                snapshots.append(dict(seconds=time.monotonic() - start, values=values))
            except (OSError, ValueError) as exc:
                snapshots.append(dict(seconds=time.monotonic() - start, error=type(exc).__name__))
            stop.wait(0.05)

    def one(path, mode, index, delay=0):
        if delay:
            time.sleep(delay)
        submitted = time.monotonic() - start
        result = run_one(options, path, mode, index)
        return dict(submitted_seconds=submitted, completed_seconds=time.monotonic() - start, raw=result)

    thread = threading.Thread(target=monitor)
    thread.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as parents:
            long_jobs = [parents.submit(one, path, "transcription", i, 0.5 * i) for i, path in enumerate(args.long)]
            time.sleep(0.25)
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.short_concurrency) as ordinary:
                short_jobs = [ordinary.submit(one, args.short[i % len(args.short)], "chat", i) for i in range(args.short_requests)]
                short = [job.result() for job in short_jobs]
            long = [job.result() for job in long_jobs]
    finally:
        elapsed = time.monotonic() - start
        stop.set()
        thread.join()

    def identity(raw):
        value = raw["response"]
        return dict(text=value["choices"][0]["message"]["content"], tokens=value.get("tensorfold", {}).get("token_ids"))

    warmup_ok = all(complete_response(row) for row in warmup)
    success = all(complete_response(row["raw"]) for row in short + long)
    equal = warmup_ok and all(complete_response(row["raw"]) and identity(row["raw"]) == identity(warmup[i % len(warmup)])
                             for i, row in enumerate(short))
    end_long = max(row["completed_seconds"] for row in long)
    overlapped = [row for row in short if row["submitted_seconds"] < end_long]
    latencies = [row["raw"]["seconds"] for row in short if complete_response(row["raw"])]
    overlap_latencies = [row["raw"]["seconds"] for row in overlapped if complete_response(row["raw"])]
    queued = [row["raw"]["response"].get("tensorfold", {}).get("queued_seconds") for row in short if complete_response(row["raw"])]
    queued = [n for n in queued if isinstance(n, (int, float))]
    overlap = overlap_observed(snapshots)
    result = dict(schema=1, complete=warmup_ok and success and equal and overlap, warmup=warmup, short=short, long=long,
                  live_overlap_observed=overlap,
                  short_identity_equal=equal, observations=snapshots,
                  settings=dict(short_requests=args.short_requests, short_concurrency=args.short_concurrency,
                                long_arrivals_seconds=[0, 0.5], short_start_seconds=0.25, context=8192,
                                temperature=0, seed=0, output_cap=1024),
                  measurements=dict(makespan_seconds=elapsed, short_p50_seconds=percentile(latencies, .5),
                                    short_p95_seconds=percentile(latencies, .95), short_max_seconds=max(latencies, default=None),
                                    overlap_short_requests=len(overlapped), overlap_short_p95_seconds=percentile(overlap_latencies, .95),
                                    observed_queue_max_seconds=max(queued, default=None),
                                    long_completion_seconds=[row["completed_seconds"] for row in long]),
                  limitations=["Makespan includes the fixed arrival delays; short latency begins at HTTP submission.",
                               "Observed queue maximum is not an unbounded-load waiting-time guarantee.",
                               "Native chunk stopping and admission telemetry are required for this mixed-load gate."])
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(complete=result["complete"], **result["measurements"])))
    return not result["complete"]


if __name__ == "__main__":
    raise SystemExit(main())
