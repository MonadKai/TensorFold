"""Measure ASR batch makespan, latency, work, resource samples and explicit cache conditions."""
import argparse
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time
import urllib.request
import wave
from types import SimpleNamespace

from bench_qwen3_asr import run_one


def percentile(values, q):
    if not values:
        return None
    xs = sorted(values)
    at = (len(xs) - 1) * q
    lo, hi = math.floor(at), math.ceil(at)
    return xs[lo] + (xs[hi] - xs[lo]) * (at - lo)


def metrics(url):
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=10) as response:
            lines = response.read().decode().splitlines()
        return {line.split()[0]: float(line.split()[-1]) for line in lines
                if not line.startswith("#") and re.search(r"(?:(?:prefix_cache|mm_cache|encoder_cache).*?(?:hits|queries|requests).*_total|generation_tokens_total|device_memory_peak_bytes|pinned_memory(?:_peak)?_bytes|asr_decode_.*|time_to_first_token_seconds_.*)(?:\{| )", line)}
    except (OSError, ValueError):
        return {}


def proc(pid):
    if pid is None:
        return None
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()
        status = Path(f"/proc/{pid}/status").read_text()
        rss = int(re.search(r"VmRSS:\s+(\d+)", status)[1]) * 1024
        import os
        return {"cpu_seconds": (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK"), "rss_bytes": rss}
    except (OSError, TypeError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("model")
    parser.add_argument("clips", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--cache-condition", required=True, choices=("disabled", "enabled", "production"))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=0)
    parser.add_argument("--min-requests", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workload", choices=("natural", "equal_tokens"), default="natural")
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--mode", choices=("chat", "transcription"), default="chat")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--extra-pid", type=int, action="append", default=[])
    args = parser.parse_args()
    if min(args.concurrency, args.reps, args.tokens) < 1:
        parser.error("concurrency, reps and tokens must be positive")
    if args.seconds < 0 or args.min_requests < 0 or args.warmup < 0:
        parser.error("seconds, min-requests and warmup cannot be negative")
    if args.workload == "equal_tokens" and args.mode != "chat":
        parser.error("equal token diagnostics require chat ignore_eos")
    identity = json.loads(args.identity.read_text())
    if identity.get("cache_condition") != args.cache_condition:
        parser.error("runtime identity must explicitly agree with the cache condition")
    fixtures = []
    for path in args.clips:
        with wave.open(str(path)) as clip:
            duration = clip.getnframes() / clip.getframerate()
        fixtures.append({"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "audio_seconds": duration})
    options = SimpleNamespace(model=args.model, base_url=args.base_url, temperature=0, seed=args.seed,
                              tokens=args.tokens, language=None, ignore_eos=args.workload == "equal_tokens")
    def valid(r):
        value = r["response"]
        if r["status"] != 200 or not isinstance(value, dict):
            return False
        if args.mode == "transcription":
            return isinstance(value.get("text"), str)
        choices = value.get("choices")
        return isinstance(choices, list) and bool(choices) and isinstance(choices[0].get("message", {}).get("content"), str)


    warmup_results = [run_one(options, clip, args.mode) for _ in range(args.warmup) for clip in args.clips]
    before = metrics(args.base_url)
    resources = []
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            value = proc(args.pid)
            if value is not None:
                resources.append(value)
            stop.wait(.1)

    monitor = threading.Thread(target=sample)
    cpu_pids = list(dict.fromkeys(([args.pid] if args.pid is not None else []) + args.extra_pid))

    def cpu_total():
        readings = [proc(pid) for pid in cpu_pids]
        return sum(r["cpu_seconds"] for r in readings) if readings and all(r is not None for r in readings) else None

    initial_cpu = cpu_total()
    monitor.start()
    start = time.monotonic()
    results = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            if args.seconds or args.min_requests:
                issued = 0
                issue_lock = threading.Lock()

                def worker():
                    nonlocal issued
                    rows = []
                    while True:
                        with issue_lock:
                            if time.monotonic() - start >= args.seconds and issued >= max(args.min_requests, args.reps * len(args.clips)):
                                break
                            index = issued
                            issued += 1
                        rows.append(run_one(options, args.clips[index % len(args.clips)], args.mode, index))
                    return rows

                jobs = [pool.submit(worker) for _ in range(args.concurrency)]
                for job in jobs:
                    results.extend(job.result())
                results.sort(key=lambda r: r["index"])
            else:
                jobs = [pool.submit(run_one, options, path, args.mode, i) for i in range(args.reps) for path in args.clips]
                for job in jobs:
                    results.append(job.result())
    finally:
        elapsed = time.monotonic() - start
        stop.set()
        monitor.join()
    final_cpu = cpu_total()
    after = metrics(args.base_url)

    successful = [r for r in results if valid(r)]
    durations = {f["sha256"]: f["audio_seconds"] for f in fixtures}
    attempted_audio = sum(durations[r["sha256"]] for r in results)
    audio = sum(durations[r["sha256"]] for r in successful)
    token_counters = [after[key] - before[key] for key in after if key in before and "generation_tokens_total" in key]
    tokens = sum(r["response"].get("usage", {}).get("completion_tokens", 0) for r in successful) if args.mode == "chat" else (sum(token_counters) if token_counters and len(successful) == len(results) else None)
    cpu = final_cpu - initial_cpu if initial_cpu is not None and final_cpu is not None else None
    intervals = [r["response"].get("tensorfold", {}).get("audio_gpu_event_intervals") for r in successful]
    ttft = [r["response"].get("tensorfold", {}).get("time_to_first_token") for r in successful]
    ttft = [x for x in ttft if isinstance(x, (int, float))]
    work_matches = args.workload != "equal_tokens" or all(r["response"].get("usage", {}).get("completion_tokens") == args.tokens for r in successful)
    def natural_stop(r):
        value = r["response"]
        if args.mode == "chat":
            return value["choices"][0].get("finish_reason") == "stop"
        chunks = value.get("tensorfold", {}).get("chunks", [])
        return all(x.get("finish_reason") == "stop" for x in chunks) if chunks else None

    stops = [natural_stop(r) for r in successful] if args.workload == "natural" else []
    full_audio = args.workload == "natural" and bool(stops) and all(x is True for x in stops)
    result = {
        "schema": "asr-performance-v1", "runtime": identity, "model": args.model,
        "workload": {"fixtures": fixtures, "concurrency": args.concurrency, "reps": args.reps, "warmup_excluded": args.warmup,
                     "mode": args.mode, "temperature": 0, "seed": args.seed, "output_cap": args.tokens,
                     "kind": args.workload, "minimum_seconds": args.seconds, "minimum_requests": args.min_requests,
                     "arrival_model": "closed loop; request latency begins at each HTTP submission"},
        "measurements": {"batch_wall_seconds": elapsed, "requests": len(results), "successful": len(successful),
                         "latency_p50_seconds": percentile([r["seconds"] for r in successful], .5),
                         "latency_p95_seconds": percentile([r["seconds"] for r in successful], .95),
                         "ttft_p50_seconds": percentile(ttft, .5), "ttft_p95_seconds": percentile(ttft, .95),
                         "ttft_observations": len(ttft),
                         "attempted_audio_seconds": attempted_audio, "successful_audio_seconds": audio,
                         "audio_seconds_per_wall_second": audio / elapsed if full_audio else None,
                         "successful_requests_per_second": len(successful) / elapsed, "generated_tokens": tokens,
                         "generated_tokens_per_wall_second": tokens / elapsed if tokens is not None else None,
                         "cpu_seconds": cpu, "cpu_core_equivalents": cpu / elapsed if cpu is not None else None,
                         "sampled_peak_process_rss_bytes": max((x["rss_bytes"] for x in resources), default=None),
                         "device_allocation_peak_bytes": next((after[key] for key in after if "device_memory_peak_bytes" in key), None),
                         "pinned_allocation_bytes_after": next((after[key] for key in after if "pinned_memory_bytes" in key), None),
                         "pinned_allocation_peak_bytes": next((after[key] for key in after if "pinned_memory_peak_bytes" in key), None), "gpu_active_seconds": None},
        "cache_metric_deltas": {key: after[key] - before[key] for key in after if key in before and any(name in key for name in ("prefix_cache", "mm_cache", "encoder_cache"))},
        "gpu_intervals": intervals, "requests": results, "warmup_requests": warmup_results,
        "complete": bool(results) and len(successful) == len(results) and all(valid(r) for r in warmup_results) and work_matches and not any(x is False for x in stops),
        "natural_stop_observations": stops,
        "equal_token_work_matches": work_matches if args.workload == "equal_tokens" else None,
        "decode_batch_metric_deltas": {key: after[key] - before[key] for key in after if key in before and "asr_decode_" in key},
        "ttft_metric_deltas": {key: after[key] - before[key] for key in after if key in before and "time_to_first_token_seconds_" in key},
        "cpu_processes": cpu_pids,
        "generation_token_source": "response usage" if args.mode == "chat" else "isolated server counter delta, includes chunk control tokens",
        "limitations": ["CUDA event intervals are elapsed stream intervals, not device active time.",
                        "Shared-batch GPU intervals are charged whole to each participating request; do not sum them.",
                        "CPU includes explicitly selected PIDs only; RSS is the primary PID alone, not aggregate physical memory.",
                        "Missing resource counters remain null; zeros are not substituted.",
                        "Transcription token counters require an isolated server with no unrelated requests.",
                        "Device allocation peak, when exported, is the server lifetime peak including startup and warmup."]}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["measurements"]))
    return int(not result["complete"])


if __name__ == "__main__":
    raise SystemExit(main())
