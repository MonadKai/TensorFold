#!/usr/bin/env python3
"""Abort active and queued ASR clients, then verify sibling and recovery outputs."""

import argparse
import base64
import concurrent.futures
import json
import re
import socket
import time
import urllib.parse
import urllib.request
from pathlib import Path

from bench_qwen3_asr import request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("model")
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--lanes", type=int, default=8)
    args = parser.parse_args()
    address = urllib.parse.urlsplit(args.base_url)

    def payload(filename, hold=False):
        data = base64.b64encode((args.fixtures / filename).read_bytes()).decode()
        value = {"model": args.model, "temperature": 0, "max_tokens": 256 if hold else 1024,
                 "ignore_eos": hold, "return_token_ids": True,
                 "messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}]}]}
        return json.dumps(value).encode()

    def metrics():
        with urllib.request.urlopen(args.base_url + "/metrics", timeout=10) as response:
            return response.read().decode()

    def run(body):
        return request(args.base_url, "/v1/chat/completions", body, "application/json")

    observations = []

    def wait_for(label, predicate):
        deadline = time.monotonic() + 120
        while True:
            raw = metrics()
            counts = {name: int(re.search(r"^tensorfold:" + name + r" (\d+)$", raw, re.M).group(1))
                      for name in ("num_requests_running", "num_requests_waiting", "client_disconnections_total")}
            if predicate(counts):
                observations.append(dict(label=label, counts=counts, metrics=raw))
                return counts
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{label}: scheduler state not observed: {counts}")
            time.sleep(0.01)

    def aborted(body, label, predicate):
        with socket.create_connection((address.hostname, address.port), timeout=30) as client:
            header = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: {address.netloc}\r\n"
                      f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode()
            client.sendall(header + body)
            wait_for(label, predicate)

    short, long = payload("asr_en_16k.wav"), payload("en_65s.wav", hold=True)
    before = metrics()
    baseline_short, baseline_long = run(short), run(long)
    active_before = wait_for("before_active_cancel", lambda counts: counts["num_requests_running"] == 0
                             and counts["num_requests_waiting"] == 0)
    aborted(long, "active_abort", lambda counts: counts["num_requests_running"] >= 1)
    wait_for("active_drained", lambda counts: counts["num_requests_running"] == counts["num_requests_waiting"] == 0
             and counts["client_disconnections_total"] == active_before["client_disconnections_total"] + 1)
    recovered = run(short)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2 * args.lanes) as pool:
        pending = [pool.submit(run, long) for _ in range(2 * args.lanes)]
        counts = wait_for("saturated", lambda counts: counts["num_requests_running"] == args.lanes
                          and counts["num_requests_waiting"] >= args.lanes)
        during = metrics()
        aborted(short, "queued_abort", lambda current: current["num_requests_running"] == args.lanes
                and current["num_requests_waiting"] > counts["num_requests_waiting"])
        wait_for("queued_cancelled", lambda current: current["client_disconnections_total"]
                 == counts["client_disconnections_total"] + 1)
        siblings = [item.result() for item in pending]
    wait_for("queue_drained", lambda counts: counts["num_requests_running"] == counts["num_requests_waiting"] == 0)
    final = run(short)

    def ids(item):
        return item["response"]["tensorfold"]["token_ids"] if item["status"] == 200 else None

    result = {
        "baseline_short": baseline_short, "baseline_long": baseline_long,
        "after_active_abort": recovered, "siblings": siblings, "after_queued_abort": final,
        "metrics_before": before, "metrics_during": during, "metrics_after": metrics(),
        "observations": observations, "lanes": args.lanes,
        "hold_work_matches": baseline_long.get("response", {}).get("usage", {}).get("completion_tokens") == 256,
        "limitation": "Capped ignore_eos lifecycle diagnostics; not complete ASR or throughput measurements.",
        "recovery_equal": ids(baseline_short) == ids(recovered) == ids(final) and ids(final) is not None,
        "siblings_equal": all(ids(v) == ids(baseline_long) and ids(v) is not None for v in siblings),
    }
    result["passed"] = all(result[k] for k in ("recovery_equal", "siblings_equal", "hold_work_matches"))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("passed", "recovery_equal", "siblings_equal", "hold_work_matches")}))
    return not result["passed"]


if __name__ == "__main__":
    raise SystemExit(main())
