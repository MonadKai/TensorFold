#!/usr/bin/env python3
"""Abort active and queued ASR clients, then verify sibling and recovery outputs."""

import argparse
import base64
import concurrent.futures
import json
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
    args = parser.parse_args()
    address = urllib.parse.urlsplit(args.base_url)

    def payload(filename):
        data = base64.b64encode((args.fixtures / filename).read_bytes()).decode()
        value = {"model": args.model, "temperature": 0, "max_tokens": 1024, "return_token_ids": True,
                 "messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}]}]}
        return json.dumps(value).encode()

    def metrics():
        with urllib.request.urlopen(args.base_url + "/metrics", timeout=10) as response:
            return response.read().decode()

    def run(body):
        return request(args.base_url, "/v1/chat/completions", body, "application/json")

    def aborted(body, delay):
        with socket.create_connection((address.hostname, address.port), timeout=30) as client:
            header = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: {address.netloc}\r\n"
                      f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode()
            client.sendall(header + body)
            time.sleep(delay)

    short, long = payload("asr_en_16k.wav"), payload("en_65s.wav")
    before = metrics()
    baseline_short, baseline_long = run(short), run(long)
    aborted(long, 0.3)
    recovered = run(short)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(run, long) for _ in range(4)]
        time.sleep(0.5)
        during = metrics()
        aborted(short, 0.05)
        siblings = [item.result() for item in pending]
    final = run(short)

    def ids(item):
        return item["response"]["tensorfold"]["token_ids"] if item["status"] == 200 else None

    result = {
        "baseline_short": baseline_short, "baseline_long": baseline_long,
        "after_active_abort": recovered, "siblings": siblings, "after_queued_abort": final,
        "metrics_before": before, "metrics_during": during, "metrics_after": metrics(),
        "recovery_equal": ids(baseline_short) == ids(recovered) == ids(final) and ids(final) is not None,
        "siblings_equal": all(ids(v) == ids(baseline_long) and ids(v) is not None for v in siblings),
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("recovery_equal", "siblings_equal")}))
    return not (result["recovery_equal"] and result["siblings_equal"])


if __name__ == "__main__":
    raise SystemExit(main())
