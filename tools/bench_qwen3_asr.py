#!/usr/bin/env python3
"""Exercise the real non-streaming ASR APIs and retain comparable responses."""

import argparse
import base64
import concurrent.futures
import hashlib
import http.client
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def request(base_url, route, body, content_type):
    req = urllib.request.Request(
        base_url.rstrip("/") + route,
        data=body,
        headers={"Content-Type": content_type},
    )
    start = time.monotonic()
    try:
        try:
            with urllib.request.urlopen(req, timeout=600) as response:
                status, data = response.status, response.read()
                media = response.headers.get("Content-Type")
        except urllib.error.HTTPError as exc:
            status, data, media = exc.code, exc.read(), exc.headers.get("Content-Type")
    except (OSError, TimeoutError, urllib.error.URLError, http.client.HTTPException) as exc:
        return {"status": 0, "content_type": None, "seconds": time.monotonic() - start,
                "response": {"transport_error": type(exc).__name__, "detail": str(exc)}}
    elapsed = time.monotonic() - start
    try:
        value = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        value = data.decode("utf-8", errors="replace")
    return {"status": status, "content_type": media, "seconds": elapsed, "response": value}


def run_one(args, path, mode, index=0):
    data = path.read_bytes()
    common = {"model": args.model, "temperature": args.temperature, "max_completion_tokens": args.tokens}
    if args.seed is not None:
        common["seed"] = args.seed
    if mode == "chat":
        common.update(
            stream=False,
            return_token_ids=True,
            messages=[{"role": "user", "content": [
                {"type": "input_audio", "input_audio": {
                    "data": base64.b64encode(data).decode(), "format": "wav"
                }}
            ]}],
        )
        result = request(args.base_url, "/v1/chat/completions", json.dumps(common).encode(), "application/json")
    else:
        boundary = "tensorfold-asr-fixture-5f50a04b"
        fields = {**common, "stream": "false", "response_format": "json"}
        if args.language:
            fields["language"] = args.language
        parts = []
        for name, value in fields.items():
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
        parts += [
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="fixture.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode(),
            data, f"\r\n--{boundary}--\r\n".encode(),
        ]
        result = request(args.base_url, "/v1/audio/transcriptions", b"".join(parts), f"multipart/form-data; boundary={boundary}")
    result.update(fixture=path.name, sha256=hashlib.sha256(data).hexdigest(), mode=mode, index=index)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("model")
    parser.add_argument("clips", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--language")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--modes", nargs="+", choices=("chat", "transcription"), default=("chat", "transcription"))
    args = parser.parse_args()
    result = {"model": args.model, "tokens": args.tokens, "language": args.language, "solo": [], "concurrent": []}
    for path in args.clips:
        for mode in args.modes:
            result["solo"].append(run_one(args, path, mode))
    jobs = [(path, mode, i) for i in range(args.concurrency) for path in args.clips for mode in args.modes]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = [pool.submit(run_one, args, *job) for job in jobs]
        for future in futures:
            result["concurrent"].append(future.result())
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    failed = sum(r["status"] != 200 for k in ("solo", "concurrent") for r in result[k])
    print(json.dumps({"requests": len(result["solo"]) + len(result["concurrent"]), "failed": failed}))
    return bool(failed)


if __name__ == "__main__":
    raise SystemExit(main())
