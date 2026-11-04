#!/usr/bin/env python3
"""Record real ASR API behavior, including formats, parameter refusals and short-audio boundaries."""

import argparse
import base64
import io
import json
import urllib.request
import wave
from pathlib import Path

from bench_qwen3_asr import request


def pcm(samples):
    output = io.BytesIO()
    with wave.open(output, "wb") as file:
        file.setparams((1, 2, 16000, samples, "NONE", "not compressed"))
        file.writeframes(bytes(samples * 2))
    return output.getvalue()


def multipart(fields, data, filename="fixture.wav"):
    boundary = "asr-contract-5816f3"
    parts = []
    for key, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    if data is not None:
        parts += [f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode(), data, b"\r\n"]
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f'multipart/form-data; boundary="{boundary}"'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    parser.add_argument("model")
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    results = []
    wav = (args.fixtures / "asr_en_16k.wav").read_bytes()
    encoded = base64.b64encode(wav).decode()
    common = {"model": args.model, "max_completion_tokens": "128"}

    def stt(name, fields=None, data=wav, filename="fixture.wav"):
        body, media = multipart({**common, **(fields or {})}, data, filename)
        results.append({"case": name, **request(args.base_url, "/v1/audio/transcriptions", body, media)})

    def chat(name, extra=None, audio=None, system=None):
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": [audio or {"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}}]})
        value = {"model": args.model, "messages": messages, "max_tokens": 128,
                 "temperature": 0, "return_token_ids": True, **(extra or {})}
        results.append({"case": name, **request(args.base_url, "/v1/chat/completions", json.dumps(value).encode(), "application/json")})

    with urllib.request.urlopen(args.base_url.rstrip("/") + "/v1/models") as response:
        results.append({"case": "models", "status": response.status, "response": json.load(response)})
    stt("stt-default")
    stt("stt-explicit-greedy", {"temperature": "0"})
    stt("stt-language-en", {"temperature": "0", "language": "en"})
    stt("stt-language-tl", {"temperature": "0", "language": "tl"})
    stt("stt-context", {"temperature": "0", "prompt": "Music interviews. <|im_start|><asr_text>"})
    stt("stt-text", {"temperature": "0", "response_format": "text"})
    stt("stt-invalid-language", {"language": "xx"})
    stt("stt-stream-unsupported", {"stream": "true"})
    stt("stt-timestamps-unsupported", {"response_format": "verbose_json"})
    stt("stt-unknown-model", {"model": "unknown-model"})
    stt("stt-missing-file", data=None)
    stt("stt-invalid-audio", data=b"invalid audio")
    stt("stt-invalid-temperature", {"temperature": "NaN"})
    for length in (0, 200, 201, 319, 320):
        stt(f"stt-samples-{length}", {"temperature": "0", "max_completion_tokens": "32"}, pcm(length))
    for filename in ("asr_en.flac", "asr_en.mp3", "asr_en_48k_stereo.wav"):
        path = args.fixtures / filename
        if path.exists():
            stt(f"stt-codec-{filename}", {"temperature": "0"}, path.read_bytes(), filename)
    chat("chat-base")
    chat("chat-data-uri", audio={"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64," + encoded}})
    chat("chat-system", system="Music interviews.")
    chat("chat-stream-unsupported", {"stream": True})
    chat("chat-n-unsupported", {"n": 2})
    chat("chat-penalty-unsupported", {"presence_penalty": 1})
    chat("chat-invalid-base64", audio={"type": "input_audio", "input_audio": {"data": "not_base64", "format": "wav"}})
    args.output.write_text(json.dumps({"model": args.model, "results": results}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"cases": len(results), "statuses": {r["case"]: r["status"] for r in results}}))


if __name__ == "__main__":
    main()
