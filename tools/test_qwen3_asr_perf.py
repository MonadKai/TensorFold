"""Exercise benchmark failure accounting through a real local HTTP server."""
import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
import wave


@contextlib.contextmanager
def server(fail_first=False):
    class Handler(BaseHTTPRequestHandler):
        requests = 0

        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            Handler.requests += 1
            failed = fail_first and Handler.requests == 1
            response = {"error": "warmup failure"} if failed else {
                "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 3},
            }
            body = json.dumps(response).encode()
            self.send_response(503 if failed else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}"
    finally:
        http.shutdown()
        thread.join()
        http.server_close()


class BenchmarkAccounting(unittest.TestCase):
    def run_cell(self, url, *arguments):
        with tempfile.TemporaryDirectory(prefix="asr-benchmark-test-") as directory:
            root = Path(directory)
            clips = []
            for index in range(2):
                path = root / f"{index}.wav"
                with wave.open(str(path), "wb") as audio:
                    audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    audio.writeframes((index + 1).to_bytes(2, "little") * 3200)
                clips.append(path)
            identity = root / "identity.json"
            identity.write_text(json.dumps({"cache_condition": "disabled"}))
            output = root / "result.json"
            command = [sys.executable, str(Path(__file__).with_name("bench_qwen3_asr_perf.py")), url, "fixture",
                       *map(str, clips), "--identity", str(identity), "--cache-condition", "disabled",
                       "--output", str(output), "--concurrency", "1", *arguments]
            process = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertTrue(output.exists(), process.stderr)
            return process.returncode, json.loads(output.read_text())

    def test_failed_warmup_is_retained_and_invalidates_cell(self):
        with server(fail_first=True) as url:
            status, result = self.run_cell(url, "--warmup-requests", "1")
        self.assertEqual(status, 1)
        self.assertFalse(result["complete"])
        self.assertEqual(result["warmup_requests"][0]["status"], 503)
        self.assertEqual(result["measurements"]["successful"], 2)
        self.assertIsNone(result["measurements"]["gpu_active_seconds"])

    def test_unique_pool_exhaustion_cannot_pass_a_sustained_cell(self):
        with server() as url:
            status, result = self.run_cell(url, "--seconds", "0.05", "--min-requests", "8", "--unique-requests")
        self.assertEqual(status, 1)
        self.assertFalse(result["complete"])
        self.assertTrue(result["unique_pool_exhausted"])
        self.assertEqual(len(result["requests"]), 2)
        self.assertEqual(len({r["sha256"] for r in result["requests"]}), 2)

    def test_equal_token_mismatch_does_not_report_full_audio_throughput(self):
        with server() as url:
            status, result = self.run_cell(url, "--workload", "equal_tokens", "--tokens", "4")
        self.assertEqual(status, 1)
        self.assertFalse(result["complete"])
        self.assertFalse(result["equal_token_work_matches"])
        self.assertIsNone(result["measurements"]["audio_seconds_per_wall_second"])


if __name__ == "__main__":
    unittest.main()
