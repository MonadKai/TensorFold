"""Verify that compiler activity resets admission and remains visible on timeout."""
import unittest
from unittest import mock
import hashlib
import json
from pathlib import Path
import tempfile
import wave

from qwen3_asr_compiler_guard import wait_quiet, ObservationError, snapshot
import bench_qwen3_asr_perf as benchmark


class Guard(unittest.TestCase):
    def run_guard(self, activity, timeout=20):
        clock = [0]
        def sleep(seconds):
            clock[0] += seconds
        def probe():
            return [dict(pid=1, name="nvcc")] if int(clock[0]) in activity else []
        return wait_quiet(2, timeout, probe, lambda: clock[0], sleep)

    def test_idle(self):
        self.assertEqual(self.run_guard(set())[-1]["elapsed_seconds"], 2)

    def test_resets_after_activity(self):
        result = self.run_guard({1, 3})
        self.assertEqual(result[-1]["elapsed_seconds"], 6)
        self.assertEqual(sum(bool(row["processes"]) for row in result), 2)

    def test_timeout_keeps_observations(self):
        with self.assertRaises(TimeoutError) as error:
            self.run_guard(set(range(100)), 3)
        self.assertEqual(len(error.exception.args[0]), 4)

    def test_probe_error_keeps_observations(self):
        with mock.patch("time.sleep"), self.assertRaises(ObservationError) as error:
            probe = mock.Mock(side_effect=[[dict(name="ptxas")], RuntimeError("ps unavailable")])
            wait_quiet(2, 20, probe=probe, sleep=lambda seconds: None)
        self.assertEqual(error.exception.observations[0]["processes"][0]["name"], "ptxas")

    def test_foreign_cpu_compilers_are_observed_without_blocking(self):
        clock = [0]
        def sleep(seconds):
            clock[0] += seconds
        result = wait_quiet(2, 20, probe=lambda: [dict(name="nvcc", in_scope=False)],
                            now=lambda: clock[0], sleep=sleep)
        self.assertEqual(result[-1]["elapsed_seconds"], 2)
        self.assertFalse(result[0]["processes"][0]["in_scope"])

    def test_cuda_backends_and_scope(self):
        fake = mock.Mock(returncode=0, stderr="", stdout="123 ptxas\n")
        with mock.patch("qwen3_asr_compiler_guard.subprocess.run", return_value=fake) as run, \
             mock.patch.object(Path, "read_text", return_value="docker-" + "a" * 64):
            self.assertTrue(snapshot("a" * 64)[0]["in_scope"])
            self.assertFalse(snapshot("b" * 64)[0]["in_scope"])
        for name in ("ptxas", "cudafe++", "fatbinary", "nvlink"):
            self.assertIn(name, run.call_args.args[0][2].split(","))

    def benchmark_fixture(self, folder):
        clip = Path(folder) / "clip.wav"
        with wave.open(str(clip), "wb") as audio:
            audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            audio.writeframes(b"\0\0" * 160)
        identity = Path(folder) / "identity.json"
        identity.write_text('{"cache_condition":"disabled"}')
        output = Path(folder) / "result.json"
        argv = ["bench", "http://unused", "model", str(clip), "--identity", str(identity),
                "--output", str(output), "--cache-condition", "disabled", "--warmup-requests", "1",
                "--min-requests", "1", "--compiler-quiet-seconds", "5"]
        row = dict(status=200, index=0, sha256=hashlib.sha256(clip.read_bytes()).hexdigest(), seconds=.01,
                   response=dict(choices=[dict(message=dict(content="speech"), finish_reason="stop")],
                                 usage=dict(completion_tokens=1)))
        return argv, output, row

    def test_admission_failure_preserves_warmup(self):
        with tempfile.TemporaryDirectory() as folder:
            argv, output, row = self.benchmark_fixture(folder)
            with mock.patch("sys.argv", argv), mock.patch.object(benchmark, "run_one", return_value=row), \
                 mock.patch.object(benchmark, "wait_quiet", side_effect=TimeoutError([dict(processes=["nvcc"])])):
                self.assertEqual(benchmark.main(), 1)
            report = json.loads(output.read_text())
            self.assertFalse(report["complete"])
            self.assertEqual(report["warmup_requests"], [row])
            self.assertEqual(report["schema"], "asr-performance-admission-v1")

    def test_observer_error_invalidates_measured_cell(self):
        with tempfile.TemporaryDirectory() as folder:
            argv, output, row = self.benchmark_fixture(folder)
            with mock.patch("sys.argv", argv), mock.patch.object(benchmark, "run_one", return_value=row), \
                 mock.patch.object(benchmark, "wait_quiet", return_value=[]), \
                 mock.patch.object(benchmark, "compiler_snapshot", side_effect=RuntimeError("observation failed")), \
                 mock.patch.object(benchmark, "metrics", return_value={}):
                self.assertEqual(benchmark.main(), 1)
            report = json.loads(output.read_text())
            self.assertFalse(report["complete"])
            self.assertTrue(report["compiler_guard"]["observation_errors"])
            self.assertEqual(report["measurements"]["successful"], 1)


if __name__ == "__main__":
    unittest.main()
