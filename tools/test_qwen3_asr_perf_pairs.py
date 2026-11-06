"""Reject invalid work types even when both paired reports contain the same error."""
import copy
import unittest

from compare_qwen3_asr_perf import paired


def fixture():
    return dict(complete=True, warmup_work_matches=True,
                runtime=dict(context=8192, model_revision="revision", dtype="bfloat16", cache_condition="disabled"),
                workload=dict(kind="natural", mode="chat", temperature=0, seed=0, output_cap=1024, concurrency=8),
                requests=[dict(sha256=f"{i:064x}", status=200,
                               response=dict(choices=[dict(token_ids=[1, 2], finish_reason="stop",
                                                           message=dict(content="speech"))],
                                             usage=dict(completion_tokens=2))) for i in range(64)])


class Pairs(unittest.TestCase):
    def test_valid(self):
        report = fixture()
        self.assertTrue(paired(report, copy.deepcopy(report))["passed"])

    def test_common_boolean_count(self):
        report = fixture()
        for row in report["requests"]:
            row["response"]["usage"]["completion_tokens"] = True
            row["response"]["choices"][0]["token_ids"] = [1]
        with self.assertRaises(ValueError):
            paired(report, copy.deepcopy(report))

    def test_common_zero_work(self):
        report = fixture()
        for row in report["requests"]:
            row["response"]["usage"]["completion_tokens"] = 0
            row["response"]["choices"][0]["token_ids"] = []
        with self.assertRaises(ValueError):
            paired(report, copy.deepcopy(report))

    def test_common_invalid_token_identity(self):
        for value in (-1, True, 1.0):
            report = fixture()
            report["requests"][0]["response"]["choices"][0]["token_ids"][0] = value
            with self.assertRaises(ValueError):
                paired(report, copy.deepcopy(report))


if __name__ == "__main__":
    unittest.main()
