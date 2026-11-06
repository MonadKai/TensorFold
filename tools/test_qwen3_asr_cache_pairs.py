"""Negative controls for repeated-content cache comparisons."""
import copy
import unittest

from compare_qwen3_asr_cache import compare


def fixture(condition):
    def rows(count, warmup=False):
        return [dict(index=0 if warmup else i, sha256=str(i % 4), status=200,
                     response=dict(choices=[dict(token_ids=[1, 2], finish_reason="stop",
                                                 message=dict(content="speech"))],
                                   usage=dict(completion_tokens=2))) for i in range(count)]
    return dict(complete=True, warmup_work_matches=True,
                runtime=dict(engine="vllm", enforce_eager=True, context=8192,
                             model_revision="revision", dtype="bfloat16", image_id="image", cache_condition=condition),
                workload=dict(kind="natural", mode="chat", temperature=0, seed=0, output_cap=1024, concurrency=8),
                requests=rows(64), warmup_requests=rows(32, True), cache_metric_deltas={},
                measurements=dict(batch_wall_seconds=2 if condition == "disabled" else 1))


class CachePairs(unittest.TestCase):
    def setUp(self):
        self.old, self.new = fixture("disabled"), fixture("enabled")

    def test_equal_work(self):
        self.assertEqual(compare(self.old, self.new)["disabled_over_enabled_makespan"], 2)

    def test_different_work_remains_visible(self):
        self.new["requests"][0]["response"]["choices"][0]["token_ids"].append(3)
        self.new["requests"][0]["response"]["usage"]["completion_tokens"] = 3
        result = compare(self.old, self.new)
        self.assertFalse(result["equal_output_work"])
        self.assertIsNone(result["disabled_over_enabled_makespan"])
        self.assertEqual(len(result["output_differences"]), 1)

    def test_order_or_warmup_failure(self):
        self.new["warmup_requests"][0]["status"] = 500
        with self.assertRaises(ValueError):
            compare(self.old, self.new)
        self.new = fixture("enabled")
        self.new["requests"][0], self.new["requests"][1] = self.new["requests"][1], self.new["requests"][0]
        with self.assertRaises(ValueError):
            compare(self.old, self.new)

    def test_no_compile_or_identity_confounds(self):
        for key, value in (("enforce_eager", False), ("image_id", "other")):
            candidate = copy.deepcopy(self.new)
            candidate["runtime"][key] = value
            with self.assertRaises(ValueError):
                compare(self.old, candidate)

    def test_common_duplicate_does_not_pass(self):
        self.old["requests"][1]["index"] = 0
        self.new["requests"][1]["index"] = 0
        with self.assertRaises(ValueError):
            compare(self.old, self.new)

    def test_truncation_or_invalid_work(self):
        self.new["requests"][0]["response"]["choices"][0]["finish_reason"] = "length"
        with self.assertRaises(ValueError):
            compare(self.old, self.new)
        self.new = fixture("enabled")
        self.new["requests"][0]["response"]["usage"]["completion_tokens"] = True
        with self.assertRaises(ValueError):
            compare(self.old, self.new)

    def test_common_invalid_ids_cannot_pass(self):
        for ids in ([False, False], [-1, -1], ["1", "2"]):
            for report in (self.old, self.new):
                report["requests"][0]["response"]["choices"][0]["token_ids"] = ids
            with self.assertRaises(ValueError):
                compare(self.old, self.new)

    def test_conflicting_token_layouts_cannot_pass(self):
        self.new["requests"][0]["response"]["tensorfold"] = dict(token_ids=[1, 3])
        with self.assertRaises(ValueError):
            compare(self.old, self.new)


if __name__ == "__main__":
    unittest.main()
