"""Reject mismatched workloads and preserve native/cross-engine token disagreements."""
import copy
import unittest
from compare_qwen3_asr_perf import paired


def report():
    return dict(complete=True, warmup_work_matches=True,
                workload=dict(kind="natural", mode="chat", temperature=0, seed=0, output_cap=1024, concurrency=8),
                runtime=dict(context=8192, model_revision="fixed", dtype="bfloat16"),
                requests=[dict(sha256=f"{i:064x}", status=200, response=dict(usage=dict(completion_tokens=2),
                               choices=[dict(message=dict(content="hello"), token_ids=[1, 2], finish_reason="stop")]))
                          for i in range(64)])


class PairedWork(unittest.TestCase):
    def test_native_token_disagreement_fails_but_cross_engine_keeps_it(self):
        old, new = report(), report()
        new["requests"][0]["response"]["choices"][0]["token_ids"] = [1, 3]
        self.assertFalse(paired(old, new)["passed"])
        cross = paired(old, new, cross_engine=True)
        self.assertTrue(cross["passed"])
        self.assertEqual(len(cross["differing_waveforms"]), 1)

    def test_common_missing_inputs_and_duplicates_cannot_pass(self):
        for duplicate in (True, False):
            value = report()
            value["requests"].pop()
            if duplicate:
                value["requests"].append(copy.deepcopy(value["requests"][0]))
            with self.assertRaises(ValueError):
                paired(value, value)

    def test_missing_token_work_or_truncation_cannot_pass(self):
        for missing in (True, False):
            old, new = report(), report()
            if missing:
                new["requests"][0]["response"].pop("usage")
            else:
                new["requests"][0]["response"]["choices"][0]["finish_reason"] = "length"
            with self.assertRaises(ValueError):
                paired(old, new, cross_engine=True)

    def test_sampling_mismatch_is_not_an_ablation(self):
        old, new = report(), report()
        new["workload"]["seed"] = 7
        with self.assertRaisesRegex(ValueError, "seed"):
            paired(old, new)


if __name__ == "__main__":
    unittest.main()
