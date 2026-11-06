"""Do not compare scheduler latency when workload, chunk order or warmup evidence differs."""
import copy
import unittest

from compare_qwen3_asr_schedule import compare


def chat(index):
    return dict(index=index, sha256=f"{index % 2:064x}", status=200,
                response=dict(choices=[dict(message=dict(content="short"), finish_reason="stop")],
                              tensorfold=dict(token_ids=[1, 2]), usage=dict(completion_tokens=2)))


def mixed():
    long = [dict(raw=dict(index=i, sha256=f"{1000+i:064x}", status=200, response=dict(text="long",
            tensorfold=dict(chunks=[dict(token_sha256=f"{i:064x}", completion_tokens=2, finish_reason="stop")]))))
            for i in range(2)]
    warmup = [chat(i) for i in range(2)]
    for row in warmup:
        row["index"] = 0
    return dict(schema=1, complete=True, live_overlap_observed=True, short_identity_equal=True,
                settings=dict(short_requests=64, short_concurrency=4, context=8192, temperature=0,
                              seed=0, output_cap=1024, long_arrivals_seconds=[0, 0.5], short_start_seconds=0.25),
                long=long, short=[dict(raw=chat(i)) for i in range(64)], warmup=warmup)


class SchedulePairs(unittest.TestCase):
    def test_same_fixed_work_passes(self):
        value = mixed()
        self.assertTrue(compare(value, value)["passed"])

    def test_common_missing_requests_and_reordered_inputs_do_not_pass(self):
        old = mixed()
        old["short"].pop()
        with self.assertRaises(ValueError):
            compare(old, old)
        old, new = mixed(), mixed()
        new["long"].reverse()
        with self.assertRaises(ValueError):
            compare(old, new)

    def test_chunk_work_and_text_differences_are_preserved(self):
        for change in ["tokens", "text"]:
            old, new = mixed(), mixed()
            if change == "tokens":
                new["long"][0]["raw"]["response"]["tensorfold"]["chunks"][0]["completion_tokens"] = 3
            else:
                new["long"][0]["raw"]["response"]["text"] = "different"
            result = compare(old, new)
            self.assertFalse(result["passed"])
            self.assertEqual(len(result["differences"]), 1)

    def test_failed_warmup_cannot_be_hidden_by_complete_flag(self):
        old, new = mixed(), mixed()
        new["warmup"][0]["status"] = 500
        self.assertFalse(compare(old, new)["passed"])

    def test_common_missing_or_duplicate_ordinals_are_rejected(self):
        for change in ["missing", "duplicate", "noninteger"]:
            value = mixed()
            if change == "missing":
                for row in value["short"]:
                    row["raw"].pop("index")
            elif change == "duplicate":
                value["short"][1] = copy.deepcopy(value["short"][0])
            else:
                value["short"][0]["raw"]["index"] = False
            with self.assertRaises(ValueError):
                compare(value, value)

    def test_invalid_chunk_work_cannot_match_itself(self):
        for tokens in [-1, 0, True, 2.0]:
            value = mixed()
            value["long"][0]["raw"]["response"]["tensorfold"]["chunks"][0]["completion_tokens"] = tokens
            self.assertFalse(compare(value, value)["passed"])


if __name__ == "__main__":
    unittest.main()
