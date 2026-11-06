"""Reject invalid or truncated evidence in cancellation identity probes."""
import copy
import unittest

from check_qwen3_asr_seed_cancellation import fingerprint


def response():
    return dict(status=200, response=dict(usage=dict(completion_tokens=128), choices=[dict(
        message=dict(content="speech"), finish_reason="length")], tensorfold=dict(token_ids=[1] * 128)))


class CancellationEvidence(unittest.TestCase):
    def test_actual_capped_probe(self):
        self.assertIsNotNone(fingerprint(response()))

    def test_invalid_ids_counts_and_layouts(self):
        for token in [False, -1, "1"]:
            raw = response()
            raw["response"]["tensorfold"]["token_ids"][0] = token
            self.assertIsNone(fingerprint(raw))
        raw = response()
        raw["response"]["usage"]["completion_tokens"] = 128.0
        self.assertIsNone(fingerprint(raw))
        raw = response()
        raw["response"]["choices"][0]["token_ids"] = [2] * 128
        self.assertIsNone(fingerprint(raw))

    def test_uncapped_or_failed_probe(self):
        for change in ["finish", "status", "message"]:
            raw = copy.deepcopy(response())
            if change == "finish":
                raw["response"]["choices"][0]["finish_reason"] = "stop"
            elif change == "status":
                raw["status"] = 500
            else:
                raw["response"]["choices"][0]["message"] = None
            self.assertIsNone(fingerprint(raw))


if __name__ == "__main__":
    unittest.main()
