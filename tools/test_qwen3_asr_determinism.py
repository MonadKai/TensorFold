"""Require actual logical rows and token evidence in request determinism qualification."""
import copy
import unittest

from check_qwen3_asr_determinism import fingerprint, observed_rows


def response():
    return dict(status=200, response=dict(usage=dict(completion_tokens=2), choices=[dict(
        message=dict(content="example"), finish_reason="stop")], tensorfold=dict(token_ids=[1, 2])))


class DeterminismEvidence(unittest.TestCase):
    def test_actual_native_and_reference_token_layouts(self):
        native = response()
        reference = copy.deepcopy(native)
        reference["response"]["choices"][0]["token_ids"] = reference["response"].pop("tensorfold")["token_ids"]
        self.assertIsNotNone(fingerprint(native))
        self.assertEqual(fingerprint(native), fingerprint(reference))
        reference["response"]["tensorfold"] = dict(token_ids=[1, 3])
        self.assertIsNone(fingerprint(reference))

    def test_physical_padding_does_not_prove_logical_batch(self):
        key = 'tensorfold:asr_decode_batches_total{mode="throughput",rows="1",physical_rows="8"}'
        self.assertFalse(observed_rows({key: 10}, 8))
        self.assertTrue(observed_rows({key: 10}, 1))

    def test_target_requires_positive_exact_row_counter(self):
        key = 'tensorfold:asr_decode_batches_total{mode="throughput",rows="8",physical_rows="8"}'
        for values in [{}, {key: 0}, {key: -1}]:
            self.assertFalse(observed_rows(values, 8))
        self.assertTrue(observed_rows({key: 1}, 8))

    def test_equal_text_and_work_do_not_hide_token_changes(self):
        old = response()
        new = copy.deepcopy(old)
        new["response"]["tensorfold"]["token_ids"] = [1, 3]
        self.assertNotEqual(fingerprint(old), fingerprint(new))

    def test_missing_ids_work_or_response_cannot_pass(self):
        for change in ["ids", "usage", "count", "message", "choices"]:
            value = response()
            body = value["response"]
            if change == "ids":
                body["tensorfold"].pop("token_ids")
            elif change == "usage":
                body["usage"] = None
            elif change == "count":
                body["usage"]["completion_tokens"] = 3
            elif change == "message":
                body["choices"][0]["message"] = None
            else:
                body["choices"] = []
            self.assertIsNone(fingerprint(value))


if __name__ == "__main__":
    unittest.main()
