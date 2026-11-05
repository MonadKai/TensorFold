"""Reject incomplete or forged paired ASR receipts before scoring quality."""
import copy
import hashlib
import json
import unittest
from unittest.mock import patch

from compare_qwen3_asr_quality import validate_report


def fixture():
    clips = [dict(id=f"{language}-{speaker}-{rank}", language=language, speaker=str(speaker),
                  reference="hello", sha256="a" * 64)
             for language in ("en", "zh") for speaker in range(16) for rank in range(8)]
    manifest = dict(schema="qwen3-asr-bilingual-eval-v1", clips=clips,
                    mandarin=dict(revision="2a509ef3a7a88d3234205eaeecba39fdd9d50518"))
    sha = hashlib.sha256((json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()).hexdigest()
    records = [dict(id=c["id"], language=c["language"], reference="hello", valid=True, identity_matches=True,
                    normalized_reference="hello", normalized_hypothesis="hello", natural_stop_observed=True,
                    reference_words=1, reference_characters=5, word_edits=0, character_edits=0,
                    raw=dict(sha256=c["sha256"], status=200,
                             response=dict(text="hello", tensorfold=dict(chunks=[dict(finish_reason="stop")]))))
               for c in clips]
    return dict(manifest=manifest, manifest_sha256=sha, records=records,
                complete=True, requires_observed_natural_stop=True), sha


class QualityIntegrity(unittest.TestCase):
    def setUp(self):
        self.report, sha = fixture()
        self.guard = patch("compare_qwen3_asr_quality.FROZEN_MANIFEST_SHA256", sha)
        self.guard.start()
        self.addCleanup(self.guard.stop)

    def test_complete_receipt(self):
        validate_report(self.report)

    def test_missing_or_duplicated_utterance(self):
        for duplicate in (False, True):
            report = copy.deepcopy(self.report)
            report["records"].pop()
            if duplicate:
                report["records"].append(copy.deepcopy(report["records"][0]))
            with self.assertRaisesRegex(ValueError, "uniquely and exactly"):
                validate_report(report)

    def test_modified_manifest_cannot_keep_its_identity(self):
        self.report["manifest"]["clips"][0]["reference"] = "changed"
        with self.assertRaisesRegex(ValueError, "frozen manifest"):
            validate_report(self.report)

    def test_raw_proof_and_counts_are_checked(self):
        for name, value in (("word_edits", -1), ("reference_characters", 0),
                            ("natural_stop_observed", False), ("identity_matches", False)):
            report = copy.deepcopy(self.report)
            report["records"][0][name] = value
            with self.assertRaises(ValueError, msg=name):
                validate_report(report)
        self.report["records"][0]["raw"]["response"]["tensorfold"]["chunks"][0]["finish_reason"] = "length"
        with self.assertRaisesRegex(ValueError, "natural stopping"):
            validate_report(self.report)


if __name__ == "__main__":
    unittest.main()
