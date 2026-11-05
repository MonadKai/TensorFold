"""Check the precision envelope at BF16 binade boundaries and at zero."""
import unittest
import numpy as np
from verify_qwen3_asr_gemm_accuracy import envelope


class ProjectionBound(unittest.TestCase):
    def test_bf16_half_ulp_and_subnormal_floor(self):
        ref = np.array([1.0, -2.0, 0.0], dtype=np.float64)
        strict, existing, uncertainty = envelope(ref, np.zeros(3), 0)
        np.testing.assert_array_equal(strict, [2.0 ** -8, 2.0 ** -7, 2.0 ** -134])
        self.assertTrue(np.all(existing >= strict))
        self.assertTrue(np.all(uncertainty == 0))

    def test_depth_and_sumabs_increase_both_bounds(self):
        reference = np.array([1.0])
        short = envelope(reference, np.array([8.0]), 2048)
        long = envelope(reference, np.array([8.0]), 6144)
        self.assertGreater(long[0][0], short[0][0])
        self.assertGreater(long[1][0], short[1][0])
        self.assertGreater(long[2][0], short[2][0])


if __name__ == "__main__":
    unittest.main()
