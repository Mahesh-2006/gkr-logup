import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.field import FieldElement, BABYBEAR
from src.baseline_logup import montgomery_batch_invert, baseline_logup_sum, OpCounter


class TestBaselineLogup(unittest.TestCase):
    def test_montgomery_inversion_identity(self):
        p = BABYBEAR
        raw_values = [3, 7, 11, 45, 99, 1001, 876543]
        elements = [FieldElement(x, p) for x in raw_values]

        inverses = montgomery_batch_invert(elements)

        for orig, inv in zip(elements, inverses):
            self.assertEqual((orig * inv).value, 1)

    def test_operation_counts(self):
        p = 19
        counter = OpCounter()
        trace = [2, 5, 7, 1]  # N = 4
        alpha = 3

        _, stats = baseline_logup_sum(trace, alpha, p=p, counter=counter)

        # For N = 4 elements:
        # Inversions: 1
        # Multiplications: 3 * (4 - 1) = 9
        self.assertEqual(stats["inversions"], 1)
        self.assertEqual(stats["multiplications"], 9)

    def test_sum_matches_naive(self):
        p = 19
        trace = [1, 2, 3, 4]
        alpha = 5

        batch_res, _ = baseline_logup_sum(trace, alpha, p=p)

        naive_sum = FieldElement(0, p)
        for val in trace:
            d = FieldElement(alpha + val, p)
            naive_sum = naive_sum + d.inv()

        self.assertEqual(batch_res.value, naive_sum.value)


if __name__ == "__main__":
    unittest.main()