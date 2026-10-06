import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.field import FieldElement, BABYBEAR
from src.baseline_logup import OpCounter, baseline_logup_sum
from src.gkr_tree import build_gkr_tree, verify_lookup_roots


class TestGKRTree(unittest.TestCase):
    def setUp(self):
        self.p = BABYBEAR
        self.alpha = 42

    def test_tree_matches_baseline_sum(self):
        trace = [3, 7, 11, 15, 19, 23, 27, 31]  # N = 8
        alpha_elem = FieldElement(self.alpha, self.p)

        leaf_p = [FieldElement(1, self.p) for _ in trace]
        leaf_q = [alpha_elem + FieldElement(val, self.p) for val in trace]

        tree = build_gkr_tree(leaf_p, leaf_q, p=self.p)
        p_root, q_root = tree.root

        # Single inversion at the very end to check equivalence
        tree_sum = p_root / q_root
        baseline_sum, _ = baseline_logup_sum(trace, self.alpha, p=self.p)

        self.assertEqual(tree_sum, baseline_sum)

    def test_zero_inversions_in_tree(self):
        counter = OpCounter()
        trace = [1, 2, 3, 4, 5, 6, 7, 8]  # N = 8
        alpha_elem = FieldElement(self.alpha, self.p)

        leaf_p = [FieldElement(1, self.p) for _ in trace]
        leaf_q = [alpha_elem + FieldElement(val, self.p) for val in trace]

        build_gkr_tree(leaf_p, leaf_q, counter=counter, p=self.p)

        # Inversions must be strictly zero inside the tree
        self.assertEqual(counter.inversions, 0)
        # For N = 8, there are 7 internal node merges: 7 * 3 = 21 mults
        self.assertEqual(counter.multiplications, 21)
        self.assertEqual(counter.additions, 7)

    def test_valid_lookup_table_equality(self):
        # Trace has multiset values
        trace = [5, 2, 5, 9]
        # Table has unique values: [2, 5, 9] with multiplicities [1, 2, 1]
        table = [2, 5, 9]
        mults = [1, 2, 1]

        alpha_elem = FieldElement(self.alpha, self.p)

        # Build Trace Tree
        trace_p = [FieldElement(1, self.p) for _ in trace]
        trace_q = [alpha_elem + FieldElement(x, self.p) for x in trace]
        trace_tree = build_gkr_tree(trace_p, trace_q, p=self.p)

        # Build Table Tree
        table_p = [FieldElement(m, self.p) for m in mults]
        table_q = [alpha_elem + FieldElement(t, self.p) for t in table]
        table_tree = build_gkr_tree(table_p, table_q, p=self.p)

        # Verify cross-multiplication equality at the root
        self.assertTrue(verify_lookup_roots(trace_tree.root, table_tree.root))

    def test_invalid_lookup_fails(self):
        trace = [5, 2, 5, 999]  # 999 is not in table
        table = [2, 5, 9]
        mults = [1, 2, 1]

        alpha_elem = FieldElement(self.alpha, self.p)

        trace_p = [FieldElement(1, self.p) for _ in trace]
        trace_q = [alpha_elem + FieldElement(x, self.p) for x in trace]
        trace_tree = build_gkr_tree(trace_p, trace_q, p=self.p)

        table_p = [FieldElement(m, self.p) for m in mults]
        table_q = [alpha_elem + FieldElement(t, self.p) for t in table]
        table_tree = build_gkr_tree(table_p, table_q, p=self.p)

        self.assertFalse(verify_lookup_roots(trace_tree.root, table_tree.root))


if __name__ == "__main__":
    unittest.main()