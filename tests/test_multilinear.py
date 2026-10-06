import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.field import FieldElement, BABYBEAR
from src.multilinear import MultilinearExtension, lagrange_basis_eval


class TestMultilinearExtension(unittest.TestCase):
    def setUp(self):
        self.p = BABYBEAR

    def test_hypercube_boolean_evaluations(self):
        # Define MLE with 4 evaluations over {0, 1}^2
        # f(0,0)=3, f(0,1)=7, f(1,0)=11, f(1,1)=15
        evals = [FieldElement(x, self.p) for x in [3, 7, 11, 15]]
        mle = MultilinearExtension(evals, self.p)

        # Checking boolean inputs matches direct lookup
        self.assertEqual(mle.evaluate([FieldElement(0, self.p), FieldElement(0, self.p)]), FieldElement(3, self.p))
        self.assertEqual(mle.evaluate([FieldElement(0, self.p), FieldElement(1, self.p)]), FieldElement(7, self.p))
        self.assertEqual(mle.evaluate([FieldElement(1, self.p), FieldElement(0, self.p)]), FieldElement(11, self.p))
        self.assertEqual(mle.evaluate([FieldElement(1, self.p), FieldElement(1, self.p)]), FieldElement(15, self.p))

    def test_folding_equivalence(self):
        evals = [FieldElement(x, self.p) for x in [2, 4, 6, 8, 10, 12, 14, 16]]
        mle = MultilinearExtension(evals, self.p)  # 3 variables

        r0 = FieldElement(5, self.p)
        r1 = FieldElement(7, self.p)
        r2 = FieldElement(11, self.p)

        # 1. Direct evaluation at (r0, r1, r2)
        direct_val = mle.evaluate([r0, r1, r2])

        # 2. Sequential folding
        folded_1 = mle.fold(r0)
        folded_2 = folded_1.fold(r1)
        folded_3 = folded_2.fold(r2)

        self.assertEqual(direct_val, folded_3.evaluations[0])

    def test_lagrange_basis_identity(self):
        evals = [FieldElement(x, self.p) for x in [5, 12, 19, 26]]
        mle = MultilinearExtension(evals, self.p)

        point = [FieldElement(9, self.p), FieldElement(13, self.p)]
        eval_folded = mle.evaluate(point)

        # Sum_{b in {0,1}^2} f(b) * chi_b(point)
        eval_lagrange = FieldElement(0, self.p)
        for b in range(4):
            chi = lagrange_basis_eval(b, point, self.p)
            eval_lagrange = eval_lagrange + (evals[b] * chi)

        self.assertEqual(eval_folded, eval_lagrange)


if __name__ == "__main__":
    unittest.main()
    