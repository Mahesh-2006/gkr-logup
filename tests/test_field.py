import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.field import FieldElement, BABYBEAR


class TestFieldElement(unittest.TestCase):
    def setUp(self):
        self.p_toy = 19
        self.p_bb = BABYBEAR

    def test_basic_arithmetic(self):
        a = FieldElement(7, self.p_toy)
        b = FieldElement(15, self.p_toy)

        self.assertEqual(a + b, FieldElement(3, self.p_toy))
        self.assertEqual(a - b, FieldElement(11, self.p_toy))
        self.assertEqual(a * b, FieldElement(10, self.p_toy))

    def test_inversion_over_f19(self):
        for x in range(1, self.p_toy):
            elem = FieldElement(x, self.p_toy)
            inv_elem = elem.inv()
            self.assertEqual((elem * inv_elem).value, 1)

    def test_division(self):
        a = FieldElement(8, self.p_toy)
        b = FieldElement(3, self.p_toy)
        quotient = a / b
        self.assertEqual(quotient * b, a)

    def test_zero_inversion_raises_error(self):
        zero = FieldElement(0, self.p_toy)
        with self.assertRaises(ZeroDivisionError):
            zero.inv()

    def test_babybear_arithmetic(self):
        x = FieldElement(123456789, self.p_bb)
        x_inv = x.inv()
        self.assertEqual((x * x_inv).value, 1)


if __name__ == "__main__":
    unittest.main()