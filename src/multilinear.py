"""
Multilinear Extension (MLE) and Hypercube Folding Engine.
Provides evaluation and round-folding algorithms over {0, 1}^v.
"""

from typing import List
from src.field import FieldElement, BABYBEAR


class MultilinearExtension:
    """
    Represents a multilinear polynomial over v variables
    defined by its 2^v evaluations on the Boolean hypercube {0, 1}^v.
    Evaluations are stored in standard lexicographical order.
    """

    def __init__(self, evaluations: List[FieldElement], p: int = BABYBEAR):
        n = len(evaluations)
        assert n > 0 and (n & (n - 1)) == 0, "Evaluations length must be a power of 2."
        self.evaluations = evaluations
        self.num_vars = (n - 1).bit_length()
        self.p = p

    def __len__(self) -> int:
        return len(self.evaluations)

    def evaluate(self, point: List[FieldElement]) -> FieldElement:
        """
        Evaluates the MLE at an arbitrary point r in F^v using sequential linear folding.
        Runs in O(2^v) field operations.
        """
        assert len(point) == self.num_vars, (
            f"Point dimension {len(point)} must match num_vars {self.num_vars}."
        )

        current = list(self.evaluations)
        for r_i in point:
            half = len(current) // 2
            next_evals = []
            one_minus_r = FieldElement(1, self.p) - r_i

            for j in range(half):
                v0 = current[j]
                v1 = current[j + half]
                # Linear interpolation: (1 - r_i) * v0 + r_i * v1
                val = (one_minus_r * v0) + (r_i * v1)
                next_evals.append(val)

            current = next_evals

        return current[0]

    def fold(self, r: FieldElement) -> "MultilinearExtension":
        """
        Binds the most significant variable X_0 to challenge r,
        returning a new MLE over (v - 1) variables of size 2^(v-1).
        This is the core sub-routine of the Sum-Check prover.
        """
        assert self.num_vars > 0, "Cannot fold a 0-variable MLE."

        half = len(self.evaluations) // 2
        one_minus_r = FieldElement(1, self.p) - r
        folded_evals = []

        for j in range(half):
            v0 = self.evaluations[j]
            v1 = self.evaluations[j + half]
            val = (one_minus_r * v0) + (r * v1)
            folded_evals.append(val)

        return MultilinearExtension(folded_evals, self.p)


def lagrange_basis_eval(b: int, point: List[FieldElement], p: int = BABYBEAR) -> FieldElement:
    """
    Evaluates the multilinear Lagrange basis polynomial chi_b(point).
    chi_b(x) = prod_{i=0}^{v-1} [ b_i * x_i + (1 - b_i) * (1 - x_i) ]
    """
    v = len(point)
    result = FieldElement(1, p)
    one = FieldElement(1, p)

    for i in range(v):
        # Read the i-th bit from MSB to LSB: index i corresponds to bit (v - 1 - i)
        bit = (b >> (v - 1 - i)) & 1
        r_i = point[i]

        if bit == 1:
            term = r_i
        else:
            term = one - r_i

        result = result * term

    return result