"""
Baseline LogUp Implementation using Montgomery's Batch Inversion Trick.
Includes an algebraic operation tracker to benchmark against GKR-LogUp.
"""

from typing import List, Tuple
from src.field import FieldElement, BABYBEAR


class OpCounter:
    def __init__(self):
        self.additions = 0
        self.multiplications = 0
        self.inversions = 0

    def reset(self):
        self.additions = 0
        self.multiplications = 0
        self.inversions = 0

    def summary(self) -> dict:
        return {
            "additions": self.additions,
            "multiplications": self.multiplications,
            "inversions": self.inversions,
        }


def montgomery_batch_invert(
    elements: List[FieldElement], counter: OpCounter = None
) -> List[FieldElement]:
    """
    Computes [1/d_0, 1/d_1, ..., 1/d_{N-1}] using 1 inversion and 3*(N-1) multiplications.
    """
    n = len(elements)
    if n == 0:
        return []

    p = elements[0].p

    # Phase 1: Forward prefix products
    prefixes = [FieldElement(1, p)] * n
    prefixes[0] = elements[0]

    for i in range(1, n):
        prefixes[i] = prefixes[i - 1] * elements[i]
        if counter:
            counter.multiplications += 1

    # Phase 2: Single inversion of the total product
    inv_all = prefixes[-1].inv()
    if counter:
        counter.inversions += 1

    # Phase 3: Backward pass to extract individual inverses
    inverses = [FieldElement(0, p)] * n
    for i in range(n - 1, 0, -1):
        # d_i^(-1) = inv_all * prefix_{i-1}
        inverses[i] = inv_all * prefixes[i - 1]
        # inv_all = inv_all * d_i
        inv_all = inv_all * elements[i]
        if counter:
            counter.multiplications += 2

    inverses[0] = inv_all
    return inverses


def baseline_logup_sum(
    trace: List[int], alpha: int, p: int = BABYBEAR, counter: OpCounter = None
) -> Tuple[FieldElement, dict]:
    """
    Evaluates sum_{i=0}^{N-1} 1 / (alpha + f_i) mod p using Montgomery batch inversion.
    """
    if counter:
        counter.reset()

    alpha_elem = FieldElement(alpha, p)
    denominators = []
    for f in trace:
        d = alpha_elem + FieldElement(f, p)
        denominators.append(d)
        if counter:
            counter.additions += 1

    inv_denominators = montgomery_batch_invert(denominators, counter)

    total_sum = FieldElement(0, p)
    for inv_d in inv_denominators:
        total_sum = total_sum + inv_d
        if counter:
            counter.additions += 1

    stats = counter.summary() if counter else {}
    return total_sum, stats