"""
GKR Layered Fraction Tree Accumulator.
Aggregates rational fractions into a binary circuit without modular inversions.
"""

from typing import List, Tuple, Optional
from src.field import FieldElement, BABYBEAR
from src.baseline_logup import OpCounter


class TreeLayer:
    """Represents a single layer of the GKR arithmetic circuit."""
    __slots__ = ("numerators", "denominators")

    def __init__(self, numerators: List[FieldElement], denominators: List[FieldElement]):
        assert len(numerators) == len(denominators), "Numerator and denominator sizes must match."
        self.numerators = numerators
        self.denominators = denominators

    def __len__(self) -> int:
        return len(self.numerators)


class GKRTree:
    """
    Balanced binary circuit tree of rational functions.
    Layer d = inputs (leaves)
    Layer 0 = output (root)
    """

    def __init__(self, layers: List[TreeLayer], counter: Optional[OpCounter] = None):
        self.layers = layers
        self.depth = len(layers) - 1
        self.counter = counter

    @property
    def root(self) -> Tuple[FieldElement, FieldElement]:
        """Returns the single accumulated fraction (P_root, Q_root)."""
        return self.layers[0].numerators[0], self.layers[0].denominators[0]


def pad_to_power_of_two(
    numerators: List[FieldElement],
    denominators: List[FieldElement],
    p: int = BABYBEAR
) -> Tuple[List[FieldElement], List[FieldElement]]:
    """Pads leaves with dummy 0/1 fractions until length is a power of 2."""
    n = len(numerators)
    target_len = 1 << (n - 1).bit_length() if n > 0 else 1

    padded_p = list(numerators)
    padded_q = list(denominators)

    while len(padded_p) < target_len:
        # Dummy fraction: 0 / 1 adds nothing to the rational sum
        padded_p.append(FieldElement(0, p))
        padded_q.append(FieldElement(1, p))

    return padded_p, padded_q


def build_gkr_tree(
    leaf_numerators: List[FieldElement],
    leaf_denominators: List[FieldElement],
    counter: Optional[OpCounter] = None,
    p: int = BABYBEAR
) -> GKRTree:
    """
    Builds the complete GKR circuit tree from leaf fractions.
    Total operations for N = 2^d leaves:
      - 3 * (N - 1) multiplications
      - 1 * (N - 1) additions
      - 0 inversions
    """
    p_leaves, q_leaves = pad_to_power_of_two(leaf_numerators, leaf_denominators, p)

    # Circuit layers: index 0 will be the root, index d will be leaves
    all_layers: List[TreeLayer] = []
    current_layer = TreeLayer(p_leaves, q_leaves)
    all_layers.append(current_layer)

    # Accumulate upwards until reaching a single root node
    while len(current_layer) > 1:
        next_p: List[FieldElement] = []
        next_q: List[FieldElement] = []

        for i in range(0, len(current_layer), 2):
            p_L = current_layer.numerators[i]
            q_L = current_layer.denominators[i]
            p_R = current_layer.numerators[i + 1]
            q_R = current_layer.denominators[i + 1]

            # P_parent = p_L * q_R + p_R * q_L
            term1 = p_L * q_R
            term2 = p_R * q_L
            p_parent = term1 + term2

            # Q_parent = q_L * q_R
            q_parent = q_L * q_R

            if counter:
                counter.multiplications += 3
                counter.additions += 1

            next_p.append(p_parent)
            next_q.append(q_parent)

        current_layer = TreeLayer(next_p, next_q)
        all_layers.append(current_layer)

    # Reverse so layers[0] is Root and layers[d] is Leaves (standard GKR convention)
    all_layers.reverse()
    return GKRTree(all_layers, counter)


def verify_lookup_roots(
    trace_root: Tuple[FieldElement, FieldElement],
    table_root: Tuple[FieldElement, FieldElement],
    counter: Optional[OpCounter] = None
) -> bool:
    """
    Checks if P_trace / Q_trace == P_table / Q_table via cross-multiplication:
    P_trace * Q_table - P_table * Q_trace == 0 mod p.
    Requires 2 multiplications, 1 subtraction, and 0 divisions.
    """
    p_trace, q_trace = trace_root
    p_table, q_table = table_root

    lhs = p_trace * q_table
    rhs = p_table * q_trace
    diff = lhs - rhs

    if counter:
        counter.multiplications += 2
        counter.additions += 1

    return diff.value == 0