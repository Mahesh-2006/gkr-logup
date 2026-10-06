"""
Finite Field Arithmetic Engine
Supports arbitrary prime fields F_p, operator overloading, and modular inversions.
"""

class FieldElement:
    __slots__ = ("value", "p")

    def __init__(self, value: int, p: int):
        self.p = p
        self.value = value % p

    def __add__(self, other):
        val = other.value if isinstance(other, FieldElement) else other
        return FieldElement((self.value + val) % self.p, self.p)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        val = other.value if isinstance(other, FieldElement) else other
        return FieldElement((self.value - val) % self.p, self.p)

    def __rsub__(self, other):
        val = other.value if isinstance(other, FieldElement) else other
        return FieldElement((val - self.value) % self.p, self.p)

    def __mul__(self, other):
        val = other.value if isinstance(other, FieldElement) else other
        return FieldElement((self.value * val) % self.p, self.p)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return FieldElement((-self.value) % self.p, self.p)

    def inv(self):
        if self.value == 0:
            raise ZeroDivisionError("Cannot invert 0 in finite field.")
        # By Fermat's Little Theorem: a^(p-2) = a^(-1) (mod p)
        return FieldElement(pow(self.value, self.p - 2, self.p), self.p)

    def __truediv__(self, other):
        if isinstance(other, FieldElement):
            return self * other.inv()
        return self * FieldElement(other, self.p).inv()

    def __rtruediv__(self, other):
        return FieldElement(other, self.p) / self

    def __eq__(self, other):
        if isinstance(other, FieldElement):
            return self.value == other.value and self.p == other.p
        return self.value == (other % self.p)

    def __pow__(self, exponent: int):
        return FieldElement(pow(self.value, exponent, self.p), self.p)

    def __repr__(self):
        return f"{self.value}"


# Common Production Primes
BABYBEAR = 2013265921  # 2^31 - 2^27 + 1