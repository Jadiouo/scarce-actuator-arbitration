"""REQ-S1-19: feature sufficiency checks."""
from typing import List


def groups() -> List[List[int]]:
    """REQ-S1-19: the six leave-one-group-out index groups over f01..f25 (0-based)."""
    return [[0, 1], [2, 3, 4], list(range(5, 13)), list(range(13, 21)), [21, 22], [23, 24]]


def history64_dim() -> int:
    """REQ-S1-19: 134 (won_{t-j}, d_{t-j}, j=1..64, plus f01-f05, f22)."""
    return 2 * 64 + 5 + 1
