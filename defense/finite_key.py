import numpy as np

def finite_key_penalty(n: int, epsilon: float = 1e-10) -> float:
    """Conservative composable finite-size penalty in bits/pulse."""
    if n <= 0:
        return 0.0
    return 3.0 * np.sqrt(np.log(1/epsilon) / n) / np.log(2)
