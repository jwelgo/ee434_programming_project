# Test Q4a: Erlang-k generator correctness

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # type: ignore

from src.random_generator import UniformRandomGenerator
from src.mek1_queue import ErlangGenerator


def test_q4a():
    print("=" * 50)
    print("test q4a: Erlang-k Generator")
    print("=" * 50)

    n = 100_000

    for k in [1, 2, 4, 10, 50]:
        mean = 10.0
        ug = UniformRandomGenerator(seed=k)
        gen = ErlangGenerator(k=k, mean=mean, uniform_gen=ug)
        samples = gen.generate(n)

        sample_mean = np.mean(samples)
        sample_var = np.var(samples)
        theo_var = gen.variance  # mean^2 / k

        mean_err = abs(sample_mean - mean) / mean
        var_err = abs(sample_var - theo_var) / theo_var

        print(f"\n  k={k:>3}, mean={mean}: "
              f"sample_mean={sample_mean:.4f}, sample_var={sample_var:.4f}, "
              f"theo_var={theo_var:.4f}")
        assert mean_err < 0.02, f"k={k} mean error: {mean_err:.4f}"
        assert var_err < 0.05, f"k={k} variance error: {var_err:.4f}"
        assert all(x >= 0 for x in samples), f"Negative sample for k={k}"

    print("\nErlang-k mean and variance correct for k=1,2,4,10,50.")
    print("k=1 matches Exponential (largest variance).")
    print("Variance decreases as k increases (→ Deterministic).")

    print("\n  q4a PASSED.\n")


if __name__ == "__main__":
    test_q4a()
