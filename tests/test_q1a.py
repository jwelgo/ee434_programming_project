# Test q1a: Uniform random number generator
# Verifies that generated samples are in [0,1] and have correct statistics.

import sys
import os
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.random_generator import UniformRandomGenerator


def test_q1a():
    print("=" * 50)
    print("test q1a: Uniform Random Number Generator")
    print("=" * 50)

    gen = UniformRandomGenerator(seed=0)
    n = 100000
    samples = gen.generate(n)

    # All in [0, 1]
    assert all(0.0 <= s <= 1.0 for s in samples), "Sample out of [0,1] range!"
    print(f"All {n:,} samples are in [0, 1]")

    mean = statistics.mean(samples)
    print(f"  Sample mean  : {mean:.5f}  (expected ~0.5)")
    assert abs(mean - 0.5) < 0.01, f"Mean too far from 0.5: {mean}"
    print("Mean is close to 0.5")

    variance = statistics.variance(samples)
    print(f"  Sample var   : {variance:.5f}  (expected ~0.0833)")
    assert abs(variance - 1 / 12) < 0.005, f"Variance off: {variance}"
    print("Variance is close to 1/12")

    print("\n  q1a PASSED.\n")


if __name__ == "__main__":
    test_q1a()
