# Test q2a: Exponential and Poisson generators (mean, variance)

import sys
import os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np #type: ignore
from src.random_generator import UniformRandomGenerator
from src.distributions import ExponentialGenerator, PoissonGenerator


def test_q2a():
    print("=" * 50)
    print("test q2a: Exponential and Poisson generators")
    print("=" * 50)

    mean = 10.0
    n = 200000

    # Exponential
    ug = UniformRandomGenerator(seed=1)
    exp_gen = ExponentialGenerator(mean=mean, uniform_gen=ug)
    exp_samples = exp_gen.generate(n)

    exp_mean = np.mean(exp_samples)
    exp_var = np.var(exp_samples)
    print(f"\n  Exponential(mean={mean}):")
    print(f"    Sample mean : {exp_mean:.4f}  (expected {mean})")
    print(f"    Sample variance : {exp_var:.4f}  (expected {mean**2:.1f})")
    assert abs(exp_mean - mean) / mean < 0.02, "Exponential mean too far off"
    assert abs(exp_var - mean ** 2) / mean ** 2 < 0.05, "Exponential variance too far off"
    assert all(x >= 0 for x in exp_samples), "Negative exponential sample!"
    print("Mean, variance, and non-negativity checks passed.")

    # Poisson 
    ug2 = UniformRandomGenerator(seed=2)
    poi_gen = PoissonGenerator(mean=mean, uniform_gen=ug2)
    poi_samples = poi_gen.generate(n)

    poi_mean = np.mean(poi_samples)
    poi_var = np.var(poi_samples)
    print(f"\n  Poisson(mean={mean}):")
    print(f"    Sample mean : {poi_mean:.4f}  (expected {mean})")
    print(f"    Sample variance : {poi_var:.4f}  (expected {mean})")
    assert abs(poi_mean - mean) / mean < 0.02, "Poisson mean too far off"
    assert abs(poi_var - mean) / mean < 0.05, "Poisson variance too far off"
    assert all(isinstance(x, int) and x >= 0 for x in poi_samples), "Invalid Poisson sample!"
    print("Mean, variance, and non-negativity checks passed.")

    print("\n  q2a PASSED.\n")


if __name__ == "__main__":
    test_q2a()
