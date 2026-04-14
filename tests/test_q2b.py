# Test Q2b: Tail probability plots for Exponential and Poisson.

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np #type: ignore
from src.random_generator import UniformRandomGenerator
from src.distributions import ExponentialGenerator, PoissonGenerator, run_question_2


def test_q2b():
    print("=" * 50)
    print("test q2b: Tail Probability Plots")
    print("=" * 50)

    mean = 10.0
    n = 200_000

    ug = UniformRandomGenerator(seed=3)
    exp_gen = ExponentialGenerator(mean=mean, uniform_gen=ug)
    exp_samples = exp_gen.generate(n)

    ug2 = UniformRandomGenerator(seed=4)
    poi_gen = PoissonGenerator(mean=mean, uniform_gen=ug2)
    poi_samples = poi_gen.generate(n)

    # Check a few tail probabilities
    test_xs_exp = [5, 10, 20, 30]
    print("\n  Exponential tail check:")
    print(f"    {'x':>5}  {'sim':>10}  {'theory':>10}  {'rel_err':>10}")
    for x in test_xs_exp:
        sim_p = np.mean(np.array(exp_samples) > x)
        theo_p = exp_gen.theoretical_tail(x)
        rel_err = abs(sim_p - theo_p) / max(theo_p, 1e-9)
        print(f"    {x:>5}  {sim_p:>10.6f}  {theo_p:>10.6f}  {rel_err:>10.4f}")
        assert rel_err < 0.05, f"Exponential tail error too large at x={x}"
    print("Exponential tail within 5% of theory.")

    test_xs_poi = [5, 10, 15, 20]
    print("\n  Poisson tail check:")
    print(f"    {'x':>5}  {'sim':>10}  {'theory':>10}  {'abs_err':>10}")
    for x in test_xs_poi:
        sim_p = np.mean(np.array(poi_samples) > x)
        theo_p = poi_gen.theoretical_tail(x)
        abs_err = abs(sim_p - theo_p)
        print(f"    {x:>5}  {sim_p:>10.6f}  {theo_p:>10.6f}  {abs_err:>10.6f}")
        assert abs_err < 0.02, f"Poisson tail error too large at x={x}"
    print("Poisson tail within 0.02 of theory.")

    # Save plot
    os.makedirs("outputs", exist_ok=True)
    run_question_2(save_dir="outputs", n_samples=100_000)
    print("\nPlot saved.")

    print("\n  q2b PASSED.\n")


if __name__ == "__main__":
    test_q2b()
