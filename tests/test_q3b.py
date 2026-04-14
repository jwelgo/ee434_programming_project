# Test Q3b: M/M/1 P_n distribution plot

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np #type: ignore
from src.mm1_queue import MM1Queue


def test_q3b():
    print("=" * 50)
    print("test q3b: M/M/1 steady state distribution P_n")
    print("=" * 50)

    lam, mu = 5.0, 6.0
    rho = lam / mu

    sim = MM1Queue(arrival_rate=lam, service_rate=mu, seed=2)
    results = sim.simulate(n_packets=200_000, warmup=10_000)

    pn_sim = results["pn_sim"]
    pn_theory = results["pn_theory"]

    # Check P_0 through P_5
    print(f"\n  {'n':>4}  {'P_n sim':>12}  {'P_n theory':>12}  {'abs_err':>10}")
    print("  " + "-" * 44)
    for n in range(6):
        err = abs(pn_sim[n] - pn_theory[n])
        print(f"  {n:>4}  {pn_sim[n]:>12.6f}  {pn_theory[n]:>12.6f}  {err:>10.6f}")
        assert err < 0.01, f"P_{n} error too large: {err:.6f}"

    print("\nP_n matches theory within 0.01 for n=0..5.")

    # Verify distributions sum to ~1
    assert abs(sum(pn_sim) - 1.0) < 1e-6, "Simulated P_n doesn't sum to 1"
    print("Simulated P_n sums to 1.")

    # Save plot
    os.makedirs("outputs", exist_ok=True)
    MM1Queue.plot_pn(results, save_path="outputs/test_q3b_pn.png")
    print("Plot saved to outputs/test_q3b_pn.png")

    print("\n  q3b PASSED.\n")


if __name__ == "__main__":
    test_q3b()
