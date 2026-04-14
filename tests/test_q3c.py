# Test Q3c: M/M/1 E[N] and E[T] directly from simulation

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mm1_queue import MM1Queue


def test_q3c():
    print("=" * 50)
    print("test q3c: M/M/1 E[N] and E[T] from simulation")
    print("=" * 50)

    lam, mu = 5.0, 6.0
    rho = lam / mu

    sim = MM1Queue(arrival_rate=lam, service_rate=mu, seed=3)
    results = sim.simulate(n_packets=200_000, warmup=10_000)

    print(f"\n  ρ = {rho:.4f}")

    # E[N]
    e_n_sim = results["e_n_sim"]
    e_n_theory = rho / (1 - rho)
    en_err = abs(e_n_sim - e_n_theory) / e_n_theory
    print(f"\n  E[N] simulated : {e_n_sim:.5f}")
    print(f"  E[N] theoretical : {e_n_theory:.5f}")
    print(f"  Relative error : {en_err*100:.2f}%")
    assert en_err < 0.05

    # E[T]
    e_t_sim = results["e_t_sim"]
    e_t_theory = 1.0 / (mu - lam)
    et_err = abs(e_t_sim - e_t_theory) / e_t_theory
    print(f"\n  E[T] simulated : {e_t_sim:.6f}")
    print(f"  E[T] theoretical : {e_t_theory:.6f}")
    print(f"  Relative error : {et_err*100:.2f}%")
    assert et_err < 0.05

    print("\nBoth E[N] and E[T] within 5% of theory.")

    # Little's Law cross-check (verification only, not derivation)
    lln = lam * e_t_sim
    print(f"\n  Little's Law verification: λ·E[T] = {lln:.5f}  ≈  E[N] = {e_n_sim:.5f}")
    assert abs(lln - e_n_sim) / e_n_sim < 0.03
    print("Little's Law verified.")

    print("\n  q3c PASSED.\n")


if __name__ == "__main__":
    test_q3c()
