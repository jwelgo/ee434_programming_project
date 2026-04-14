# Test Q3a: M/M/1 Queue – verify simulator correctness against theory

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mm1_queue import MM1Queue


def test_q3a():
    print("=" * 50)
    print("test q3a: M/M/1 Queue Simulator Correctness")
    print("=" * 50)

    lam, mu = 5.0, 6.0
    sim = MM1Queue(arrival_rate=lam, service_rate=mu, seed=1)
    results = sim.simulate(n_packets=200_000, warmup=10_000)

    rho = lam / mu
    print(f"\n  ρ = {rho:.4f}")

    # E[N]
    en_sim = results["e_n_sim"]
    en_theory = results["e_n_theory"]
    en_rel_err = abs(en_sim - en_theory) / en_theory
    print(f"\n  E[N] simulated : {en_sim:.4f}")
    print(f"  E[N] theoretical : {en_theory:.4f}")
    print(f"  Relative error : {en_rel_err:.4f} ({en_rel_err*100:.2f}%)")
    assert en_rel_err < 0.05, f"E[N] relative error too large: {en_rel_err:.4f}"
    print("E[N] within 5% of theory.")

    # E[T]
    et_sim = results["e_t_sim"]
    et_theory = results["e_t_theory"]
    et_rel_err = abs(et_sim - et_theory) / et_theory
    print(f"\n  E[T] simulated : {et_sim:.6f}")
    print(f"  E[T] theoretical : {et_theory:.6f}")
    print(f"  Relative error : {et_rel_err:.4f} ({et_rel_err*100:.2f}%)")
    assert et_rel_err < 0.05, f"E[T] relative error too large: {et_rel_err:.4f}"
    print("E[T] within 5% of theory.")

    # Little's Law: E[N] = lambda * E[T]
    lln_en = lam * et_sim
    lln_err = abs(en_sim - lln_en) / en_sim
    print(f"\n  Little's Law check: λ*E[T] = {lln_en:.4f}  vs  E[N] = {en_sim:.4f}")
    assert lln_err < 0.03, f"Little's Law check failed: {lln_err:.4f}"
    print("Little's Law satisfied (within 3%).")

    print("\n  q3a PASSED.\n")


if __name__ == "__main__":
    test_q3a()
