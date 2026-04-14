# Test Q4c: E[N] vs rho for M/E50/1 vs M/D/1 (theory) vs M/M/1 (theory)

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np #type: ignore
import matplotlib #type: ignore
matplotlib.use('Agg')
import matplotlib.pyplot as plt #type: ignore

from src.mek1_queue import MEk1Queue


def test_q4c():
    print("=" * 50)
    print("test q4c: E[N] vs ρ — M/E50/1 vs M/D/1 vs M/M/1")
    print("=" * 50)

    k = 50
    mu = 6.0
    rho_vals = [0.3, 0.5, 0.7, 0.85, 0.9]

    print(f"\n  k={k},  μ={mu}")
    print(f"  {'rho':>6}  {'ME50/1 sim':>12}  {'MD/1 theory':>13}  {'MM/1 theory':>13}  {'rel_err':>10}")
    print("  " + "-" * 60)

    en_sim_list, en_md1_list, en_mm1_list = [], [], []

    for rho in rho_vals:
        lam_r = rho * mu
        sim_r = MEk1Queue(arrival_rate=lam_r, service_rate=mu, k=k, seed=99)
        res_r = sim_r.simulate(n_packets=80_000, warmup=5_000)
        en_s = res_r["e_n_sim"]
        en_md1 = MEk1Queue.md1_expected_n(rho)
        en_mm1 = MEk1Queue.mm1_expected_n(rho)
        rel_err = abs(en_s - en_md1) / en_md1

        en_sim_list.append(en_s)
        en_md1_list.append(en_md1)
        en_mm1_list.append(en_mm1)

        print(f"  {rho:>6.2f}  {en_s:>12.4f}  {en_md1:>13.4f}  {en_mm1:>13.4f}  {rel_err:>10.4f}")
        assert rel_err < 0.10, f"M/E50/1 vs M/D/1 relative error too large at rho={rho}: {rel_err:.4f}"
        assert en_s < en_mm1, f"M/E50/1 should have fewer packets than M/M/1 at rho={rho}"

    print("\nM/E50/1 E[N] within 10% of M/D/1 for all tested ρ.")
    print("M/E50/1 E[N] < M/M/1 for all tested ρ.")
    print("Convergence to M/D/1 confirmed as k=50 >> 1.")

    # Save plot
    rho_dense = np.linspace(0.1, 0.94, len(rho_vals))
    os.makedirs("outputs", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rho_vals, en_sim_list, 'o-', label=f"M/E{k}/1 Simulated", color="#7c3aed", linewidth=2)
    ax.plot(rho_vals, en_md1_list, 's--', label="M/D/1 Theoretical (P-K)", color="#059669", linewidth=2)
    ax.plot(rho_vals, en_mm1_list, '^:', label="M/M/1 Theoretical", color="#dc2626", linewidth=2)
    ax.set_xlabel("ρ")
    ax.set_ylabel("E[N]")
    ax.set_title(f"Test q4c: E[N] vs ρ  (k={k})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/test_q4c_en_vs_rho.png", dpi=150)
    plt.close(fig)
    print("Plot saved to outputs/test_q4c_en_vs_rho.png")

    print("\n  q4c PASSED.\n")


if __name__ == "__main__":
    test_q4c()
