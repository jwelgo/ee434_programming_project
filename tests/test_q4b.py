# Test Q4b: M/E4/1 queue - P_n plot and E[N] comparison with M/M/1

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np #type: ignore
import matplotlib #type: ignore
matplotlib.use('Agg')
import matplotlib.pyplot as plt #type: ignore

from src.mek1_queue import MEk1Queue
from src.mm1_queue import MM1Queue


def test_q4b():
    print("=" * 50)
    print("test q4b: M/E4/1 vs M/M/1 – P_n and E[N]")
    print("=" * 50)

    k, lam, mu = 4, 5.0, 6.0
    rho = lam / mu

    print(f"\n  Simulating M/E4/1: k={k}, λ={lam}, μ={mu}, ρ={rho:.4f} ...")
    sim_ek1 = MEk1Queue(arrival_rate=lam, service_rate=mu, k=k, seed=10)
    res_ek1 = sim_ek1.simulate(n_packets=200_000, warmup=10_000)

    print(f"  Simulating M/M/1:  λ={lam}, μ={mu} ...")
    sim_mm1 = MM1Queue(arrival_rate=lam, service_rate=mu, seed=11)
    res_mm1 = sim_mm1.simulate(n_packets=200_000, warmup=10_000)

    # E[N] comparisons
    en_ek1 = res_ek1["e_n_sim"]
    en_pk = res_ek1["e_n_pk"]
    en_mm1 = res_mm1["e_n_sim"]

    print(f"\n  E[N] M/E4/1 simulated : {en_ek1:.4f}")
    print(f"  E[N] M/E4/1 P-K formula: {en_pk:.4f}")
    print(f"  E[N] M/M/1  simulated : {en_mm1:.4f}")

    # M/Ek/1 should have lower E[N] than M/M/1
    assert en_ek1 < en_mm1, "M/Ek/1 E[N] should be less than M/M/1 E[N]!"
    print("E[N] M/E4/1 < E[N] M/M/1 (lower service variance reduces congestion).")

    # P-K formula accuracy
    pk_err = abs(en_ek1 - en_pk) / en_pk
    assert pk_err < 0.05, f"P-K formula error too large: {pk_err:.4f}"
    print(f"E[N] within 5% of P-K formula (err={pk_err*100:.2f}%).")

    # Save comparison plot
    os.makedirs("outputs", exist_ok=True)
    n_vals = np.arange(len(res_ek1["pn_sim"]))
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.4
    ax.bar(n_vals - width/2, res_ek1["pn_sim"], width,
           label="M/E4/1 Simulated", color="#7c3aed", alpha=0.8)
    ax.bar(n_vals + width/2, res_mm1["pn_sim"][:len(n_vals)], width,
           label="M/M/1 Simulated", color="#dc2626", alpha=0.8)
    ax.set_xlabel("n")
    ax.set_ylabel("P_n")
    ax.set_title(f"Test q4b: P_n – M/E4/1 vs M/M/1  (λ={lam}, μ={mu})")
    ax.legend()
    ax.set_xlim(-1, 25)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("outputs/test_q4b_pn.png", dpi=150)
    plt.close(fig)
    print("Plot saved to outputs/test_q4b_pn.png")

    print("\n  q4b PASSED.\n")


if __name__ == "__main__":
    test_q4b()
