# Question 4: M/Ek/1 queue simulation for Erlang-k service times

import os
import math
import heapq
import numpy as np #type: ignore
import matplotlib.pyplot as plt #type: ignore
import matplotlib #type: ignore
matplotlib.use('Agg')

from src.mm1_queue import MM1Queue
from src.distributions import ExponentialGenerator
from src.random_generator import UniformRandomGenerator


# Generates Erlang-k(mean) samples:
#     each sample = sum of k i.i.d. Exp(k/mean) rvs
#     E[Erlang-k] = k * (mean/k) = mean
class ErlangGenerator:

    def __init__(self, k: int, mean: float, uniform_gen: UniformRandomGenerator = None):
        self.k = k
        self.mean = mean

        # each phase has mean = mean/k  ->  rate = k/mean
        self._phase_gen = ExponentialGenerator(
            mean=mean / k,
            uniform_gen=uniform_gen or UniformRandomGenerator()
        )

    # generate n Erlang-k samples
    def generate(self, n: int) -> list[float]:
        phases = self._phase_gen.generate(n * self.k)
        return [sum(phases[i * self.k:(i + 1) * self.k]) for i in range(n)]

    @property
    def variance(self) -> float:
        return self.mean ** 2 / self.k


# event driven M/Ek/1 queue simulator
# Arrivals ~ Poisson(lambda), service ~ Erlang-k(mean=1/mu)
class MEk1Queue:

    def __init__(self, arrival_rate: float, service_rate: float, k: int, seed: int = 42):
        if arrival_rate >= service_rate:
            raise ValueError("System is unstable: arrival_rate must be < service_rate.")
        
        self.lam = arrival_rate
        self.mu = service_rate
        self.k = k
        self.rho = arrival_rate / service_rate

        ug_arr = UniformRandomGenerator(seed=seed)
        ug_svc = UniformRandomGenerator(seed=seed + 100)

        self._arr_gen = ExponentialGenerator(mean=1.0 / arrival_rate, uniform_gen=ug_arr)
        self._svc_gen = ErlangGenerator(k=k, mean=1.0 / service_rate, uniform_gen=ug_svc)

    # Event-driven simulation of M/Ek/1 queue
    def simulate(self, n_packets: int = 200_000, warmup: int = 10_000) -> dict:
        ARRIVAL = 0
        DEPARTURE = 1

        clock = 0.0
        queue_len = 0
        server_busy = False
        waiting_queue = []
        arrival_times_map = {}

        last_event_time = 0.0
        n_history = []
        sojourn_times = []

        completed = 0
        packet_id = 0
        total_arrivals = 0

        events = []

        # pre generate pools
        batch = max(n_packets * 2, 50_000)
        arr_pool = iter(self._arr_gen.generate(batch))
        svc_pool = iter(self._svc_gen.generate(batch))

        def next_arr():
            try:
                return next(arr_pool)
            except StopIteration:
                return self._arr_gen.generate(1)[0]

        def next_svc():
            try:
                return next(svc_pool)
            except StopIteration:
                return self._svc_gen.generate(1)[0]

        heapq.heappush(events, (next_arr(), ARRIVAL, packet_id))
        packet_id += 1

        while completed < n_packets:
            if not events:
                break

            event_time, event_type, pid = heapq.heappop(events)
            dt = event_time - last_event_time
            n_history.append((dt, queue_len))
            last_event_time = event_time
            clock = event_time

            if event_type == ARRIVAL:
                total_arrivals += 1
                queue_len += 1
                arrival_times_map[pid] = clock

                if not server_busy:
                    server_busy = True
                    heapq.heappush(events, (clock + next_svc(), DEPARTURE, pid))
                else:
                    waiting_queue.append(pid)

                if total_arrivals < n_packets:
                    new_pid = packet_id
                    packet_id += 1
                    heapq.heappush(events, (clock + next_arr(), ARRIVAL, new_pid))

            elif event_type == DEPARTURE:
                queue_len -= 1
                completed += 1

                if completed > warmup:
                    arr_t = arrival_times_map.get(pid, clock)
                    sojourn_times.append(clock - arr_t)
                arrival_times_map.pop(pid, None)

                if waiting_queue:
                    next_pid = waiting_queue.pop(0)
                    heapq.heappush(events, (clock + next_svc(), DEPARTURE, next_pid))
                else:
                    server_busy = False

        # stats
        total_time = sum(dt for dt, _ in n_history)
        e_n_sim = sum(dt * n for dt, n in n_history) / total_time if total_time else float('nan')
        e_t_sim = float(np.mean(sojourn_times)) if sojourn_times else float('nan')

        # steady state distribution P_n
        max_n = 30
        pn_sim = np.zeros(max_n + 1)
        for dt, n in n_history:
            if n <= max_n:
                pn_sim[n] += dt
        pn_sim /= pn_sim.sum()

        # mean value formula for M/G/1:
        # E[N] = rho + rho^2 * (1 + Cv^2) / (2*(1-rho))
        # For Erlang-k: Cv^2 = 1/k
        cv2 = 1.0 / self.k
        e_n_pk = self.rho + (self.rho ** 2) * (1 + cv2) / (2 * (1 - self.rho))

        return {
            "e_n_sim": e_n_sim,
            "e_n_pk": e_n_pk,
            "e_t_sim": e_t_sim,
            "pn_sim": pn_sim,
            "rho": self.rho,
            "lam": self.lam,
            "mu": self.mu,
            "k": self.k,
        }

    # theioretical M/D/1 E[N] for comparison in 4c
    # 
    # M/D/1 mean number in system:
    #   E[N] = rho + rho^2 / (2*(1-rho))
    # service -> Cv^2 = 0
    @staticmethod
    def md1_expected_n(rho: float) -> float:
        if rho >= 1:
            return float('inf')
        return rho + rho ** 2 / (2 * (1 - rho))

    # M/M/1: E[N] = rho / (1 - rho)
    @staticmethod
    def mm1_expected_n(rho: float) -> float:
        if rho >= 1:
            return float('inf')
        return rho / (1 - rho)


# runner for all parts of Q4
def run_question_4(save_dir: str = "outputs"):
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("Q4: M/Ek/1 Queue Simulation")
    print("=" * 60)

    # 4a + 4b: k=4, lambda=5, mu=6
    k, lam, mu = 4, 5.0, 6.0
    print(f"\n[4a/4b] Parameters: k={k}, λ={lam}, μ={mu}, ρ={lam/mu:.4f}")
    print("  Simulating M/E4/1 queue ...")

    sim4 = MEk1Queue(arrival_rate=lam, service_rate=mu, k=k, seed=42)
    res4 = sim4.simulate(n_packets=200_000, warmup=10_000)

    # compare with M/M/1 (
    # re-use results from question 3 if possible
    sim_mm1 = MM1Queue(arrival_rate=lam, service_rate=mu, seed=99)
    res_mm1 = sim_mm1.simulate(n_packets=200_000, warmup=10_000)

    print(f"\n[4b] E[N] M/E4/1 simulated = {res4['e_n_sim']:.4f}")
    print(f"     E[N] M/E4/1 P-K formula = {res4['e_n_pk']:.4f}")
    print(f"     E[N] M/M/1  simulated = {res_mm1['e_n_sim']:.4f}")
    print(f"     E[N] M/M/1  theoretical = {res_mm1['e_n_theory']:.4f}")
    print("\n  M/Ek/1 has fewer packets on average than M/M/1 because")
    print("  Erlang-k service has lower variance (Cv^2=1/k < 1).")

    # Plot P_n for M/E4/1
    fig, ax = plt.subplots(figsize=(10, 5))
    n_vals = np.arange(len(res4["pn_sim"]))
    width = 0.4
    ax.bar(n_vals - width / 2, res4["pn_sim"], width, label="M/E4/1 Simulated", color="#7c3aed", alpha=0.8)
    ax.bar(n_vals + width / 2, res_mm1["pn_sim"][:len(n_vals)], width, label="M/M/1 Simulated", color="#dc2626", alpha=0.8)
    ax.set_xlabel("n  (number of packets in system)", fontsize=13)
    ax.set_ylabel("P_n", fontsize=13)
    ax.set_title(
        f"Question 4(b): P_n — M/E4/1 vs M/M/1\n"
        f"λ={lam}, μ={mu}, ρ={lam/mu:.4f}",
        fontsize=13
    )
    ax.legend(fontsize=11)
    ax.set_xlim(-1, 25)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    save_path = os.path.join(save_dir, "q4b_pn_mek1_vs_mm1.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\n  [Saved] {save_path}")

    # 4c: E[N] vs rho for M/Ek/1 (k=50), M/D/1 theoretical, M/M/1
    print("\n[4c] Plotting E[N] vs ρ for M/E50/1, M/D/1, M/M/1 ...")
    rho_vals = np.linspace(0.1, 0.95, 18)
    k50 = 50
    base_mu = 6.0

    en_mek1_sim = []
    en_md1_theory = []
    en_mm1_theory = []

    for rho in rho_vals:
        lam_r = rho * base_mu
        sim_r = MEk1Queue(arrival_rate=lam_r, service_rate=base_mu, k=k50, seed=77)
        res_r = sim_r.simulate(n_packets=100_000, warmup=5_000)
        en_mek1_sim.append(res_r["e_n_sim"])
        en_md1_theory.append(MEk1Queue.md1_expected_n(rho))
        en_mm1_theory.append(MEk1Queue.mm1_expected_n(rho))
        print(f"    ρ={rho:.2f}  E[N] M/E50/1={res_r['e_n_sim']:.3f}  "
              f"M/D/1={en_md1_theory[-1]:.3f}  M/M/1={en_mm1_theory[-1]:.3f}")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(rho_vals, en_mek1_sim, 'o-', label=f"M/E{k50}/1 Simulated", color="#7c3aed", linewidth=2)
    ax.plot(rho_vals, en_md1_theory, 's--', label="M/D/1 Theoretical (P-K)", color="#059669", linewidth=2)
    ax.plot(rho_vals, en_mm1_theory, '^:', label="M/M/1 Theoretical", color="#dc2626", linewidth=2)
    ax.set_xlabel("ρ (utilization)", fontsize=13)
    ax.set_ylabel("E[N]  (expected packets in system)", fontsize=13)
    ax.set_title(f"Question 4(c): E[N] vs ρ\nM/E{k50}/1 (sim) vs M/D/1 (theory) vs M/M/1 (theory)", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "q4c_en_vs_rho.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\n  [Saved] {save_path}")
    print("\n  As k→infinity, Erlang-k → Deterministic (Cv^2→0).")
    print("  M/E50/1 E[N] closely matches M/D/1, confirming P-K formula.")
    print("  Both are well below M/M/1, showing that reducing service variance")
    print("  significantly reduces congestion at equal load.")

    return sim4, res4
