# Question 3: M/M/1 event driven queue simulation

import os
import math
import heapq
import numpy as np #type: ignore
import matplotlib.pyplot as plt #type: ignore
import matplotlib #type: ignore
matplotlib.use('Agg')

from src.distributions import ExponentialGenerator
from src.random_generator import UniformRandomGenerator

# M/M/1 event driven queue simulation
#
# Arrivals ~ Poisson(lambda) => inter-arrival ~ Exp(1/lambda)
# Service  ~ Exp(1/mu)
#
# Collects:
#   - Time-average queue length E[N]
#   - Per-packet sojourn time E[T]
#   - Steady-state distribution P_n
class MM1Queue:

    def __init__(self, arrival_rate: float, service_rate: float, seed: int = 42):
        if arrival_rate >= service_rate:
            raise ValueError("System is unstable: arrival_rate must be < service_rate.")
        
        self.lam = arrival_rate
        self.mu = service_rate
        self.rho = arrival_rate / service_rate

        ug = UniformRandomGenerator(seed=seed)
        ug2 = UniformRandomGenerator(seed=seed + 1)
        self._arr_gen = ExponentialGenerator(mean=1.0 / arrival_rate, uniform_gen=ug)
        self._svc_gen = ExponentialGenerator(mean=1.0 / service_rate, uniform_gen=ug2)


    # simulation
    #
    # Simulate the M/M/1 queue for n_packets arrivals.
    # The first warmup packets are discarded (transient phase).
    #
    # Returns a dict of statistics.
    def simulate(self, n_packets: int = 200_000, warmup: int = 10_000):

        # event types
        ARRIVAL = 0
        DEPARTURE = 1

        # state
        clock = 0.0
        queue_len = 0          # total in system (waiting + in service)
        server_busy = False
        departure_times = {}   # packet_id -> departure time
        arrival_times_map = {} # packet_id -> arrival time
        waiting_queue = []     # FIFO list of arrival times for waiting packets

        # stats accumulators
        last_event_time = 0.0
        n_in_system_history = []  # (duration, n)
        sojourn_times = []

        # count of packets that completed service (for warmup)
        completed = 0
        packet_id = 0

        # pqueue: (event_time, event_type, packet_id)
        events = []

        # schedule first arrival
        t_first_arr = self._arr_gen.generate(1)[0]
        heapq.heappush(events, (t_first_arr, ARRIVAL, packet_id))
        packet_id += 1

        # pre-generate many inter-arrivals and services to avoid per call overhead
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

        total_arrivals = 0

        while total_arrivals < n_packets or events:
            if not events:
                break

            event_time, event_type, pid = heapq.heappop(events)

            # accumulaye time ave n
            dt = event_time - last_event_time
            n_in_system_history.append((dt, queue_len))
            last_event_time = event_time
            clock = event_time

            if event_type == ARRIVAL:
                total_arrivals += 1
                queue_len += 1
                arrival_times_map[pid] = clock

                if not server_busy:
                    server_busy = True
                    svc_time = next_svc()
                    heapq.heappush(events, (clock + svc_time, DEPARTURE, pid))
                else:
                    waiting_queue.append(pid)

                # schedule next arrival
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

                # free memory
                arrival_times_map.pop(pid, None)

                if waiting_queue:
                    next_pid = waiting_queue.pop(0)
                    svc_time = next_svc()
                    heapq.heappush(events, (clock + svc_time, DEPARTURE, next_pid))
                else:
                    server_busy = False

            # quit once enough departures processed 
            if completed >= n_packets:
                break

        # compute stats
        # time ave E[N]
        total_time = sum(dt for dt, _ in n_in_system_history)
        if total_time > 0:
            e_n_sim = sum(dt * n for dt, n in n_in_system_history) / total_time
        else:
            e_n_sim = float('nan')

        e_t_sim = float(np.mean(sojourn_times)) if sojourn_times else float('nan')

        # theoretical values
        e_n_theory = self.rho / (1 - self.rho)
        e_t_theory = 1.0 / (self.mu - self.lam)

        # steady-state distribution P_n (simulation)
        max_n = 30
        pn_sim = np.zeros(max_n + 1)
        for dt, n in n_in_system_history:
            if n <= max_n:
                pn_sim[n] += dt
        pn_sim /= pn_sim.sum()

        # theoretical P_n = (1 - rho) * rho^n
        pn_theory = np.array([(1 - self.rho) * (self.rho ** n) for n in range(max_n + 1)])

        return {
            "e_n_sim": e_n_sim,
            "e_n_theory": e_n_theory,
            "e_t_sim": e_t_sim,
            "e_t_theory": e_t_theory,
            "pn_sim": pn_sim,
            "pn_theory": pn_theory,
            "rho": self.rho,
            "lam": self.lam,
            "mu": self.mu,
        }


    # plot helpers
    # plot P_n (simulated vs theoretical) for M/M/1
    @staticmethod
    def plot_pn(results: dict, save_path: str = None, ax=None):
        pn_sim = results["pn_sim"]
        pn_theory = results["pn_theory"]
        n_vals = np.arange(len(pn_sim))

        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=(10, 5))

        width = 0.4
        ax.bar(n_vals - width / 2, pn_sim, width, label="Simulated P_n", color="#2563eb", alpha=0.8)
        ax.bar(n_vals + width / 2, pn_theory, width, label="Theoretical P_n", color="#dc2626", alpha=0.8)
        ax.set_xlabel("n  (number of packets in system)", fontsize=13)
        ax.set_ylabel("P_n", fontsize=13)
        ax.set_title(
            f"Question 3(b): M/M/1 Steady-State Distribution\n"
            f"λ={results['lam']}, μ={results['mu']}, ρ={results['rho']:.4f}",
            fontsize=13
        )
        ax.legend(fontsize=11)
        ax.set_xlim(-1, 25)
        ax.grid(True, alpha=0.3, axis="y")

        if own_fig:
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150)
                print(f"  [Saved] {save_path}")
            else:
                plt.show()
            plt.close(fig)


# runner for all parts of Q3
def run_question_3(save_dir: str = "outputs"):
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("Q3: M/M/1 Queue Simulation")
    print("=" * 60)

    lam, mu = 5.0, 6.0
    rho = lam / mu

    print(f"\n  Parameters: λ={lam}, μ={mu}, ρ={rho:.4f}")
    print("  Simulating M/M/1 queue (200,000 packets, 10,000 warmup)...")

    sim = MM1Queue(arrival_rate=lam, service_rate=mu, seed=42)
    results = sim.simulate(n_packets=200_000, warmup=10_000)

    # 3b: plot P_n
    print("\n[3b] Plotting P_n vs n ...")
    save_path = os.path.join(save_dir, "q3b_pn_distribution.png")
    MM1Queue.plot_pn(results, save_path=save_path)

    # 3c: report E[N] and E[T]
    print("\n[3c] Expected number of packets and expected delay:")
    print(f"  E[N] simulated = {results['e_n_sim']:.4f}")
    print(f"  E[N] theoretical = {results['e_n_theory']:.4f}  (ρ/(1-ρ))")
    print(f"  E[T] simulated = {results['e_t_sim']:.6f}")
    print(f"  E[T] theoretical = {results['e_t_theory']:.6f}  (1/(μ-λ))")

    lln_check = results['e_n_sim'] / lam
    print(f"\n  Little's Law check: E[N]/λ = {lln_check:.6f}  vs E[T]_sim = {results['e_t_sim']:.6f}")

    return sim, results
