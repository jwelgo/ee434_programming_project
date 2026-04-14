# Question 2: Exponential and Poisson random variables via inverse transform sampling

import os
import math
import numpy as np #type: ignore
import matplotlib.pyplot as plt #type: ignore
import matplotlib #type: ignore
matplotlib.use('Agg')

from src.random_generator import UniformRandomGenerator


# Generates Exponential (rate=1/mean) samples via inverse transform:
#     X = -mean * ln(U),  U ~ U(0,1)
class ExponentialGenerator:

    def __init__(self, mean: float, uniform_gen: UniformRandomGenerator = None):
        self.mean = mean
        self._uniform = uniform_gen or UniformRandomGenerator()

    def generate(self, n: int) -> list[float]:
        uniforms = self._uniform.generate(n)
        return [-self.mean * math.log(max(u, 1e-15)) for u in uniforms] # Avoid log(0) by clamping

    # P(X > x) = exp(-x / mean)
    def theoretical_tail(self, x: float) -> float:
        return math.exp(-x / self.mean)


# Generates Poisson(mean) samples via the inter arrival method:
# Count exponential interarrivals until cumulative sum > 1
#
#  N = number of U_i whose product >= e^(-mean)
class PoissonGenerator:

    def __init__(self, mean: float, uniform_gen: UniformRandomGenerator = None):
        self.mean = mean
        self._uniform = uniform_gen or UniformRandomGenerator()

    # generate a single Poisson(mean) sample
    def _generate_one(self) -> int:
        limit = math.exp(-self.mean)
        product = 1.0
        k = 0
        while True:
            u = self._uniform.generate(1)[0]
            product *= u
            if product < limit:
                return k
            k += 1

    def generate(self, n: int) -> list[int]:
        return [self._generate_one() for _ in range(n)]

    # P(Y > x) = 1 -> CDF(x) computed via summing PMF
    def theoretical_tail(self, x: int) -> float:
        cdf = 0.0
        lam = self.mean
        log_pmf = -lam  # log P(Y=0)
        for k in range(int(x) + 1):
            if k > 0:
                log_pmf += math.log(lam) - math.log(k)
            cdf += math.exp(log_pmf)
        return max(1.0 - cdf, 0.0)


# runner for all parts of Q2 
def run_question_2(save_dir: str = "outputs", n_samples: int = 100_000):
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("Q2: Exponential and Poisson Random Variables")
    print("=" * 60)

    mean = 10.0
    seed_gen = UniformRandomGenerator(seed=7)
    exp_gen = ExponentialGenerator(mean=mean, uniform_gen=seed_gen)

    seed_gen2 = UniformRandomGenerator(seed=13)
    poi_gen = PoissonGenerator(mean=mean, uniform_gen=seed_gen2)

    # 2a: generate and preview
    print(f"\n[2a] Generating samples with E(X)=E(Y)={mean}")

    exp_samples = exp_gen.generate(n_samples)
    poi_samples = poi_gen.generate(n_samples)

    print(f"  Exponential: sample mean = {np.mean(exp_samples):.4f}  (expected {mean})")
    print(f"  Poisson: sample mean = {np.mean(poi_samples):.4f}  (expected {mean})")

    # 2b: plot P(X > x) and P(Y > x) on log-scale
    print("\n[2b] Plotting P(X > x) and P(Y > x) on log-scale ...")

    # Exponential tail
    x_exp = np.linspace(0, 60, 300)
    sim_exp_tail = [np.mean(np.array(exp_samples) > xv) for xv in x_exp]
    theo_exp_tail = [exp_gen.theoretical_tail(xv) for xv in x_exp]

    # Poisson tail 
    x_poi = np.arange(0, 50)
    sim_poi_tail = [np.mean(np.array(poi_samples) > xv) for xv in x_poi]
    theo_poi_tail = [poi_gen.theoretical_tail(xv) for xv in x_poi]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Exponential
    axes[0].semilogy(x_exp, np.clip(sim_exp_tail, 1e-6, 1), label="Simulated", color="#2563eb", linewidth=2)
    axes[0].semilogy(x_exp, np.clip(theo_exp_tail, 1e-6, 1), label="Theoretical", color="#dc2626", linestyle="--", linewidth=2)
    axes[0].set_xlabel("x", fontsize=13)
    axes[0].set_ylabel("P(X > x)  [log scale]", fontsize=13)
    axes[0].set_title("Question 2(b): Exponential Tail Probability", fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3, which="both")

    # Poisson
    axes[1].semilogy(x_poi, np.clip(sim_poi_tail, 1e-6, 1), 'o-', label="Simulated", color="#2563eb", linewidth=2, markersize=4)
    axes[1].semilogy(x_poi, np.clip(theo_poi_tail, 1e-6, 1), 's--', label="Theoretical", color="#dc2626", linewidth=2, markersize=4)
    axes[1].set_xlabel("x", fontsize=13)
    axes[1].set_ylabel("P(Y > x)  [log scale]", fontsize=13)
    axes[1].set_title("Question 2(b): Poisson Tail Probability", fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "q2b_tail_probabilities.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] {save_path}")

    return exp_gen, poi_gen
