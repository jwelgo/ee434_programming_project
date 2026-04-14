# Question 1: uniform random number generator and P(U > x) estimation

import os
import random
import numpy as np #type: ignore
import matplotlib.pyplot as plt #type: ignore
import matplotlib #type: ignore
matplotlib.use('Agg')


# generates U[0,1] random numbers and estimates tail probabilities
class UniformRandomGenerator:

    def __init__(self, seed: int = None):
        self._rng = random.Random(seed)

    # gen n U[0,1] random numbers
    def generate(self, n: int) -> list[float]:
        return [self._rng.random() for _ in range(n)]

    # estimate P(U > x) by counting fraction of samples > x
    def estimate_tail_probability(self, samples: list[float], x: float) -> float:
        count_above = sum(1 for s in samples if s > x)
        return count_above / len(samples)


    # plot estimated P(U > x) for x in (0.5, 1)
    # theoretical value: P(U > x) = 1 - x (straight line)
    def plot_tail_probability(self, n_samples: int = 100_000, x_values: np.ndarray = None, save_path: str = None, ax=None):
        if x_values is None:
            x_values = np.linspace(0.501, 0.999, 100)

        samples = self.generate(n_samples)
        estimated = [self.estimate_tail_probability(samples, x) for x in x_values]
        theoretical = [1 - x for x in x_values]

        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(x_values, estimated, label=f"Simulated (n={n_samples:,})", color="#2563eb", linewidth=2)
        ax.plot(x_values, theoretical, label="Theoretical: 1 - x", color="#dc2626", linestyle="--", linewidth=2)
        ax.set_xlabel("x", fontsize=13)
        ax.set_ylabel("P(U > x)", fontsize=13)
        ax.set_title("Question 1(b): Estimated P(U > x) for x ∈ (0.5, 1)", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        if own_fig:
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150)
                print(f"  [Saved] {save_path}")
            else:
                plt.show()
            plt.close(fig)

        return x_values, estimated, theoretical


# runner for all parts of Q1
def run_question_1(save_dir: str = "outputs", show_inline: bool = False):
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("Q1: Uniform Random Variable")
    print("=" * 60)

    gen = UniformRandomGenerator(seed=42)

    # 1a: generate samples and print summary
    print("\n[1a] Generating 10 U[0,1] samples (preview):")
    samples_preview = gen.generate(10)
    for i, s in enumerate(samples_preview, 1):
        print(f"  Sample {i:2d}: {s:.6f}")

    # 1b: plot
    print("\n[1b] Estimating and plotting P(U > x) for x in (0.5, 1) ...")
    save_path = os.path.join(save_dir, "q1b_tail_probability.png")
    gen.plot_tail_probability(n_samples=100_000, save_path=save_path)
    print("  As expected, P(U > x) = 1 - x is approximately a straight line.")

    return gen
