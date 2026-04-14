# Test q1b: P(U > x) estimation and plot.
# Verifies the estimated tail probability is close to 1-x for several x values.

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.random_generator import UniformRandomGenerator


def test_q1b():
    print("=" * 50)
    print("test q1b: P(U > x) estimation")
    print("=" * 50)

    gen = UniformRandomGenerator(seed=42)
    n = 500000
    samples = gen.generate(n)

    test_xs = [0.55, 0.65, 0.75, 0.85, 0.95]
    print(f"\n  {'x':>6}  {'P(U>x) sim':>12}  {'1-x (theory)':>14}  {'error':>8}")
    print("  " + "-" * 46)
    for x in test_xs:
        estimated = gen.estimate_tail_probability(samples, x)
        theoretical = 1 - x
        error = abs(estimated - theoretical)
        print(f"  {x:>6.2f}  {estimated:>12.6f}  {theoretical:>14.6f}  {error:>8.6f}")
        assert error < 0.005, f"Tail probability error too large at x={x}: {error}"

    print("\nAll estimates within 0.5% of theoretical value.")

    # Generate and save plot
    os.makedirs("outputs", exist_ok=True)
    gen2 = UniformRandomGenerator(seed=42)
    gen2.plot_tail_probability(n_samples=200_000, save_path="outputs/test_q1b_tail.png")
    print("Plot saved to outputs/test_q1b_tail.png")

    print("\n  q1b PASSED.\n")


if __name__ == "__main__":
    test_q1b()
