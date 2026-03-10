import numpy as np
import matplotlib.pyplot as plt

from simple_ris_env import SimpleRISEnv


def jain_fairness(R1: float, R2: float) -> float:
    num = (R1 + R2) ** 2
    den = 2.0 * (R1**2 + R2**2)
    return float(num / den) if den > 0 else 0.0


def main():
    # match ECE setup: N = 128 (can change if needed)
    N = 128
    mc_per_point = 50  # channel realizations per N1

    env = SimpleRISEnv(N=N, max_steps=1, mc_samples=1)

    fairness_means = []
    partitions = list(range(0, N + 1))

    for N1 in partitions:
        N2 = N - N1
        vals = []
        for _ in range(mc_per_point):
            ch = env._generate_rayleigh_channels(N1, N2)
            sinr1, sinr2 = env._compute_sinr(ch, N1, N2)
            R1 = np.log2(1.0 + sinr1)
            R2 = np.log2(1.0 + sinr2)
            vals.append(jain_fairness(R1, R2))

        fairness_means.append(float(np.mean(vals)))

    plt.figure(figsize=(8, 4))
    plt.plot(partitions, fairness_means, marker="o", markersize=2)
    plt.xlabel("N1 (active RIS elements for U1)")
    plt.ylabel("Average Jain fairness")
    plt.title(f"ECE RIS–NOMA model check (N = {N}, MC = {mc_per_point})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

