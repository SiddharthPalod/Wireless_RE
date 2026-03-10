# Based on the strategy mentioned in the paper Dynamic RIS partitioning in NOMA systems using deep reinforcement learning
import numpy as np


class SimpleRISEnv:
    """
    Python RL environment that wraps a simplified version of the MATLAB
    `ris_noma_channel.m` model.

    State  : [distance_user2, partition]
    Action : N1 (number of active RIS elements for U1), N2 = N - N1
    Reward : Jain's fairness index based on rates from the RIS-NOMA channel.
    """

    def __init__(self, N: int = 100, max_steps: int = 25, mc_samples: int = 10):
        # RIS configuration
        self.N = N  # total RIS elements
        self.max_steps = max_steps
        self.mc_samples = mc_samples  # Monte Carlo samples per step for ergodic rates

        # ---- System parameters (mirroring ris_noma_channel.m) ----
        self.fc = 1.8e9  # carrier frequency (Hz)
        self.c = 3e8
        self.lmbda = self.c / self.fc

        # Positions (2-D, meters)
        self.area = 20.0
        self.pos_BS = np.array([10.0, 0.0])
        self.pos_RIS = np.array([0.0, 10.0])
        self.pos_U1 = np.array([18.0, 18.0])  # fixed

        # Path loss
        self.PL_exp = 2.0

        # Noise
        self.BW = 20e6
        self.NF_dB = 10.0
        self.N0_dBm = -174 + 10 * np.log10(self.BW) + self.NF_dB
        self.N0 = 10 ** (self.N0_dBm / 10.0) * 1e-3  # W

        # RIS & transmit power
        self.Pt_dBm = 8.0
        self.Pt = 10 ** (self.Pt_dBm / 10.0) * 1e-3  # W
        self.alpha = 2.0
        self.sigma2_z = self.N0

        # Direction along which U2 moves (from RIS towards upper-right)
        v = np.array([1.0, 1.0])
        self.u2_dir = v / np.linalg.norm(v)

        self.reset()

    # ----------------------- RL interface ----------------------- #
    def reset(self):
        # user 2 radial distance from RIS (in meters, within the 20m x 20m area)
        self.dis_user2 = float(np.random.uniform(5.0, 20.0))
        self.pos_U2 = self.pos_RIS + self.u2_dir * self.dis_user2

        # initial RIS partition
        self.partition = int(np.random.randint(0, self.N + 1))  # N1 in [0, N]
        self.t = 0
        state = np.array([self.dis_user2, self.partition], dtype=float)
        return state

    def step(self, action: int):
        self.t += 1

        N1 = int(action)
        if N1 < 0:
            N1 = 0
        elif N1 > self.N:
            N1 = self.N
        N2 = self.N - N1

        # simple mobility: user 2 moves a bit each step (within the 20m x 20m area)
        self.dis_user2 += float(np.random.uniform(-0.5, 0.5))
        self.dis_user2 = float(np.clip(self.dis_user2, 5.0, 20.0))
        self.pos_U2 = self.pos_RIS + self.u2_dir * self.dis_user2

        # --- Realistic RIS-NOMA channel & rates (ported from ris_noma_channel.m) ---
        # Use Monte Carlo averaging over several channel realizations
        R1_vals = []
        R2_vals = []
        for _ in range(self.mc_samples):
            ch = self._generate_rayleigh_channels(N1, N2)
            sinr1, sinr2 = self._compute_sinr(ch, N1, N2)
            R1_vals.append(np.log2(1.0 + sinr1))
            R2_vals.append(np.log2(1.0 + sinr2))

        R1 = float(np.mean(R1_vals))
        R2 = float(np.mean(R2_vals))

        # Jain’s Fairness Index as reward
        reward = (R1 + R2) ** 2 / (2.0 * (R1**2 + R2**2))

        self.partition = N1
        state_next = np.array([self.dis_user2, self.partition], dtype=float)
        done = self.t >= self.max_steps
        return state_next, float(reward), done

    # ----------------------- Channel model helpers ----------------------- #
    def _path_loss(self, d: float) -> float:
        """Friis free-space path loss with exponent PL_exp."""
        return (self.lmbda / (4.0 * np.pi)) ** 2 * (d ** (-self.PL_exp))

    def _generate_rayleigh_channels(self, N1: int, N2: int):
        """
        Rough Python port of `generate_rayleigh_channels` from ris_noma_channel.m.
        Returns a dict with complex-valued channel vectors and phase shifts.
        """
        # Distances
        d_U1_RIS = float(np.linalg.norm(self.pos_U1 - self.pos_RIS))
        d_U2_RIS = float(np.linalg.norm(self.pos_U2 - self.pos_RIS))
        d_RIS_BS = float(np.linalg.norm(self.pos_RIS - self.pos_BS))

        # Path-loss-scaled Rayleigh fading
        def rayleigh_vec(n, d):
            if n <= 0:
                return np.zeros((0,), dtype=np.complex128)
            scale = np.sqrt(self._path_loss(d) / 2.0)
            return scale * (
                np.random.randn(n).astype(np.complex128)
                + 1j * np.random.randn(n).astype(np.complex128)
            )

        h1 = rayleigh_vec(N1, d_U1_RIS)  # U1 -> active
        h2 = rayleigh_vec(N1, d_U2_RIS)  # U2 -> active
        g1 = rayleigh_vec(N2, d_U1_RIS)  # U1 -> passive
        g2 = rayleigh_vec(N2, d_U2_RIS)  # U2 -> passive
        hBS = rayleigh_vec(N1, d_RIS_BS)  # active -> BS
        gBS = rayleigh_vec(N2, d_RIS_BS)  # passive -> BS

        # Optimal phase shifts
        if N1 > 0:
            theta_act_diag = np.exp(-1j * np.angle(h1 * hBS))
        else:
            theta_act_diag = np.zeros((0,), dtype=np.complex128)

        if N2 > 0:
            theta_pas_diag = np.exp(-1j * np.angle(g2 * gBS))
        else:
            theta_pas_diag = np.zeros((0,), dtype=np.complex128)

        return {
            "h1": h1,
            "h2": h2,
            "g1": g1,
            "g2": g2,
            "hBS": hBS,
            "gBS": gBS,
            "theta_act": theta_act_diag,
            "theta_pas": theta_pas_diag,
        }

    def _compute_sinr(self, ch, N1: int, N2: int):
        """
        Python port of `compute_sinr` from ris_noma_channel.m (Eq. (2) in the paper).
        Returns (sinr1, sinr2) in linear scale.
        """
        Pt = self.Pt
        alpha = self.alpha
        sigma2z = self.sigma2_z
        N0 = self.N0

        h1 = ch["h1"]
        h2 = ch["h2"]
        g1 = ch["g1"]
        g2 = ch["g2"]
        hBS = ch["hBS"]
        gBS = ch["gBS"]
        theta_act = ch["theta_act"]
        theta_pas = ch["theta_pas"]

        # Active/passive segments may be size 0; handle safely via slices.
        if N1 > 0:
            U1_via_active = alpha * np.sum(np.abs(h1[:N1]) * np.abs(hBS[:N1]))
            # amplifier noise term (mirroring MATLAB expression)
            noise_active = sigma2z * alpha * (
                np.sum(np.abs(theta_act[:N1]) * np.abs(hBS[:N1]))
            ) ** 2
        else:
            U1_via_active = 0.0
            noise_active = 0.0

        if N2 > 0:
            # Non-coherent for U1 on passive segment
            U1_via_passive = np.vdot(
                g1[:N2], theta_pas[:N2] * gBS[:N2]
            )  # g1^H * diag(theta_pas) * gBS
        else:
            U1_via_passive = 0.0

        if N1 > 0:
            # U2 through active segment (non-coherent for U2)
            U2_via_active = np.vdot(
                h2[:N1], theta_act[:N1] * hBS[:N1]
            )  # h2^H * diag(theta_act) * hBS
        else:
            U2_via_active = 0.0

        if N2 > 0:
            # Coherent for U2 on passive segment
            U2_via_passive = np.sum(np.abs(g2[:N2]) * np.abs(gBS[:N2]))
        else:
            U2_via_passive = 0.0

        # SINR for U1
        num1 = Pt * np.abs(U1_via_active + U1_via_passive) ** 2
        den1 = Pt * np.abs(U2_via_active + U2_via_passive) ** 2 + noise_active + N0
        sinr1 = float(num1 / den1) if den1 > 0 else 0.0

        # SINR for U2 (after SIC)
        num2 = Pt * np.abs(U2_via_active + U2_via_passive) ** 2
        den2 = noise_active + N0
        sinr2 = float(num2 / den2) if den2 > 0 else 0.0

        return sinr1, sinr2