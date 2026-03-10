# env_ris.py
import gym
from gym import spaces
import numpy as np

class RISEnv(gym.Env):
    def __init__(self, channels, params):
        super().__init__()
        self.channels = channels  # dict of arrays (N x L)
        self.params = params
        self.N = channels['h_DnR'].shape[0]
        self.L = channels['h_DnR'].shape[1]
        self.idx = 0

        # observation: real & imag concatenation for all base channels for current sample
        obs_dim = 12 * self.L  # same as supervised input
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        # action: continuous, per-element in [-1,1] (we map to [0,2pi])
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.L,), dtype=np.float32)

    def reset(self):
        self.idx = np.random.randint(0, self.N)
        return self._get_obs(self.idx)

    def step(self, action):
        # map action to phases
        theta = (action + 1.0) * np.pi  # [-1,1] -> [0,2pi]
        wsr = self._compute_wsr(self.idx, theta)
        reward = wsr  # you may scale reward
        done = True  # single-step episode (stateless per sample)
        info = {'wsr': wsr}
        return self._get_obs(self.idx), reward, done, info

    def _get_obs(self, i):
        h_DnR = self.channels['h_DnR'][i]
        h_DfR = self.channels['h_DfR'][i]
        h_RSn = self.channels['h_RSn'][i]
        h_RSf = self.channels['h_RSf'][i]
        h_SR  = self.channels['h_SR'][i]
        h_RT  = self.channels['h_RT'][i]
        obs = np.concatenate([np.real(h_DnR), np.imag(h_DnR),
                              np.real(h_DfR), np.imag(h_DfR),
                              np.real(h_RSn), np.imag(h_RSn),
                              np.real(h_RSf), np.imag(h_RSf),
                              np.real(h_SR),  np.imag(h_SR),
                              np.real(h_RT),  np.imag(h_RT)])
        return obs.astype(np.float32)

    def _compute_wsr(self, i, theta):
        # compute WSR similarly as earlier
        h_DnR = self.channels['h_DnR'][i]
        h_DfR = self.channels['h_DfR'][i]
        h_RSn = self.channels['h_RSn'][i]
        h_RSf = self.channels['h_RSf'][i]
        h_SR  = self.channels['h_SR'][i]
        h_RT  = self.channels['h_RT'][i]
        Phi = np.exp(1j*theta)
        h_Dn = h_DnR * Phi * h_RSn
        h_Df = h_DfR * Phi * h_RSf
        h_T1 = h_RT * Phi * h_SR
        h_T2 = h_SR * Phi * h_RT
        h_T = h_T1 * h_T2
        h_Dn_abs2 = np.abs(np.sum(h_Dn))**2
        h_Df_abs2 = np.abs(np.sum(h_Df))**2
        h_T_abs2  = (np.abs(np.sum(h_T))**2) * self.params['gamma_const_sq'] * self.params['BW']**2 * (1/(8*np.pi**2*(np.sqrt(self.params['gamma_const_sq'])*self.params['BW']/(2*np.pi))**2*(self.params['T'])*self.params['BW']*(self.params['P_BS']/self.params['noise_power'])))
        Dn_power = self.params['Pn'] * self.params['PL_Dn'] * h_Dn_abs2
        Df_power = self.params['Pf'] * self.params['PL_Df'] * h_Df_abs2
        Echo_power = self.params['P_BS'] * self.params['PL_d_T'] * h_T_abs2
        SINR_Df = Df_power / (Dn_power + Echo_power + self.params['noise_power'])
        SINR_Dn = Dn_power / (Echo_power + self.params['noise_power'])
        Rn = np.log2(1 + SINR_Dn)
        Rf = np.log2(1 + SINR_Df)
        return self.params['w_n'] * Rn + self.params['w_f'] * Rf
