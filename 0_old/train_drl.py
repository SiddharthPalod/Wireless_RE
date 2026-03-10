# train_drl.py
import numpy as np, h5py
from env_ris import RISEnv
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import gym

# Load dataset.mat
f = h5py.File('dataset.mat','r')
channels = {
    'h_DnR': np.array(f['channels']['h_DnR']).T,
    'h_DfR': np.array(f['channels']['h_DfR']).T,
    'h_RSn': np.array(f['channels']['h_RSn']).T,
    'h_RSf': np.array(f['channels']['h_RSf']).T,
    'h_SR':  np.array(f['channels']['h_SR']).T,
    'h_RT':  np.array(f['channels']['h_RT']).T
}
params = {
    'gamma_const_sq': float(np.array(f['params']['gamma_const_sq'])),
    'BW': float(np.array(f['params']['BW'])),
    'T': 1e-6,
    'P_BS': float(np.array(f['params']['P_BS_range'])),
    'Pn': float(np.array(f['params']['Pn'])),
    'Pf': float(np.array(f['params']['Pf'])),
    'PL_Dn': float(np.array(f['params']['PL_Dn'])),
    'PL_Df': float(np.array(f['params']['PL_Df'])),
    'PL_d_T': float(np.array(f['params']['PL_d_T'])),
    'w_n': float(np.array(f['params']['w_n'])),
    'w_f': float(np.array(f['params']['w_f'])),
    'noise_power': float(np.array(f['params']['noise_power']))
}
f.close()

env = RISEnv(channels, params)
# action noise for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1*np.ones(n_actions))

model = TD3('MlpPolicy', env, verbose=1, action_noise=action_noise, buffer_size=100000, learning_starts=1000)
model.learn(total_timesteps=500000)  # tune timesteps based on compute
model.save('td3_ris')

# After training, evaluate
