# eval_supervised.py
import numpy as np, h5py, torch
from math import atan2
from torch import nn

data = h5py.File('dataset.mat','r')
channels = data['channels']
h_DnR = np.array(channels['h_DnR']).T
h_DfR = np.array(channels['h_DfR']).T
h_RSn = np.array(channels['h_RSn']).T
h_RSf = np.array(channels['h_RSf']).T
h_SR  = np.array(channels['h_SR']).T
h_RT  = np.array(channels['h_RT']).T
labels = np.array(data['labels']['best_phase']).T
params = data['params']
L = int(np.array(params['L']))

# load model
ckpt = torch.load('phasenet.pth', map_location='cpu')
# reconstruct model architecture same as training
in_dim = np.concatenate([np.real(h_DnR), np.imag(h_DnR),
                    np.real(h_DfR), np.imag(h_DfR),
                    np.real(h_RSn), np.imag(h_RSn),
                    np.real(h_RSf), np.imag(h_RSf),
                    np.real(h_SR),  np.imag(h_SR),
                    np.real(h_RT),  np.imag(h_RT)], axis=1).shape[1]

class PhaseNet(nn.Module):
    def __init__(self, in_dim, out_dim, HIDDEN=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN//2),
            nn.ReLU(),
            nn.Linear(HIDDEN//2, out_dim)
        )
    def forward(self,x):
        return self.net(x)

model = PhaseNet(in_dim, 2*L)
model.load_state_dict(ckpt['model_state'])
model.eval()

scaler_mean = ckpt['scaler_mean']; scaler_scale = ckpt['scaler_scale']

# Build X
X = np.concatenate([np.real(h_DnR), np.imag(h_DnR),
                    np.real(h_DfR), np.imag(h_DfR),
                    np.real(h_RSn), np.imag(h_RSn),
                    np.real(h_RSf), np.imag(h_RSf),
                    np.real(h_SR),  np.imag(h_SR),
                    np.real(h_RT),  np.imag(h_RT)], axis=1)
X = (X - scaler_mean)/scaler_scale

X_torch = torch.from_numpy(X).float()
with torch.no_grad():
    pred = model(X_torch).numpy()

pred_cos = pred[:, :L]
pred_sin = pred[:, L:]
pred_theta = np.arctan2(pred_sin, pred_cos)  # shape (N,L)

# Now compute WSR for predicted phases and compare to labels
# Use same parameters from .mat
w_n = float(np.array(params['w_n'])); w_f = float(np.array(params['w_f']))
PL_Dn = float(np.array(params['PL_Dn'])); PL_Df = float(np.array(params['PL_Df'])); PL_d_T = float(np.array(params['PL_d_T']))
Pn = float(np.array(params['Pn'])); Pf = float(np.array(params['Pf']))
noise_power = float(np.array(params['noise_power']))
gamma_const_sq = float(np.array(params['gamma_const_sq']))
BW = float(np.array(params['BW']))

def compute_wsr_for_sample(i, theta_vec):
    # reconstruct complex channels for sample i
    h_DnR_i = h_DnR[i,:]; h_DfR_i = h_DfR[i,:]
    h_RSn_i = h_RSn[i,:]; h_RSf_i = h_RSf[i,:]
    h_SR_i = h_SR[i,:]; h_RT_i = h_RT[i,:]
    Phi = np.exp(1j*theta_vec)
    h_Dn = h_DnR_i * Phi * h_RSn_i
    h_Df = h_DfR_i * Phi * h_RSf_i
    h_T1 = h_RT_i * Phi * h_SR_i
    h_T2 = h_SR_i * Phi * h_RT_i
    h_T = h_T1 * h_T2
    h_Dn_abs2 = np.abs(np.sum(h_Dn))**2
    h_Df_abs2 = np.abs(np.sum(h_Df))**2
    h_T_abs2  = (np.abs(np.sum(h_T))**2) * gamma_const_sq * BW**2 * (1/(8*np.pi**2*(np.sqrt(gamma_const_sq)*BW/(2*np.pi))**2*(T:=1e-6)*BW*(10**(20/10)*1e-3)/noise_power))
    Dn_power = Pn * PL_Dn * h_Dn_abs2
    Df_power = Pf * PL_Df * h_Df_abs2
    Echo_power = (10**(20/10)*1e-3) * PL_d_T * h_T_abs2
    SINR_Df = Df_power / (Dn_power + Echo_power + noise_power)
    SINR_Dn = Dn_power / (Echo_power + noise_power)
    Rn = np.log2(1 + SINR_Dn); Rf = np.log2(1 + SINR_Df)
    return w_n*Rn + w_f*Rf

N = pred_theta.shape[0]
wsr_pred = np.zeros(N)
wsr_label = np.zeros(N)
for i in range(N):
    wsr_pred[i] = compute_wsr_for_sample(i, pred_theta[i,:])
    # label phases from file
    wsr_label[i] = compute_wsr_for_sample(i, labels[i,:])

print('Average WSR (label):', np.mean(wsr_label))
print('Average WSR (predicted):', np.mean(wsr_pred))
