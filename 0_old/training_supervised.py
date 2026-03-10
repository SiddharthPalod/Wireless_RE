# train_supervised.py
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# ---- Hyperparams
BATCH = 64
EPOCHS = 60
LR = 1e-3
HIDDEN = 1024   # adjust
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- Load dataset
f = h5py.File('dataset.mat','r')
# navigate structure: dataset.channels.h_DnR etc may be stored as MATLAB structs in h5py with weird keys
channels = f['channels']
h_DnR = np.array(channels['h_DnR']).T  # shape: (num_samples, L)
h_DfR = np.array(channels['h_DfR']).T
h_RSn = np.array(channels['h_RSn']).T
h_RSf = np.array(channels['h_RSf']).T
h_SR  = np.array(channels['h_SR']).T
h_RT  = np.array(channels['h_RT']).T

labels = f['labels']
best_phase = np.array(labels['best_phase']).T  # (num_samples, L)
best_wsr = np.array(labels['best_wsr']).squeeze()

params = f['params']
L = int(np.array(params['L']))
w_n = float(np.array(params['w_n']))
w_f = float(np.array(params['w_f']))
PL_Dn = float(np.array(params['PL_Dn']))
PL_Df = float(np.array(params['PL_Df']))
PL_d_T = float(np.array(params['PL_d_T']))
Pn = float(np.array(params['Pn']))
Pf = float(np.array(params['Pf']))
noise_power = float(np.array(params['noise_power']))
gamma_const_sq = float(np.array(params['gamma_const_sq']))
BW = float(np.array(params['BW']))

f.close()

# ---- Preprocess - create real stacked inputs: concat real and imag of each channel
X = np.concatenate([np.real(h_DnR), np.imag(h_DnR),
                    np.real(h_DfR), np.imag(h_DfR),
                    np.real(h_RSn), np.imag(h_RSn),
                    np.real(h_RSf), np.imag(h_RSf),
                    np.real(h_SR),  np.imag(h_SR),
                    np.real(h_RT),  np.imag(h_RT)], axis=1)

# target: represent phases as sin and cos
Y_sin = np.sin(best_phase)
Y_cos = np.cos(best_phase)
Y = np.concatenate([Y_cos, Y_sin], axis=1)

# normalize X
scaler = StandardScaler()
X = scaler.fit_transform(X)

# dataset object
class RISDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

dataset = RISDataset(X, Y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH, shuffle=False)

# ---- Model
class PhaseNet(nn.Module):
    def __init__(self, in_dim, out_dim):
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

in_dim = X.shape[1]
out_dim = 2*L  # cos and sin per element
model = PhaseNet(in_dim, out_dim).to(device)
opt = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# ---- Training
for ep in range(EPOCHS):
    model.train()
    loss_sum = 0
    for xb, yb in train_loader:
        xb = xb.to(device); yb = yb.to(device)
        opt.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * xb.size(0)
    train_loss = loss_sum / len(train_loader.dataset)
    
    # val
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb)
            val_loss += criterion(pred, yb).item() * xb.size(0)
    val_loss /= len(val_loader.dataset)
    print(f'Epoch {ep+1}/{EPOCHS} train_loss={train_loss:.6f} val_loss={val_loss:.6f}')

# ---- Save model and scaler
torch.save({'model_state': model.state_dict(), 'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_}, 'phasenet.pth')
print('Saved phasenet.pth')
