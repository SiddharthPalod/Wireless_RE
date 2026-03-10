%% data_generator.m
% Generates dataset for ML: channel realizations + best phase vector (by random search)
% Output: dataset.mat containing variables:
%   channels: struct with fields h_DnR, h_DfR, h_RSn, h_RSf, h_SR, h_RT (num_samples x L complex)
%   labels: best_phase (num_samples x L) in [0,2pi)
%   best_wsr: best WSR found (num_samples x 1)
%   params: struct with constants (L, BW, gamma_const_sq, sigma_tau2, PL_*, P_BS_range, Pn,Pf,w_n,w_f,noise_power)

clc; clear; close all;

% -----------------------
% Parameters (match your sim)
num_trials = 1000;     % number of samples to generate (reduce/increase as needed)
perl = 120;
L = 3*perl;
eta = 1;

BW = 10e06; fc = 5.8e09; c = 3e8;
lambda = c/fc;
sigma_RCS = 50;
Cc = (c/(4*pi*fc))^2;
Cr = (sigma_RCS*lambda^2)/(4*pi)^3;
gamma_const_sq = ((2*pi)^2)/12;
delta = 0.01;
T = 1e-6;
T_bit = T/delta;
B_rms = sqrt(gamma_const_sq)*BW/(2*pi);

% sample system constants (same as your sim)
d_SR = 100; d_RT = 110; d_RDn = 500; d_RDf = 650;
alpha_c = 2.5; alpha_r = 4;
Gc = 10^(30/10);

mu_SR  = Gc * Cc * d_SR^(-alpha_c);
mu_RS = mu_SR;
mu_RT  = Cr * d_RT^(-alpha_r);
mu_TR = mu_RT;
mu_DnR = Gc * Cc * d_RDn^(-alpha_c);
mu_DfR = Gc * Cc * d_RDf^(-alpha_c);
PL_d_T = (mu_SR*mu_RT*mu_TR*mu_RS);
PL_Dn = mu_DnR*mu_RS;
PL_Df = mu_DfR*mu_RS;

P_BS_dBm = 20;
P_BS_range = 10.^(P_BS_dBm/10).*10^-3;
P_dBm = 10;          % choose a training operating power (you can store multiple powers)
Pn = 10^(P_dBm/10)*1e-3;
P_del_dB = 2;
Pf = Pn * 10^(-P_del_dB/10);

w_n = 0.6; w_f = 0.4;

T_temp = 290; kB = 1.38e-23;
noise_power = kB*T_temp*BW;

% Nakagami settings
m = 3; omega = 1;
pd = makedist('Nakagami','mu',m,'omega',omega);

% Random search settings for label generation
N_rand_search = 200;   % number of random-phase candidates per sample (increase for stronger labels)

% Preallocate
channels.h_DnR = zeros(num_trials, L);
channels.h_DfR = zeros(num_trials, L);
channels.h_RSn = zeros(num_trials, L);
channels.h_RSf = zeros(num_trials, L);
channels.h_SR  = zeros(num_trials, L);
channels.h_RT  = zeros(num_trials, L);

labels.best_phase = zeros(num_trials, L);
labels.best_wsr = zeros(num_trials, 1);

rng('default')

for n = 1:num_trials
    % generate channels (same distributions as your sim)
    h_SR = random(pd,1,L).*exp(1j*2*pi*rand(1,L));
    h_RT = random(pd,1,L).*exp(1j*2*pi*rand(1,L));
    h_RSn = random(pd,1,L).*exp(1j*2*pi*rand(1,L));
    h_RSf = random(pd,1,L).*exp(1j*2*pi*rand(1,L));
    h_DnR = random(pd,1,L).*exp(1j*2*pi*rand(1,L));
    h_DfR = random(pd,1,L).*exp(1j*2*pi*rand(1,L));
    
    channels.h_DnR(n,:) = h_DnR;
    channels.h_DfR(n,:) = h_DfR;
    channels.h_RSn(n,:) = h_RSn;
    channels.h_RSf(n,:) = h_RSf;
    channels.h_SR(n,:)  = h_SR;
    channels.h_RT(n,:)  = h_RT;
    
    best_wsr = -inf;
    best_phase = zeros(1,L);
    
    for k = 1:N_rand_search
        theta = 2*pi*rand(1,L);     % random phase candidate
        Phi = eta * exp(1j*theta);
        
        % cascaded channels
        h_Dn = h_DnR .* Phi .* h_RSn;
        h_Df = h_DfR .* Phi .* h_RSf;
        h_T1 = h_RT .* Phi .* h_SR;
        h_T2 = h_SR .* Phi .* h_RT; % symmetric usage OK for label generation
        h_T = h_T1 .* h_T2;
        
        % effective powers (sum over RIS)
        h_Dn_abs2 = abs(sum(h_Dn)).^2;
        h_Df_abs2 = abs(sum(h_Df)).^2;
        h_T_abs2  = (abs(sum(h_T)).^2) * gamma_const_sq * BW^2 * (1/(8*pi^2*B_rms^2*(T*BW*P_BS_range/noise_power)));
        
        % received powers
        Dn_power = Pn * PL_Dn * h_Dn_abs2;
        Df_power = Pf * PL_Df * h_Df_abs2;
        Echo_power = P_BS_range * PL_d_T * h_T_abs2;
        
        SINR_Df = Df_power / (Dn_power + Echo_power + noise_power);
        SINR_Dn = Dn_power / (Echo_power + noise_power);
        
        Rn = log2(1 + SINR_Dn);
        Rf = log2(1 + SINR_Df);
        wsr = w_n * Rn + w_f * Rf;
        
        if wsr > best_wsr
            best_wsr = wsr;
            best_phase = theta;
        end
    end
    
    labels.best_phase(n,:) = best_phase;
    labels.best_wsr(n) = best_wsr;
    
    if mod(n,50)==0
        fprintf('Generated %d/%d samples\n', n, num_trials)
    end
end

% Save dataset
params.L = L; params.BW = BW; params.gamma_const_sq = gamma_const_sq;
params.PL_Dn = PL_Dn; params.PL_Df = PL_Df; params.PL_d_T = PL_d_T;
params.Pn = Pn; params.Pf = Pf; params.P_BS_range = P_BS_range;
params.w_n = w_n; params.w_f = w_f; params.noise_power = noise_power;

save('dataset.mat', 'channels', 'labels', 'params', '-v7.3');
fprintf('Dataset saved to dataset.mat\n')
