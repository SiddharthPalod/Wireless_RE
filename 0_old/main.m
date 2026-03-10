clc;
clear all;
close all;
% -------------------------------------------------------------------------
% DRDO NOMA-ISAC with RIS (Uplink)
% MODIFIED:
%  - Removed RIS partitioning
%  - Random RIS phase shift
%  - Added Weighted Sum Rate formulation
% -------------------------------------------------------------------------

BW = 10e06;
fc = 5.8e09;
T_temp = 290;
kB = 1.38*10^-23;
noise_power = kB*T_temp*BW;

c = 3*10^8;
lambda = c/fc;
num_trials = 10^4;

sigma_RCS = 50;
delta = 0.01;
T = 1e-06;

Gc_dBi = 30;
Gc = 10^(Gc_dBi/10);
Gr_dBi = 30;
Gr = 10^(Gr_dBi/10);

Cc = (c/(4*pi*fc))^2;
Cr = (sigma_RCS*lambda^2)/(4*pi)^3;

gamma_const_sq = ((2*pi)^2)/12;
gamma_const = sqrt(gamma_const_sq);

T_bit = T/delta;
B_rms = gamma_const*BW/(2*pi);

P_BS_dBm = 20;
P_BS_range = 10.^(P_BS_dBm/10).*10^-3;

ISNR = T*BW*P_BS_range/noise_power;
sigma_tau2 = 1/(8*pi^2*B_rms^2*ISNR);

alpha_c = 2.5;
alpha_r = 4;

% -------------------------------------------------------------------------
% Distances
d_SR = 100;
d_RT = 110;
elevation = 50;

d_RDn = 500;
d_RDf = 650;

% -------------------------------------------------------------------------
% Large scale fading
mu_SR  = Gc * Cc * d_SR^(-alpha_c);
mu_RS = mu_SR;
mu_RT  = Cr * d_RT^(-alpha_r);
mu_TR = mu_RT;
mu_DnR = Gc * Cc * d_RDn^(-alpha_c);
mu_DfR = Gc * Cc * d_RDf^(-alpha_c);

PL_d_T = (mu_SR*mu_RT*mu_TR*mu_RS);
PL_Dn = mu_DnR*mu_RS;
PL_Df = mu_DfR*mu_RS;

% -------------------------------------------------------------------------
% RIS SPECIFICATIONS — NO PARTITION
perl = 120;
L = 3*perl;
eta = 1;

% -------------------------------------------------------------------------
% Channel Components (same naming kept)
m = 3; omega = 1;
pd = makedist('Nakagami','mu',m,'omega',omega);

h_SR = random(pd,num_trials,L) .* exp(1j*2*pi*rand(num_trials,L));
h_RSt = (h_SR').';

h_RT = random(pd,num_trials,L) .* exp(1j*2*pi*rand(num_trials,L));
h_TR = (h_RT').';

h_RSn = random(pd,num_trials,L) .* exp(1j*2*pi*rand(num_trials,L));
h_RSf = random(pd,num_trials,L) .* exp(1j*2*pi*rand(num_trials,L));

h_DnR = random(pd,num_trials,L) .* exp(1j*2*pi*rand(num_trials,L));
h_DfR = random(pd,num_trials,L) .* exp(1j*2*pi*rand(num_trials,L));

% -------------------------------------------------------------------------
% RANDOM RIS PHASE SHIFT (instead of optimal)
theta_rand = 2*pi*rand(num_trials,L);

% -------------------------------------------------------------------------
% Combined channel definition (same style)
h_Df = (h_DfR .* eta .* exp(1j*theta_rand) .* h_RSf);
h_Dn = (h_DnR .* eta .* exp(1j*theta_rand) .* h_RSn);

h_T1 = (h_RT .* eta .* exp(1j*theta_rand) .* h_SR);
h_T2 = (h_RSt .* eta .* exp(1j*theta_rand) .* h_TR);

h_T = h_T1 .* h_T2;

% -------------------------------------------------------------------------
% Channel power across RIS
h_Df_abs_2 = (abs(sum(h_Df,2)).^2);
h_Dn_abs_2 = (abs(sum(h_Dn,2)).^2);
h_T_abs_2  = (abs(sum(h_T,2)).^2) .* gamma_const_sq .* BW^2 .* sigma_tau2;

% Average channel power gain
mu_Df = mean(abs(sum(h_Df,2)).^2);
mu_Dn = mean(abs(sum(h_Dn,2)).^2);
mu_BS = mean(abs(sum(h_T,2)).^2);

% -------------------------------------------------------------------------
% Power sweep
P_dBm = 0:0.5:20;
P = 10.^(P_dBm/10).*10^-3;

P_del_dB = 2;
P_del1 = 10^(-P_del_dB/10);

% WSR weights
w_n = 0.6;
w_f = 0.4;

for pow = 1:length(P)

    Pn = P(pow);
    Pf = P(pow)*P_del1;
    P_BS = P_BS_range;

    % Received powers
    Dn_power = Pn .* PL_Dn .* h_Dn_abs_2;
    Df_power = Pf .* PL_Df .* h_Df_abs_2;
    Echo_power = P_BS .* PL_d_T .* h_T_abs_2;

    % SINR (uplink NOMA)
    SINR_Df = Df_power ./ (Dn_power + Echo_power + noise_power);
    SINR_Dn = Dn_power ./ (Echo_power + noise_power);

    % Rates
    R_Dn = mean(log2(1 + SINR_Dn));
    R_Df = mean(log2(1 + SINR_Df));

    % Weighted Sum Rate
    WSR(pow) = w_n*R_Dn + w_f*R_Df;

    Result_WSR(pow,1) = 10*log10(Pn*1e3);
    Result_WSR(pow,2) = WSR(pow);

end

% -------------------------------------------------------------------------
% Plot
figure();
plot(Result_WSR(:,1), Result_WSR(:,2),'LineWidth',2);
grid on;
xlabel('Near user transmit power (dBm)');
ylabel('Weighted Sum Rate (bps/Hz)');
title('WSR with Random RIS Phase — No Partition');
