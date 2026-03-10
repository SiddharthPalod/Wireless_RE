% =========================================================================
% RIS-aided NOMA System - Channel Generation and SINR Computation
% Based on: "Dynamic RIS partitioning in NOMA systems using DRL"
%           Gevez, Tek, Basar (2024), Front. Antennas Propag. 2:1418412
%
% System: Uplink, 2 single-antenna users -> single-antenna BS
%         Hybrid RIS with N1 active + N2 passive elements
%         Quasi-static frequency-flat Rayleigh fading (NLOS)
%         20m x 20m outdoor area @ 1.8 GHz
% =========================================================================

%% -------------------------  SYSTEM PARAMETERS  --------------------------
params.fc        = 1.8e9;        % Carrier frequency (Hz)
params.c         = 3e8;          % Speed of light (m/s)
params.lambda    = params.c / params.fc;  % Wavelength (m)

% Area & positions (2-D, ground level)
params.area      = 20;           % 20m x 20m
params.pos_BS    = [10,  0];     % Base Station position (m)
params.pos_RIS   = [ 0, 10];     % RIS position (m)  - fixed
params.pos_U1    = [18, 18];     % User 1 position (m) - fixed (TV broadcast unit)
params.pos_U2    = [13, 13];     % User 2 initial position (m) - mobile (drone)
                                 % Change pos_U2 to test different locations

% Path loss model: Friis free-space, exponent = 2
params.PL_exp    = 2;

% Noise
params.BW        = 20e6;         % Bandwidth (Hz)
params.NF_dB     = 10;           % Noise figure (dB)
params.N0_dBm    = -174 + 10*log10(params.BW) + params.NF_dB; % ~ -90 dBm
params.N0        = 10^(params.N0_dBm/10) * 1e-3;  % Noise power (W)

% RIS & transmit power
params.N         = 128;          % Total RIS elements (also try 256, 512)
params.Pt_dBm    = 8;            % Transmit power (dBm)
params.Pt        = 10^(params.Pt_dBm/10) * 1e-3;  % Transmit power (W)
params.alpha     = 2.0;          % Active element gain factor
params.sigma2_z  = params.N0;    % Active amplifier noise variance (same order as N0)

%% =====================  FUNCTION: generate_rayleigh_channels  ===========
% Returns a struct with all small-scale fading channel vectors + path loss.
% Call once per channel realization (i.e., inside your RL environment step).
%
% Output struct `ch` fields:
%   h1  [N1 x 1]  - U1  -> RIS active segment
%   h2  [N1 x 1]  - U2  -> RIS active segment
%   g1  [N2 x 1]  - U1  -> RIS passive segment
%   g2  [N2 x 1]  - U2  -> RIS passive segment
%   hBS [N1 x 1]  - RIS active segment -> BS
%   gBS [N2 x 1]  - RIS passive segment -> BS
%   hd1  scalar   - Direct U1 -> BS (if used; set 0 for NLOS-only)
%   hd2  scalar   - Direct U2 -> BS
% =========================================================================
function ch = generate_rayleigh_channels(N1, N2, params)
    % --- Path loss (Friis, exponent = 2) ---------------------------------
    PL = @(d) (params.lambda / (4*pi))^2 * d^(-params.PL_exp);

    d_U1_RIS = norm(params.pos_U1 - params.pos_RIS);
    d_U2_RIS = norm(params.pos_U2 - params.pos_RIS);
    d_RIS_BS = norm(params.pos_RIS - params.pos_BS);
    d_U1_BS  = norm(params.pos_U1 - params.pos_BS);
    d_U2_BS  = norm(params.pos_U2 - params.pos_BS);

    % --- Small-scale fading: CN(0,1) Rayleigh ----------------------------
    % Scale by sqrt(path_loss) so |h|^2 has mean = PL(d)
    ch.h1  = sqrt(PL(d_U1_RIS)/2) * (randn(N1,1) + 1j*randn(N1,1));  % U1->active
    ch.h2  = sqrt(PL(d_U2_RIS)/2) * (randn(N1,1) + 1j*randn(N1,1));  % U2->active
    ch.g1  = sqrt(PL(d_U1_RIS)/2) * (randn(N2,1) + 1j*randn(N2,1));  % U1->passive
    ch.g2  = sqrt(PL(d_U2_RIS)/2) * (randn(N2,1) + 1j*randn(N2,1));  % U2->passive
    ch.hBS = sqrt(PL(d_RIS_BS)/2) * (randn(N1,1) + 1j*randn(N1,1));  % active->BS
    ch.gBS = sqrt(PL(d_RIS_BS)/2) * (randn(N2,1) + 1j*randn(N2,1));  % passive->BS

    % Direct links (NLOS - blocked; set to 0 as in most RIS-NOMA papers)
    ch.hd1 = 0;
    ch.hd2 = 0;

    % --- Optimal phase shifts (coherent alignment, Eq. 1 of paper) -------
    % Active segment: align U1 -> RIS -> BS path
    ch.theta_act = diag( exp(-1j * angle(ch.h1 .* ch.hBS)) );  % [N1 x N1]

    % Passive segment: align U2 -> RIS -> BS path
    ch.theta_pas = diag( exp(-1j * angle(ch.g2 .* ch.gBS)) );  % [N2 x N2]
end

%% =====================  FUNCTION: compute_sinr  =========================
% Computes SINR for both users given a channel realization and partition.
%
% Inputs:
%   ch    - struct from generate_rayleigh_channels()
%   N1    - number of RIS elements allocated to U1 (active)
%   N2    - number of RIS elements allocated to U2 (passive)  N1+N2 = N
%   params - system parameters struct
%
% Outputs:
%   sinr1, sinr2 - scalar SINR values (linear, not dB)
%
% Equations follow Eq. (2) of the paper.
% =========================================================================
function [sinr1, sinr2] = compute_sinr(ch, N1, N2, params)
    Pt      = params.Pt;
    alpha   = params.alpha;
    sigma2z = params.sigma2_z;
    N0      = params.N0;

    % ---- U1 reflected signal through ACTIVE segment ---------------------
    % Coherent sum: alpha * sum_n1 |h1_n1| * |hBS_n1|  (after phase alignment)
    U1_via_active = alpha * sum(abs(ch.h1(1:N1)) .* abs(ch.hBS(1:N1)));

    % ---- U1 reflected signal through PASSIVE segment --------------------
    % Non-coherent (phase-aligned for U2, not U1): vector product
    U1_via_passive = ch.g1(1:N2).' * ch.theta_pas(1:N2,1:N2) * ch.gBS(1:N2);

    % ---- U2 reflected signal through ACTIVE segment ---------------------
    % Active segment is phase-aligned for U1, so U2 sees non-coherent sum
    U2_via_active = ch.h2(1:N1).' * ch.theta_act(1:N1,1:N1) * ch.hBS(1:N1);

    % ---- U2 reflected signal through PASSIVE segment --------------------
    % Coherent sum: sum_n2 |g2_n2| * |gBS_n2|  (after phase alignment)
    U2_via_passive = sum(abs(ch.g2(1:N2)) .* abs(ch.gBS(1:N2)));

    % ---- Active-segment amplifier noise power at BS ---------------------
    % sigma2_z * alpha * sum_n1 |theta_act_n1 * hBS_n1|^2
    noise_active = sigma2z * alpha * sum(abs(diag(ch.theta_act(1:N1,1:N1))) .* abs(ch.hBS(1:N1))).^2;

    % ---- SINR for U1 (Eq. 2 - numerator: U1 desired signal) ------------
    % U1 uses ACTIVE segment (stronger path due to amplification)
    % U2 signal acts as interference on U1's channel

    num1 = Pt * abs(U1_via_active + U1_via_passive)^2;
    den1 = Pt * abs(U2_via_active + U2_via_passive)^2 + noise_active + N0;

    sinr1 = num1 / den1;

    % ---- SINR for U2 (Eq. 2 - U2 uses PASSIVE segment only) ------------
    % After SIC: U1's signal is cancelled at BS for decoding U2
    num2 = Pt * abs(U2_via_active + U2_via_passive)^2;
    den2 = noise_active + N0;

    sinr2 = num2 / den2;
end

%% =====================  FUNCTION: compute_rates  ========================
% Thin wrapper to match your friend's Python interface exactly.
%   R1 = log2(1 + sinr1),  R2 = log2(1 + sinr2)   [bits/s/Hz]
% =========================================================================
function [R1, R2] = compute_rates(sinr1, sinr2)
    R1 = log2(1 + sinr1);
    R2 = log2(1 + sinr2);
end

%% =====================  QUICK SANITY-CHECK DEMO  ========================
% Run this section directly in MATLAB to verify output shapes & ballpark values.

N  = params.N;
N1 = 64;          % example partition: equal split
N2 = N - N1;      % 64 passive for U2

ch = generate_rayleigh_channels(N1, N2, params);
[sinr1, sinr2] = compute_sinr(ch, N1, N2, params);
[R1, R2]       = compute_rates(sinr1, sinr2);

fprintf('=== Single realization sanity check ===\n');
fprintf('N1 = %d (active, U1)   N2 = %d (passive, U2)\n', N1, N2);
fprintf('SINR1 = %.4f (%.2f dB)\n', sinr1, 10*log10(sinr1));
fprintf('SINR2 = %.4f (%.2f dB)\n', sinr2, 10*log10(sinr2));
fprintf('R1    = %.4f bits/s/Hz\n', R1);
fprintf('R2    = %.4f bits/s/Hz\n', R2);
fprintf('Rsum  = %.4f bits/s/Hz\n', R1 + R2);

%% =====================  MONTE CARLO ERGODIC RATE  =======================
% Average over 1e4 channel realizations (Section 5 methodology)

MC = 1e4;
R1_mc = zeros(MC,1);  R2_mc = zeros(MC,1);

for k = 1:MC
    ch_k = generate_rayleigh_channels(N1, N2, params);
    [s1, s2] = compute_sinr(ch_k, N1, N2, params);
    R1_mc(k) = log2(1 + s1);
    R2_mc(k) = log2(1 + s2);
end

fprintf('\n=== Monte Carlo (%d realizations) ===\n', MC);
fprintf('Ergodic R1   = %.4f bits/s/Hz\n', mean(R1_mc));
fprintf('Ergodic R2   = %.4f bits/s/Hz\n', mean(R2_mc));
fprintf('Ergodic Rsum = %.4f bits/s/Hz\n', mean(R1_mc + R2_mc));
