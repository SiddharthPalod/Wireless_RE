clc;
clear all;
close all;
% B LOHINI PRIYANKA, PH2023502
% 12-May-2025
%---------------------------------------------------------------------------
% Implementing DRDO NOMA-ISAC proposal with RIS (Uplink)
% Two users - (far and near) and one point Target considered
% Base Paper: L. Sun et al "On the Study of Non-Orthogonal Multiple Access
% (NOMA)-Assisted Integrated Sensing and Communication (ISAC),"
% in IEEE Transactions on Communications,
% vol. 72, no. 11, pp. 7278-7293, Nov. 2024, doi: 10.1109/TCOMM.2024.3407202.

% Chao Zhang, "Semi-Integrated-Sensing-and-Communication (Semi-ISaC): From
% OMA to NOMA", IEEE TRANSACTIONS ON COMMUNICATIONS, VOL. 71, NO. 4, APRIL
% 2023.
%---------------------------------------------------------------------------
BW = 10e06;                  % Bandwidth
fc = 5.8e09;                 % Center Frequency
T_temp = 290;                % Absolute Temperature (Kelvin) 310K (273.15 +37)
kB = 1.38*10^-23;            % Boltzmann Constant (in Joules/K)
noise_power = kB*T_temp*BW;  % Thermal noise power (Watts)
noise_power_dBm = 10*log10(noise_power*10^3);   % Convert W → dBm

c = 3*10^8;             % Speed of Light
lambda = c/fc;          % Wavelength
num_trials = 10^4;      % No. of Monte Carlo runs
sigma_RCS = 50;         % Sensing RADAR cross section in meter^2
delta = 0.01;
T = 1e-06;              % Pulse Duration
Gc_dBi = 30;
Gc = 10^(Gc_dBi/10);
Gr_dBi = 30; % to 10dB
Gr = 10^(Gr_dBi/10);

Cc = (c/(4*pi*fc))^2;
Cr = (sigma_RCS*lambda^2)/(4*pi)^3;
gamma_const_sq = ((2*pi)^2)/12;
gamma_const = sqrt(gamma_const_sq);

T_bit = T/delta;
B_rms = gamma_const*BW/(2*pi);
P_BS_dBm = 20;
P_BS_range = 10.^(P_BS_dBm/10).*10^-3; % in W
ISNR = T*BW*P_BS_range/noise_power;
sigma_tau2 = 1/(8 * pi^2 * B_rms^2 * ISNR);

P_del_dB = [2];
m_SR = 3;
m_RS = 3;
m_RT = 3;
m_TR = 3;
m_DnR = 3;
m_DfR = 3;

m = 3;omega = 1;
Q = @(x) 0.5 * erfc(x / sqrt(2));

alpha_c = 2.5;
alpha_r = 4;
R_c = 1;
gamma_th = 1;
gamma_SIC = 0.4;
gamma_th_r = 0.4;

%---------------------------------------------------------------------------
% Scenario I - Close communication users Dn at 500 m and Df at 800 m,
% Distant radar target at 1000 m
d_SR = 100;
d_RS = d_SR;
elevation = 50;
d_RT =  110;
x_dist = sqrt(d_RT^2 - elevation^2);
d_ST = d_SR + x_dist;

d_RDn = 500;
d_RDf = 650;

d_TR = d_RT;

n_dist = sqrt(d_RDn^2 - elevation^2);
d_SDn = d_SR + n_dist;

f_dist = sqrt(d_RDf^2 - elevation^2);
d_SDf = d_SR + f_dist;


%---------------------------------------------------------------------------
% LARGE SCALE FADING COEFFICIENT
mu_SR  = Gc * Cc * d_SR^(-alpha_c);
mu_RS = mu_SR;
mu_RT  = Cr * d_RT^(-alpha_r);
mu_TR = mu_RT;
mu_DnR = Gc * Cc * d_RDn^(-alpha_c);
mu_DfR = Gc * Cc * d_RDf^(-alpha_c);


% %% EFFECTIVE LARGE SCALE FADING COEFFICIENT FOR CASCADED CHANNEL
PL_d_T = (mu_SR*mu_RT*mu_TR*mu_RS);
PL_Dn = mu_DnR*mu_RS;
PL_Df = mu_DfR*mu_RS;

% %% SCALING PARAMETER = 1
omega_SR = 1;
omega_RS = 1;
omega_RT = 1;
omega_TR = 1;
omega_DnR = 1;
omega_DfR = 1;

%---------------------------------------------------------------------------
% RIS SPECIFICATIONS
perl = 120;
L = 3*perl;
eta = 1;     % Refelction coefficient of each RU (Reflecting Unit)
RU = L/3;    % RIS equal partitioing

%  Defining Zones for the 3 EQUAL paritition
zf = 1:RU;          % Far user
zn = RU+1:2*RU;     % Near user
zT = (2*RU)+1:3*RU; % Target


%---------------------------------------------------------------------------

% Define Channel Components
pd = makedist('Nakagami','mu',m,'omega',omega);


h_SR = random(pd,num_trials,length(zT)) .* exp(1j * 2 * pi * rand(num_trials, length(zT)));
h_SR_zn = random(pd,num_trials,length(zn)) .* exp(1j * 2 * pi * rand(num_trials, length(zn)));
h_SR_zf = random(pd,num_trials,length(zf)) .* exp(1j * 2 * pi * rand(num_trials, length(zf)));

h_RSt = (h_SR').';h_RSt_zn = (h_SR_zn').';h_RSt_zf = (h_SR_zf').';

h_RT = random(pd,num_trials,length(zT)) .* exp(1j * 2 * pi * rand(num_trials, length(zT)));
h_RT_zn = random(pd,num_trials,length(zn)) .* exp(1j * 2 * pi * rand(num_trials, length(zn)));
h_RT_zf = random(pd,num_trials,length(zf)) .* exp(1j * 2 * pi * rand(num_trials, length(zf)));

h_TR = (h_RT').';h_TR_zn = (h_RT_zn').';h_TR_zf = (h_RT_zf').';

h_RSn = random(pd,num_trials,length(zn)) .* exp(1j * 2 * pi * rand(num_trials, length(zn)));
h_RSn_zf = random(pd,num_trials,length(zf)) .* exp(1j * 2 * pi * rand(num_trials, length(zf)));
h_RSn_zT = random(pd,num_trials,length(zT)) .* exp(1j * 2 * pi * rand(num_trials, length(zT)));

h_RSf = random(pd,num_trials,length(zf)) .* exp(1j * 2 * pi * rand(num_trials, length(zf)));
h_RSf_zn = random(pd,num_trials,length(zn)) .* exp(1j * 2 * pi * rand(num_trials, length(zn)));
h_RSf_zT = random(pd,num_trials,length(zT)) .* exp(1j * 2 * pi * rand(num_trials, length(zT)));

h_DnR = random(pd,num_trials,length(zn)) .* exp(1j * 2 * pi * rand(num_trials, length(zn)));
h_DnR_zf = random(pd,num_trials,length(zf)) .* exp(1j * 2 * pi * rand(num_trials, length(zf)));
h_DnR_zT = random(pd,num_trials,length(zT)) .* exp(1j * 2 * pi * rand(num_trials, length(zT)));

h_DfR = random(pd,num_trials,length(zf)) .* exp(1j * 2 * pi * rand(num_trials, length(zf)));
h_DfR_zn = random(pd,num_trials,length(zn)) .* exp(1j * 2 * pi * rand(num_trials, length(zn)));
h_DfR_zT = random(pd,num_trials,length(zT)) .* exp(1j * 2 * pi * rand(num_trials, length(zT)));


% Conversion to polar coordinates
[theta_h_RSn, rho_h_RSn] = cart2pol(real(h_RSn), imag(h_RSn));
[theta_h_RSf, rho_h_RSf] = cart2pol(real(h_RSf), imag(h_RSf));
[theta_h_SR, rho_h_SR] = cart2pol(real(h_SR), imag(h_SR));
[theta_h_RSt, rho_h_RSt] = cart2pol(real(h_RSt), imag(h_RSt));
[theta_h_RT, rho_h_RT] = cart2pol(real(h_RT), imag(h_RT));
[theta_h_TR, rho_h_TR] = cart2pol(real(h_TR), imag(h_TR));
[theta_h_RDn, rho_h_RDn] = cart2pol(real(h_DnR), imag(h_DnR));
[theta_h_RDf, rho_h_RDf] = cart2pol(real(h_DfR), imag(h_DfR));


% Optimal choice of RIS phaseshifts
theta_zf = -(theta_h_RSf + theta_h_RDf);
theta_zn = -(theta_h_RSn + theta_h_RDn);
theta_zT1 = -(theta_h_SR + theta_h_RT);
theta_zT2 = -(theta_h_TR + theta_h_RSt);

theta_zn_rand = 2*pi*rand(1,length(zn));
theta_zf_rand = 2*pi*rand(1,length(zf));
theta_zT2_rand = 2*pi*rand(1,length(zT));

% Combined channel definition
% Each channel below has dim = num_trials X RIS elements
h_Df = (h_DfR.*eta.*exp(1j*theta_zf).*h_RSf);
h_Dn = (h_DnR.*eta.*exp(1j*theta_zn).*h_RSn);
h_T1 = (h_RT.*eta.*exp(1j*theta_zT1).*h_SR);
h_T2 = (h_RSt.*eta.*exp(1j*theta_zT2).*h_TR);


% Combined Channel from other RIS zones - Near User
h_Dn_zf = (h_DnR_zf.*eta.*exp(1j*theta_zf_rand).*h_RSn_zf);
h_Dn_zT = (h_DnR_zT.*eta.*exp(1j*theta_zT2_rand).*h_RSn_zT);

% Combined Channel from other RIS zones - Far User
h_Df_zn = (h_DfR_zn.*eta.*exp(1j*theta_zn_rand).*h_RSf_zn);
h_Df_zT = (h_DfR_zT.*eta.*exp(1j*theta_zT2_rand).*h_RSf_zT);

% Combined Channel from other BS to RIS zones
h_T1_zn = (h_RT_zn.*eta.*exp(1j*theta_zn_rand).*h_SR_zn);
h_T1_zf = (h_RT_zf.*eta.*exp(1j*theta_zf_rand).*h_SR_zf);

% Combined Channel from other RIS zones - Target Echo
h_T2_zn = (h_RSt_zn.*eta.*exp(1j*theta_zn_rand).*h_TR_zn);
h_T2_zf = (h_RSt_zf.*eta.*exp(1j*theta_zf_rand).*h_TR_zf);

% Onward and Echo combined
h_T    = h_T1 .* h_T2;
h_T_zn = h_T1_zn .* h_T2_zn;
h_T_zf = h_T1_zf .* h_T2_zf;

% Channel power across all RIS elements
h_Df_abs_2 = (abs(sum(h_Df,2)).^2);
h_Dn_abs_2 = (abs(sum(h_Dn,2)).^2);
h_T1_abs_2 = (abs(sum(h_T1,2)).^2);
h_T2_abs_2 = (abs(sum(h_T2,2)).^2);
h_T_abs_2 = h_T1_abs_2 .* h_T2_abs_2 .* gamma_const_sq .* BW^2 .* sigma_tau2;
g_r_abs_2 = h_T_abs_2;


h_Dn_zf_abs_2 = (abs(sum(h_Dn_zf,2)).^2);
h_Dn_zT_abs_2 = (abs(sum(h_Dn_zT,2)).^2);
h_Df_zn_abs_2 = (abs(sum(h_Df_zn,2)).^2);
h_Df_zT_abs_2 = (abs(sum(h_Df_zT,2)).^2);
h_T2_zn_abs_2 = (abs(sum(h_T2_zn,2)).^2);
h_T2_zf_abs_2 = (abs(sum(h_T2_zf,2)).^2);

% Effective channel envelope
h_Df_Eff_alt = sum(h_Df,2);
h_Dn_Eff_alt = sum(h_Dn,2);
h_T_Eff_alt = sum(h_T,2);

h_Dn_zf_Eff_alt = sum(h_Dn_zf,2);
h_Dn_zT_Eff_alt = sum(h_Dn_zT,2);
h_Df_zn_Eff_alt = sum(h_Df_zn,2);
h_Df_zT_Eff_alt = sum(h_Df_zT,2);
h_T2_zf_Eff_alt = sum(h_T2_zf,2);
h_T2_zn_Eff_alt = sum(h_T2_zn,2);

% Extracting the real part as imaginary part is 0
h_Dn_env = real(h_Dn_Eff_alt);
h_Df_env = real(h_Df_Eff_alt);
h_T1_env = real(sum(h_T1,2));
h_T2_env = real(sum(h_T2,2));

h_Dn_zf_env = abs(h_Dn_zf_Eff_alt);
h_Dn_zT_env = abs(h_Dn_zT_Eff_alt);
h_Df_zn_env = abs(h_Df_zn_Eff_alt);
h_Df_zT_env = abs(h_Df_zT_Eff_alt);

% Average channel power gain
mu_Df = mean(abs(h_Df_Eff_alt).^2); % Avg.channel power gain at far user
mu_Dn = mean(abs(h_Dn_Eff_alt).^2); % Avg.channel power gain at near user
mu_BS  = mean(abs(h_T_Eff_alt).^2); % Avg.channel power gain at BS

%%% -------------------------------------
% Far user to BS channels defined via 3 different RIS zones
% h_Df    - Far user cascased channel via zf zone (REAL)
% h_Df_zn - Far user cascased channel via zn zone (COMPLEX)
% h_Df_zT - Far user cascased channel via zT zone (COMPLEX)

h_Df_allz_Eff = h_Df_Eff_alt + h_Df_zn_Eff_alt + h_Df_zT_Eff_alt;
h_Df_allz_abs_2 = (abs(h_Df_allz_Eff)).^2;

h_Dn_allz_Eff = h_Dn_Eff_alt + h_Dn_zf_Eff_alt + h_Dn_zT_Eff_alt;
h_Dn_allz_abs_2 = (abs(h_Dn_allz_Eff)).^2;

% Target echo channel defined via 3 different RIS zones
% Onward and echo chanel power is computed as per eqn(7) of Semi-ISAC paper
% h_T1_abs_2    = Onward sensing channel power from BS to Target via zT zone
% h_T2_abs_2    = Reflected echo channel power from Target to BS via zT zone
% h_T2_zn_abs_2 = Reflected echo channel power from Target to BS via zn zone
% h_T2_zf_abs_2 = Reflected echo channel power from Target to BS via zf zone

h_T_Eff    = sum(h_T,2);
h_T_zn_Eff = sum(h_T_zn,2);
h_T_zf_Eff = sum(h_T_zf,2);

h_T_allz_Eff = (h_T_Eff + h_T_zn_Eff + h_T_zf_Eff);
h_T_allz_abs_2 = (abs(h_T_allz_Eff).^2) .* gamma_const_sq .* BW^2 .* sigma_tau2;


% % % % % % % ---------------------------------------------------------------------------
% % % Total power at BS = Communication power + Sensing power

P_dBm = 0:0.5:20;
P = 10.^(P_dBm/10).*10^-3; % in W
P_del1 = 10^(-P_del_dB/10);

% Looping for different Total Power

for pow = 1:1:length(P)
    pow

    Pn = P(pow);
    Pf = P(pow)*P_del1;
    P_BS = P_BS_range;


    Tx_SNR_Dn = Pn/noise_power;
    Tx_SNR_Df = Pf/noise_power;
    Tx_SNR_BS = P_BS/noise_power;

    Rx_SNR_Dn = PL_Dn .* mu_Dn.*Tx_SNR_Dn;
    Rx_SNR_Df = PL_Df .* mu_Df.*Tx_SNR_Df;
    Rx_SNR_BS = PL_d_T .* mu_BS.*Tx_SNR_BS;
    %---------------------------------------------------------------------------
    % SINR COMPUTATION
    % imperfect SIC factor
    %     delta_n = 0.5;
    %     delta_f = 0.5;

    delta_n = 0;
    delta_f = 0;

    % Each user and target's received signal power
    Dn_power = Pn.* PL_Dn .* h_Dn_abs_2;
    Df_power_Exact = Pf.* PL_Df .* h_Df_allz_abs_2;

    Echo_power = P_BS.* PL_d_T .*h_T_abs_2;
    Echo_power_Exact = P_BS.* PL_d_T .*h_T_allz_abs_2;
    E_IR = mean(Echo_power_Exact);
    E_IR_Th = P_BS.*PL_d_T.* gamma_const_sq .* BW^2 .* sigma_tau2;

    SINR_Dn = (Dn_power)./(Df_power_Exact + E_IR_Th + noise_power);
    
    %%SUCCESS PROBABILITY
    E1 = SINR_Dn > gamma_th;
    Succ_Dn = mean(E1);
    Dn_outage_SIM = 1 - Succ_Dn;

    %%------------------------------------------------------------------------------
    % %% Update Result Matrix
    %     Result(pow,1) = 10*log10(Rx_SNR_Dn * 10^3);
    Result(pow,1) = 10*log10(Pn * 10^3);
    Result(pow,2) = Dn_outage_SIM;


end

figure();
semilogy(Result(:,1), Result(:,2), 'b -', 'linewidth',2.0,'MarkerFaceColor','b','MarkerSize',10.0); hold on;grid on;
label('Near user transmit power, $P_n~(\mathrm{dBm})$', 'Interpreter', 'latex');
ylabel('Near user outage probability , $P_{out}^{Dn}$', 'Interpreter', 'latex');
