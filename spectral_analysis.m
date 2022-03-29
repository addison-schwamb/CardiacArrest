%% Clean up the environment
clearvars;
close all;

%% Parameters
% Here are all the parameters you might want to change
filename = 'CA (12)\JA4221OG.edf'; % Name of the file to use

% Basis function params
freq_steps = 46; % Number of radial basis functions for frequency
eps_freq = [1, 2, 5, 10]; % Width(s) of the frequency Gaussians
time_steps = 350; % Number of radial basis functions for time
eps_time = [10, 20, 50, 100, 500]; % Width(s) of the time Gaussians

% LASSO params
err_freq = 0.01; % Used for finding the optimal lambda value out of the ones returned
err_time = 0.01;
%% Import and Preprocessing
[data,header] = import_data(filename);
EEG_data = bipolarMontage(data,header);

clc; % clear all the text dump from EEGLab import

%% Signal Processing and Spectrograms
% MT spectrogram parameters
params.Fs = 200;
params.tapers = [3 5];
params.fpass = [0 45];

% Loop through all channels
ch = 12;
oneChannel = EEG_data(ch,:);
oneChannel = oneChannel - mean(oneChannel);
[S,T,F] = mtspecgramc(oneChannel,[30 6],params);
plot_matrix(S,(T./60),F);
caxis([-10 50]);

%% Basis function setup
% Frequency basis function
num_freq_widths = length(eps_freq);
freq_idx = linspace(1,length(F),freq_steps);
freq_idx = round(freq_idx);
s_freq = zeros(freq_steps,num_freq_widths*freq_steps);

for i=1:num_freq_widths*freq_steps
    cur_eps = eps_freq(ceil(i/freq_steps)); % the width of the current basis function
    cur_mean = freq_idx(mod(i-1,freq_steps)+1); % the index for the center (mean) of the current basis function
    s_freq(:,i) = exp(-((F(freq_idx) - F(cur_mean)).^2)/(2*cur_eps));
end

% Time basis functions
num_time_widths = length(eps_time);
time_idx = linspace(1,length(T),time_steps);
time_idx = round(time_idx);
s_time = zeros(time_steps,num_time_widths*time_steps);

for i=1:num_time_widths*time_steps
    cur_eps = eps_time(ceil(i/time_steps)); % the width of the current basis function
    cur_mean = time_idx(mod(i-1,time_steps)+1); % the index for the center (mean) of the current basis function
    s_time(:,i) = exp(-((T(time_idx) - T(cur_mean)).^2)/(2*cur_eps));
end

%% LASSO
% Lasso across frequency
w_freq = lasso_matrix(log10(S),s_freq,err_freq,freq_idx,1);

% Lasso across time
w_time = lasso_matrix(w_freq,s_time,err_time,time_idx,2);

%% Post-processing
peak_locs_freq = reshape(w_time,num_time_widths*time_steps,freq_steps,[]);
peak_locs_freq = sum(peak_locs_freq,3);

peak_locs_ft = reshape(peak_locs_freq.',freq_steps,time_steps,[]);
peak_locs_ft = sum(peak_locs_ft,3).';

%% Plotting
figure;
imagesc(peak_locs_ft.');
axis xy
colorbar
