%% Clean up the environment
clearvars;
close all;

%% Parameters
% Here are all the parameters you might want to change
filename = 'CA (12)\JA4221OG.edf'; % Name of the file to use

% Basis function params
freq_steps = 46; % Number of radial basis functions for frequency
eps_freq = [1, 2, 5, 10, 50]; % Width(s) of the frequency Gaussians
time_steps = 700; % Number of radial basis functions for time
eps_time = [10, 20, 50, 100, 500]; % Width(s) of the time Gaussians

% LASSO params
err = 0.01; % Used for finding the optimal lambda value out of the ones returned
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

%% LASSO (frequency)
% Lasso across frequency
w_freq = zeros(length(T),num_freq_widths*freq_steps);
for i=1:length(T)
    [l,FitInfo] = lasso(s_freq,log10(S(i,freq_idx)));
    MSE_below_err = FitInfo.MSE(FitInfo.MSE<err);
    w_freq(i,:) = l(:,length(MSE_below_err)).';
end

% Post-processing frequency
w_freq_3d = reshape(w_freq,length(T),freq_steps,[]);
peak_locs_freq = sum(w_freq_3d,3);

top = round(0.08*size(peak_locs_freq,1)*size(peak_locs_freq,2));
[~,ind] = maxk(peak_locs_freq(:),top);
new_peaks = zeros(length(T),freq_steps);
new_peaks(ind) = peak_locs_freq(ind);

%% LASSO (time)
% Lasso across time
% w_time = zeros(num_time_widths*time_steps,num_freq_widths*freq_steps);
% for i=1:num_freq_widths*freq_steps
%     [l,FitInfo] = lasso(s_time,w_freq(time_idx,i));
%     MSE_below_err = FitInfo.MSE(FitInfo.MSE<err);
%     try
%         w_time(:,i) = l(:,length(MSE_below_err)).';
%     catch
%         w_time(:,i) = l(:,1).';
%     end
% end
% 
%% Post-processing
% w = reshape(w_time,num_time_widths*time_steps,freq_steps,[]);
% peak_locs = sum(w,3);
% peak_locs = reshape(peak_locs.',freq_steps,time_steps,[]);
% peak_locs = sum(peak_locs,3).';
% 
% top = round(0.01*size(peak_locs,1)*size(peak_locs,2));
% [~,ind] = maxk(peak_locs(:),top);
% new_peaks = zeros(time_steps,freq_steps);
% new_peaks(ind) = peak_locs(ind);

%% Plotting
% figure;
% imagesc(peak_locs.');
% axis xy
% colorbar

figure;
imagesc(peak_locs_freq.');
axis xy
colorbar

figure;
imagesc(new_peaks.');
axis xy