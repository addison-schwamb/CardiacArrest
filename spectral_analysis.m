%% Clean up the environment
clearvars;
close all;

%% Parameters
% Here are all the parameters you might want to change
filename = 'CA (12)\JA4221OG.edf'; % Name of the file to use

% Basis function params
freq_steps = 46; % Number of radial basis functions for frequency
eps1 = 2; % Width of the frequency Gaussians
time_steps = 700; % Number of radial basis functions for time
eps2 = 10; % Width of the time Gaussians

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
freq_idx = linspace(1,length(F),freq_steps);
freq_idx = round(freq_idx);
s1 = zeros(freq_steps);

for i=1:freq_steps
    for j=1:freq_steps
        r = norm(F(freq_idx(j))-F(freq_idx(i)));
        s1(i,j) = exp(-(eps1*r)^2);
    end
end

% Time basis functions
time_idx = linspace(1,length(T),time_steps);
time_idx = round(time_idx);
s2 = zeros(time_steps);

for i=1:time_steps
    for j=1:time_steps
        r = norm(T(time_idx(j))-T(time_idx(i)));
        s2(i,j) = exp(-(eps2*r)^2);
    end
end

%% LASSO (frequency-time)
% Lasso across frequency
w_init = zeros(length(T),freq_steps);
for i=1:length(T)
    l = lasso(s1,log10(S(i,freq_idx)));
    w_init(i,:) = l(:,1).';
end

% Lasso across time
w = zeros(time_steps,freq_steps);
for i=1:freq_steps
    l = lasso(s2,w_init(time_idx,i));
    w(:,i) = l(:,1).';
end

%% Plotting
figure;
imagesc(w_init.');
axis xy
colorbar

figure;
imagesc(w.');
axis xy
colorbar