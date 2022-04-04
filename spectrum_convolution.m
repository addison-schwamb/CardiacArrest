%% Clean up the environment
clearvars;
close all;
clc;

%% Import Data
filename = 'CA (12)\JA4221OG.edf';
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
figure;
plot_matrix(S,(T./60),F);
caxis([-10 50]);

%% Convolve
basis = exp(-((-2:400:2).^2)/2);
w = conv(basis, log10(S(27,:)));

%% Plotting
figure;
plot(F,log10(S(27,:)));

figure;
plot(w)