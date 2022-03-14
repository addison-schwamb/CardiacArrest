%% Clean up the environment
clearvars;
close all;

%% Import and Preprocessing
filename = 'CA (6)\JA4221NO.edf';
[data,header] = import_data(filename);

EEG_data = bipolarMontage(data,header);

clc;

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
eps1 = 2;
freq_idx = linspace(1,1844,46);
freq_idx = round(freq_idx);
s1 = zeros(length(freq_idx));

for i=1:length(freq_idx)
    for j=1:length(freq_idx)
        r = norm(F(freq_idx(j))-F(freq_idx(i)));
        s1(i,j) = exp(-(eps1*r)^2);
    end
end

% Time basis functions
eps2 = 10;
time_idx = linspace(1,7196,700);
time_idx = round(time_idx);
s2 = zeros(length(time_idx));

for i=1:length(time_idx)
    for j=1:length(time_idx)
        r = norm(T(time_idx(j))-T(time_idx(i)));
        s2(i,j) = exp(-(eps2*r)^2);
    end
end

%% LASSO (frequency-time)
% Lasso across frequency
w_init = zeros(7196,length(freq_idx));
for i=1:7196
    l = lasso(s1,log10(S(i,freq_idx)));
    w_init(i,:) = l(:,1).';
end

% Lasso across time
w = zeros(length(time_idx),length(freq_idx));
for i=1:length(freq_idx)
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