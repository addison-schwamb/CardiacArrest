%% Clean up the environment
clearvars;
close all;
clc;

%% Create data
mean_freq = 15; % Hz
sigma = 2;
freq_idx = 1:46;
r = freq_idx - mean_freq;
freq_vec = exp(-((freq_idx - mean_freq).^2)/(2*sigma));
% spec(:,1:700) = freq_vec.'*ones(1,700);
% spec = spec + 0.1*randn(46,700);

%% Create matrix of basis functions
eps = [1, 2];
num_widths = length(eps);
s = zeros(length(freq_idx),num_widths*length(freq_idx));
for i=1:num_widths*length(freq_idx)
    cur_eps = eps(ceil(i/length(freq_idx)));
    cur_mean = mod(i,length(freq_idx));
    s(:,i) = exp(-((freq_idx - cur_mean).^2)/(2*cur_eps));
end

%% LASSO
l = lasso(s,freq_vec);

%% Plotting
figure;
plot(freq_idx,freq_vec)

figure;
imagesc(s)
axis xy