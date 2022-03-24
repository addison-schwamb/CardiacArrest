%% Clean up the environment
clearvars;
close all;
clc;

%% Create data
mean_freq_1 = 15; % Hz
mean_freq_2 = 3;
sigma_1 = 3;
sigma_2 = 1;
freq_idx = 1:46;
freq_vec = 0.75*exp(-((freq_idx - mean_freq_1).^2)/(2*sigma_1)) + exp(-((freq_idx - mean_freq_2).^2)/(2*sigma_2));
spec(:,1:700) = freq_vec.'*ones(1,700);
spec = spec + 0.1*randn(46,700);

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
err = 0.01;
w = zeros(700,num_widths*length(freq_idx));
for i=1:700
    [l,FitInfo] = lasso(s,spec(freq_idx,i));
    MSE_below_err = FitInfo.MSE(FitInfo.MSE<err);
    l_vec = l(:,length(MSE_below_err)).';
    w(i,:) = l(:,length(MSE_below_err)).';
end

%% Postprocessing
w = reshape(w,700,length(freq_idx),[]);
peak_locs = sum(w,3);

%% Plotting
figure;
plot(freq_idx,freq_vec)

figure;
imagesc(spec)
axis xy

figure;
imagesc(peak_locs.');
axis xy

figure;
subplot(1,2,1);
imagesc(w(:,:,1).');
axis xy

subplot(1,2,2);
imagesc(w(:,:,2).');
axis xy

figure;
imagesc((w(:,:,1)*s(:,1:46).'));
axis xy