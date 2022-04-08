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

%% Convolve decreasing exponential

width = 1;
prev_err = 0;
add = 0;
inc = 0.1;
err_diff = 100;
eps = 0.001; % For best results, eps should be about 1/100 of inc

while abs(err_diff) > eps
    basis = exp(-F(2:end)/width);
    [w,lags] = xcorr(log10(S(27,:)),basis);
    [~,ind] = max(w);
    shift = lags(ind);
    
    s = exp(-(F(2:end) - F(shift+1))/width).';
    [l,FitInfo] = lasso(s,log10(S(27,2:end)));
    err_diff = FitInfo.MSE(1) - prev_err;

    if err_diff > 0
        if add
            width = width - inc;
            add = 0;
        else
            width = width + inc;
            add = 1;
        end
    elseif err_diff < 0
        if add
            width = width + inc;
        else
            width = width - inc;
        end
    end
    prev_err = FitInfo.MSE(1);
end

reduced_data = log10(S(27,2:end)) - (s*l(:,1)).';

%% Convolve std normal Gaussian
width = 0.1;
prev_err = 0;
add = 0;
inc = 0.1;
err_diff = 100;
eps = 0.001; % For best results, eps should be about 1/100 of inc

while abs(err_diff) > eps
    basis = exp(-(F(2:end) - 0.5*F(end)).^2/(2*width));
    [w,lags] = xcorr(reduced_data,basis);
    [~,ind] = max(w);
    shift = lags(ind);
    %F((length(F)/2)+shift+1)
    
    s(:,2) = exp(-(F(2:end) - F((length(F)/2)+shift+1)).^2/(2*width));
    [l,FitInfo] = lasso(s,log10(S(27,2:end)));
    err_diff = FitInfo.MSE(1) - prev_err;

    if err_diff > 0
        if add
            width = width - inc;
            add = 0;
        else
            width = width + inc;
            add = 1;
        end
    elseif err_diff < 0
        if add
            width = width + inc;
        else
            width = width - inc;
        end
    end
    prev_err = FitInfo.MSE(1);
end

reduced_data = log10(S(27,2:end)) - sum(l(:,1).*s.');

%% Convolve std normal Gaussian
width = 10;
prev_err = 0;
add = 0;
inc = 5;
err_diff = 100;
eps = 0.01; % For best results, eps should be about 1/100 of inc

while abs(err_diff) > eps
    basis = exp(-(F(2:end) - 0.5*F(end)).^2/(2*width));
    [w,lags] = xcorr(reduced_data,basis);
    [~,ind] = max(w);
    shift = lags(ind);
    %F((length(F)/2)+shift+1)
    
    s(:,3) = exp(-(F(2:end) - F((length(F)/2)+shift+1)).^2/(2*width));
    [l,FitInfo] = lasso(s,log10(S(27,2:end)));
    err_diff = FitInfo.MSE(1) - prev_err;

    if err_diff > 0
        if add
            width = width - inc;
            add = 0;
        else
            width = width + inc;
            add = 1;
        end
    elseif err_diff < 0
        if add
            width = width + inc;
        else
            width = width - inc;
        end
    end
    prev_err = FitInfo.MSE(1);
end

reduced_data = log10(S(27,2:end)) - sum(l(:,1).*s.');

%% Plotting
figure;
plot(F,log10(S(27,:)));
hold on
plot(F(2:end),sum(l(:,1).*s.'));

figure;
plot(FitInfo.MSE);

figure;
plot(F(2:end),sum(l(:,1).*s.'));

figure;
plot(F(2:end), reduced_data)