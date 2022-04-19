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

recon_spec = zeros([length(S) length(F)-1]);
for spec_ind = 1:length(S)
    % Convolve decreasing exponential
    
    width = 1;
    prev_err = 0;
    add = 0;
    inc = 0.1;
    err_diff = 100;
    eps = 0.001; % For best results, eps should be about 1/100 of inc
    
    while abs(err_diff) > eps
        basis = exp(-F(2:end)/width);
        [w,lags] = xcorr(log10(S(spec_ind,:)),basis);
        [~,ind] = max(w);
        shift = lags(ind);
        
        s = exp(-(F(2:end) - F(shift+1))/width).';
        [l,FitInfo] = lasso(s,log10(S(spec_ind,2:end)));
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
    
    reduced_data = log10(S(spec_ind,2:end)) - (s*l(:,1)).';
    fit_err = prev_err;
    
    % Convolve Gaussians
    width = [0.1, 1, 10, 100];
    prev_err = 0;
    add = 0;
    inc = [0.1, 1, 10, 100];
    eps = [0.001, 0.01, 0.1, 1]; % For best results, eps should be about 1/100 of inc
    i = 2;
    j = 1;
    breakout = 0;

    while fit_err > 0.1
        if breakout
            break
        end
        breakout = 0;
        width = [0.1, 1, 10, 100];
        err_diff = 100;
        while abs(err_diff) > eps(mod(j-1,length(width))+1)
            basis = exp(-(F(2:end) - 0.5*F(end)).^2/(2*width(mod(j-1,length(width))+1)));
            [w,lags] = xcorr(reduced_data,basis);
            if isempty(find(w~=0,1))
                breakout = 1;
                break
            end
            [~,ind] = max(abs(w));
            shift = lags(ind);
            if shift < 0
                shift = 0;
            elseif shift > length(F)/2
                shift = length(F)/2;
            end
            %F((length(F)/2)+shift+1)
            
            s(:,i) = exp(-(F(2:end) - F((length(F)/2)+shift+1)).^2/(2*width(mod(j-1,length(width))+1)));
            [l,FitInfo] = lasso(s,log10(S(spec_ind,2:end)));
            err_diff = FitInfo.MSE(1) - prev_err;
        
            if err_diff > 0
                if add
                    width(mod(j-1,length(width))+1) = width(mod(j-1,length(width))+1) - inc(mod(j-1,length(width))+1);
                    add = 0;
                else
                    width(mod(j-1,length(width))+1) = width(mod(j-1,length(width))+1) + inc(mod(j-1,length(width))+1);
                    add = 1;
                end
            elseif err_diff < 0
                if add
                    width(mod(j-1,length(width))+1) = width(mod(j-1,length(width))+1) + inc(mod(j-1,length(width))+1);
                else
                    width(mod(j-1,length(width))+1) = width(mod(j-1,length(width))+1) - inc(mod(j-1,length(width))+1);
                end
            end
            prev_err = FitInfo.MSE(1);
        end
        
        reduced_data = log10(S(spec_ind,2:end)) - sum(l(:,1).*s.');
        fit_err = prev_err;
        i = i+1;
        j = j+1;
    end
    recon_spec(i,:) = sum(l(:,1).*s.');
end
%% Plotting
figure;
plot_matrix(recon_spec,(T./60),F(2:end));
% figure;
% plot(F,log10(S(spec_ind,:)));
% hold on
% plot(F(2:end),sum(l(:,1).*s.'));
% 
% figure;
% plot(FitInfo.MSE);
% 
% figure;
% plot(F(2:end),sum(l(:,1).*s.'));
% 
% figure;
% plot(F(2:end), reduced_data)