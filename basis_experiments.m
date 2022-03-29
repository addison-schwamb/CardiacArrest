%% Clean up the environment
clearvars;
close all;
clc;

%% Create data
mean_freq_1 = 15; % Middle of each peak (Hz)
mean_freq_2 = 3; 
sigma_1 = 3; % Width of each peak (Hz)
sigma_2 = 1;

freq_idx = 1:46; % Frequency range to plot
% Create a single timepoint with two Gaussian peaks at the specified
% frequencies
freq_vec = 0.75*exp(-((freq_idx - mean_freq_1).^2)/(2*sigma_1)) + exp(-((freq_idx - mean_freq_2).^2)/(2*sigma_2));

% Create a spectrogram of the two peaks across time
% In this case the frequency vector is just multiplied by a vector of ones
% so it is the same for all time
spec(:,1:700) = freq_vec.'*ones(1,700);
% Add a little bit of noise
spec = spec + 0.1*randn(46,700);

%% Create matrix of basis functions
eps = [1, 2]; % widths of gaussians to match
num_widths = length(eps); % number of different widths to match
s = zeros(length(freq_idx),num_widths*length(freq_idx)); % empty matrix to hold all of the basis functions

% fill the empty matrix with all of the basis functions
for i=1:num_widths*length(freq_idx)
    % determine current width and mean - width varies based on how many
    % desired widths there are (set above), and mean is incremented each
    % loop (then resets when there's a new width)
    cur_eps = eps(ceil(i/length(freq_idx)));
    cur_mean = mod(i,length(freq_idx));
    
    % create a Gaussian radial basis function and assign to the current
    % column of s
    s(:,i) = exp(-((freq_idx - cur_mean).^2)/(2*cur_eps));
end

%% LASSO
err = 0.01; % amount of acceptable error
w = zeros(700,num_widths*length(freq_idx)); % empty matrix to hold the weights of the basis functions
% do this for each time point
for i=1:700
    [l,FitInfo] = lasso(s,spec(freq_idx,i)); % use LASSO algorithm to find weights
    MSE_below_err = FitInfo.MSE(FitInfo.MSE<err); % find the weightings with less error than err
    w(i,:) = l(:,length(MSE_below_err)).'; % set the weight of that row to be the most sparse vector that still has an acceptable error level
end

%% Postprocessing
w = reshape(w,700,length(freq_idx),[]); % reshape so that there's one matrix for each width of basis function
peak_locs = sum(w,3); % sum across the 3rd dimension - this will give us the center frequencies of all the peaks across time

%% Plotting

% Plot a line of the initial data
figure;
plot(freq_idx,freq_vec)
xlabel('Frequency (Hz)')
ylabel('Power (arbitrary)')
title('Two Gaussians, Used to Create Spectrogram')

% Plot the spectrogram of the data
figure;
imagesc(spec)
axis xy
xlabel('Time (min)')
ylabel('Frequency (Hz)')
c = colorbar;
c.Label.String = 'Power (arbitrary)';
title('Spectrogram of Generated Data');


figure;
imagesc(peak_locs.');
axis xy
xlabel('Time (min)')
ylabel('Frequency (Hz)')
c = colorbar;
c.Label.String = 'Power (arbitrary)';
title('Centers of Detected Peaks');

figure;
subplot(1,2,1);
imagesc(w(:,:,1).');
axis xy
xlabel('Time (min)')
ylabel('Frequency (Hz)')
c = colorbar;
c.Label.String = 'Power (arbitrary)';

subplot(1,2,2);
imagesc(w(:,:,2).');
axis xy
xlabel('Time (min)')
ylabel('Frequency (Hz)')
c = colorbar;
c.Label.String = 'Power (arbitrary)';
title('Centers of Peaks 1 Hz Wide (left) and 2 Hz Wide (right)');