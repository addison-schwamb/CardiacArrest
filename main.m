%% Clean up the environment
close all;
clear all;
clc;

%% Set parameters
% Patient info
pt_num = 7;
arrest_start = 475.10;
epoch_len = 20; % length of time to examine (minutes before and minutes during/after);

% Filepath definition
path = 'C:\Users\a.l.schwamb\Box\Cardiac Arrest (Pre-CA) project\CA EDF\';
file = strcat('CA',int2str(pt_num),'-DONE\CA',int2str(pt_num),'.edf');
locs = readlocs('C:/Users/a.l.schwamb/Documents/MATLAB/Libraries/eeglab2021.1/sample_locs/BPM.locs'); % For plotting headmaps

% Window params
win_length = 3; % window length in minutes
win_shift = 0.5; % window shift in minutes

% Channel params (for visualization & EKG)
ch = 1;
ekg_ch = 27;

% Fit params
fit_params.batchSz = 10; % Number of predictions/backprops per model parameter update
fit_params.nBatch = 4000; % Number of gradient updates

% NADAM algorithm params
nadam_params.W_rate = 2.5*10^-5; % Learning rate on WS gradient updates
% nadam_params.W12_rate = 1*10^-6;
nadam_params.xi_rate = 1.25*10^-4;
nadam_params.D2_rate = 1;
nadam_params.mu = 0.9;
nadam_params.nu = 0.95;
nadam_params.W_eps = 0.15;
% nadam_params.W12_eps = 0.15;
nadam_params.xi_eps = 0.2;
nadam_params.D2_eps = 200;
nadam_params.lambda1 = 0.075;
nadam_params.lambda2 = 0.2;
nadam_params.lambda3 = 0.05;
nadam_params.lambda4 = 0.05;
nadam_params.mxi = 0;
nadam_params.mW = 0;
% nadam_params.mW1 = 0;
% nadam_params.mW2 = 0;
nadam_params.mD2 = 0;
nadam_params.nxi = 0;
nadam_params.nW = 0;
% nadam_params.nW1 = 0;
% nadam_params.nW2 = 0;
nadam_params.nD2 = 0;

% Initial model params
alpha_randScale = 0.75;
W_randScale = 0.1;
% W12_randScale = 0.1;
D_randScale = 2.5;
m = 9;
nRec = 100;

%% Import & preprocessing
% Load data
header = pop_biosig(strcat(path,file));
data = header.data;
Fs = header.srate;
clc; % clear text dump from EEGlab import

% Preprocessing
EEG_data = bipolarMontage(data,header); % montage data
EEG_data = EEG_data - mean(EEG_data,2); % subtract mean
EEG_data = EEG_data./mad(EEG_data, 2); % divide by mean absolute deviation

% Create time vector for length of whole dataset
t_end = length(data)/(Fs*60); % end time (minutes)
t_span = 1/(Fs*60):1/(Fs*60):t_end; % time of each sample (minutes)

% Calculate epoch of interest
epoch_start = max(arrest_start-epoch_len,0);
epoch_end = min(arrest_start+epoch_len,t_end);


%% Spectrogram & heartrate plot
% Spectrogram specific parameters
spec_params.Fs = header.srate;
spec_params.tapers = [3 5];
spec_params.fpass = [0 45];

% EEG spectrogram
oneChannel = EEG_data(ch,:);
[S,T,F] = mtspecgramc(oneChannel,[30 6],spec_params);

% Calculate heartrate from EKG
% Take spectrogram of heartrate

hr_params.Fs = header.srate;
hr_params.tapers = [3 5];
hr_params.fpass = [0.25 3];
hrChannel = data(ekg_ch,:);
[S_hr,T_hr,F_hr] = mtspecgramc(hrChannel,[30 6],hr_params);

% Index with max power should give heartrate frequency
[~,f_ind] = max(S_hr,[],2);
hr = F_hr(f_ind).*60;

% Plot EEG spectrogram
figure;
subplot(2,1,1);
plot_matrix(S,(T./60),F);

% Adjust axes & coloring
xlim([epoch_start, epoch_end]);
caxis([-60 -10]);
line([arrest_start arrest_start],[0 45],'Color','r','LineWidth',1);
title({'Spectrogram', strcat('Patient ',int2str(pt_num))});

% Plot heartrate
subplot(2,1,2);
plot(T_hr./60, hr);
xlim([epoch_start, epoch_end]);
line([arrest_start arrest_start],[0 max(hr)],'Color','r','LineWidth',1);
xlabel('time (min)');
ylabel('heartrate (bpm)');
title('Heartrate from EKG');

%% Define initial model params
nX = size(EEG_data,1);
model_params.b = 20/3;
model_params.alpha = 0.1 + sqrt(alpha_randScale*(1 + abs(randn))) + (model_params.b)^2/4;
model_params.xi = model_params.b/sqrt(model_params.alpha^2 + 0.25);
model_params.W = W_randScale*randn(nX);
% model_params.W1 = W12_randScale*randn(nX,m);
% model_params.W2 = W12_randScale*randn(nX,m);
model_params.Dmin = 0.1;
model_params.D2 = 1.75 + sqrt(abs(D_randScale*randn(nX,1)));

%% Fit Model
vidName = strcat('topoVid_pt',num2str(pt_num),'.gif');

for i=epoch_start:win_shift:epoch_end - win_length
    win_start = find(t_span > i,1);
    win_end = win_start + win_length*60*Fs;
    EEG_win = EEG_data(:,win_start:win_end-1);

    model = fit_mindy(EEG_win, model_params, fit_params, nadam_params);
    alpha = model.alpha;
    b = model.b;
    W = model.W;
%     WL = model.WL;
    D2 = model.D2;
    D = model_params.Dmin + D2.^2;

    meas = EEG_data(:,win_end:win_end+nRec);
    pred = zeros(size(meas));
    pred(:,1) = meas(:,1);
    for j=2:nRec+1
        eps = 0.75*randn(nX,1);
        Psi = sqrt(alpha.^2 + (b*pred(:,j-1) + 0.5).^2) - sqrt(alpha.^2 + (b*pred(:,j-1) - 0.5).^2);
        pred(:,j) = (W*Psi - D.*pred(:,j-1) + eps)*0.1 + pred(:,j-1);
    end

    figure(3);
    clf;
    plot(meas(ch,:));
    hold on;
    plot(pred(ch,:));

    meas_spec = zeros(nX,nRec/2);
    pred_spec = zeros(nX,nRec/2);
    for j=1:nX
        meas_spec_tmp = fft(meas(j,:));
        meas_spec_tmp = abs(meas_spec_tmp)/length(meas_spec_tmp);
        meas_spec(j,:) = meas_spec_tmp(1:(nRec/2));
        pred_spec_tmp = fft(pred(j,:));
        pred_spec_tmp = abs(pred_spec_tmp)/length(pred_spec_tmp);
        pred_spec(j,:) = pred_spec_tmp(1:(nRec/2));
    end
    f = Fs*(1:(nRec/2))/nRec;
    figure(4);
    clf;
    plot(f,meas_spec(ch,:));
    hold on
    plot(f,pred_spec(ch,:));

%     cross_spec_meas = zeros(nX);
%     cross_spec_pred = zeros(nX);
%     xspec_freq = 2;
%     for ii=1:nX
% %         figure;
%         for jj=1:nX
% %             [cross_spec,f] = cpsd(MeasSet(ii,:),MeasSet(jj,:),[],[],0:0.5:15,200);
%             cross_cov = xcov(meas(ii,1:nRec),meas(jj,1:nRec),'normalized');
%             cross_spec = fft(cross_cov);
%             cross_spec = abs(cross_spec)/length(cross_cov);
%             cross_spec = cross_spec(1:ceil(length(cross_spec)/2));
%             cross_spec = cross_spec./max(cross_spec,[],2);
% %             subplot(2,1,1);
% %             hold on
% %             plot(cross_spec);
%             cross_spec_meas(ii,jj) = cross_spec(xspec_freq*2*nRec/Fs+1);
%             cross_cov = xcov(pred(ii,:),pred(jj,:),'normalized');
%             cross_spec = fft(cross_cov);
%             cross_spec = abs(cross_spec)/length(cross_cov);
%             cross_spec = cross_spec(1:ceil(length(cross_spec)/2));
%             cross_spec = cross_spec./max(cross_spec,[],2);
%             cross_spec_pred(ii,jj) = cross_spec(xspec_freq*2*nRec/Fs+1);
% %             subplot(2,1,2);
% %             hold on
% %             plot(cross_spec);
%         end
%     end
% %     imagesc(cross_spec_mat);
%     figure;
%     subplot(1,2,1);
%     imagesc(cross_spec_meas);
%     colorbar;
% %     caxis([0 1]);
%     subplot(1,2,2);
%     imagesc(cross_spec_pred);
%     colorbar;
%     caxis([0 1]);
% 
%     [meas_v, meas_d] = eig(cross_spec_meas);
%     [pred_v, pred_d] = eig(cross_spec_pred);


    fig5 = figure(5);
    subplot(1,2,1);
    title('Measured')
    [handle, Zi, ~, ~, ~] = topoplot(meas_spec(:,2), locs, 'maplimits', 'maxmin');
    clim1 = [min(min(Zi)) max(max(Zi))];
    text(-0.75,0.6,strcat('time: ',num2str(i),' min'));
    colorbar;
    subplot(1,2,2);
    title('Predicted')
    [handle, Zi] = topoplot(pred_spec(:,2), locs, 'maplimits', 'maxmin');
    clim2 = [min(min(Zi)) max(max(Zi))];
    clim = [min(clim1(1),clim2(1)) max(clim1(2),clim2(2))];
    caxis(clim);
    colorbar;
    subplot(1,2,1);
    title('Measured')
    [handle, Zi, ~, ~, ~] = topoplot(meas_spec(:,2), locs, 'maplimits', 'maxmin');
    clim1 = [min(min(Zi)) max(max(Zi))];
    text(-0.75,0.6,strcat('time: ',num2str(i)));
    caxis(clim);
    colorbar;
    
%     fig5 = figure(5);
%     clf;
%     fig5.Position = [1 41 1920 963];
%     subplot(1,2,1);
%     title('Measured')
%     spectopo(meas,0,Fs,'freq',[1,5,10],'chanlocs',locs,'limits',[NaN NaN NaN NaN -25 25]);
%     text(-45,20,strcat('time: ',num2str(i),' min'));
%     subplot(1,2,2);
%     title('Predicted')
%     spectopo(pred,0,Fs,'freq',[1,5,10],'chanlocs',locs,'limits',[NaN NaN NaN NaN -25 25]);

    frame = getframe(fig5);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    if i == epoch_start
        imwrite(imind,cm,vidName,'gif', 'Loopcount',1,"DelayTime",.5);
    else
        imwrite(imind,cm,vidName,'gif','WriteMode','append',"DelayTime",.5);
    end
    close;

    % Reseed model params for next epoch with fitted params for current
    % epoch
    model_params.alpha = alpha;
    model_params.xi = b./sqrt(alpha.^2 + 0.25);
    model_params.W = W;
%     model_params.W1 = model.W1;
%     model_params.W2 = model.W2;
    model_params.D2 = D2;
end