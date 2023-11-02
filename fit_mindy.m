function model = fit_mindy(data, model_prs, fit_prs, nadam_prs)
tic

% Extract fitting params
nX = size(data,1);
batchSz = fit_prs.batchSz; 
nBatch = fit_prs.nBatch; % Fitting iterations

% Initialize model params
alpha = model_prs.alpha;
b = model_prs.b;
xi = b./sqrt(alpha.^2 + 0.25);
W = model_prs.W;
% W1 = model_prs.W1;
% W2 = model_prs.W2;
% WL = W1*W2';
Dmin = model_prs.Dmin;
D2 = model_prs.D2;
D = Dmin + D2.^2;
alpha_hist = zeros(nX, nBatch);
W_hist = zeros(nX, nX, nBatch);
D_hist = zeros(nX, nBatch);

for iBatch=1:nBatch
    Z = zeros(nX,nX);
    RX = zeros(nX,1);
    
    iB = randi(size(data,2)-(batchSz+1)); % Begin indexing at random spot in the data
    z0 = data(:,iB:iB+batchSz); % Initial true sensor state
    x0 = z0; % Initial predicted sensor state (initialized as true sensor state)

    % Forward prediction
    P1 = sqrt(xi.^-2 + x0.*(x0 + 1/b));
    P2 = sqrt(xi.^-2 + x0.*(x0 - 1/b));
    Psi = b*(P1 - P2);
%     xRec = (WS + WL)*Psi - D.*x0;
    xRec = W*Psi - D.*x0;

    R = (data(:,iB+1:iB+batchSz+1) - z0) - xRec;
%     Q = (WS + WL)'*R;
    Q = W'*R;
    xi_update = Q.*(1./P1 - 1./P2);
    Z = Z + R*Psi';
    RX = RX + R.*z0;
    err = norm(R);
        

        % Backpropagate Error
%         figure(2);
%         clf;
%         subplot(2,1,1)
%         plot(zRec(1,:));
%         hold on
%         plot(xRec(1,:));
%         subplot(2,1,2);
%         plot(zRec(2,:));
%         hold on
%         plot(xRec(2,:));

    % Calculate cost function variables
    xi_update = mean(xi_update,2);
    Z = Z/batchSz;
    RX = mean(RX,2);

    % Update parameters
    [model_prs, nadam_prs] = nadam_param_update(model_prs, nadam_prs, xi_update, Z, RX, iBatch);
    xi = model_prs.xi;
    alpha = sqrt((b./xi).^2 - 0.25);
    W = model_prs.W;
%     W1 = model_prs.W1;
%     W2 = model_prs.W2;
%     WL = W1*W2';
    D2 = model_prs.D2;
    D = Dmin + D2.^2;

    if mod(iBatch,250)==1
        disp([iBatch nBatch]);
    end
    % Record mean squared error over batch
    model.recE(:,iBatch) = mean(err);

    % Record parameters
    alpha_hist(:,iBatch) = alpha;
%     W_hist(:,:,iBatch) = WS + WL;
    W_hist(:,:,iBatch) = W;
    D_hist(:,iBatch) = D;

    % Save true & predicted timeseries in model
    if mod(iBatch,500)==0
        model.truePred = z0;
        model.Pred = xRec;
    end
end

% Save model params
model.alpha = alpha;
model.b = b;
model.W = W;
% model.WL = W1*W2';
% model.W1 = W1;
% model.W2 = W2;
model.D2 = D2;

fprintf('Run time: %f\n', toc);

figure(2);
plot(model.recE);

end