function [model_prs, nadam_prs] = nadam_param_update(model_prs, nadam_prs, xi_update, Z, RX, iBatch)
    % Extract model parameters
    b = model_prs.b;
    xi = model_prs.xi;
    W = model_prs.W;
%     W1 = model_prs.W1;
%     W2 = model_prs.W2;
%     WL = W1*W2';
    D2 = model_prs.D2;

    % Extract NADAM parameters
    mu = nadam_prs.mu;
    nu = nadam_prs.nu;
    lambda1 = nadam_prs.lambda1;
    lambda2 = nadam_prs.lambda2;
    lambda3 = nadam_prs.lambda3;
    lambda4 = nadam_prs.lambda4;

    % xi
    mxi = nadam_prs.mxi;
    nxi = nadam_prs.nxi;
    xi_rate = nadam_prs.xi_rate;
    xi_eps = nadam_prs.xi_eps;
    gradxi = (xi.^-3).*b.*xi_update;
    mxi = mu*mxi + (1 - mu)*gradxi;
    nxi = nu*nxi + (1 - nu)*gradxi.^2;
    xi = xi - xi_rate*(((((1 - mu)/(1 - mu^(iBatch+1)))*gradxi) + ((mu/(1 - mu^(iBatch+2)))*mxi))./(sqrt(nxi/(1 - nu^(iBatch+1))) + xi_eps));
    xi(xi>2*b) = 2*b;

    % WS
    mW = nadam_prs.mW;
    nW = nadam_prs.nW;
    W_rate = nadam_prs.W_rate;
    W_eps = nadam_prs.W_eps;
    gradW = -Z + lambda1*sign(W) + lambda2*diag(sign(W));
    mW = mu*mW + (1 - mu)*gradW;
    nW = nu*nW + (1 - nu)*gradW.^2;
    W = W - W_rate*(((((1 - mu)/(1 - mu^(iBatch+1)))*gradW) + ((mu/(1 - mu^(iBatch+2)))*mW))/(sqrt(nW/(1 - nu^(iBatch+1))) + W_eps));

    % W1
%     mW1 = nadam_prs.mW1;
%     nW1 = nadam_prs.nW1;
%     W1_rate = nadam_prs.W12_rate;
%     W1_eps = nadam_prs.W12_eps;
%     gradW1 = -(Z - lambda4*WL)*W2 + lambda3*sign(W1);
%     mW1 = mu*mW1 + (1 - mu)*gradW1;
%     nW1 = nu*nW1 + (1 - nu)*gradW1.^2;
%     W1 = W1 - W1_rate*(((((1 - mu)/(1 - mu^(iBatch+1)))*gradW1) + ((mu/(1 - mu^(iBatch+2)))*mW1))./(sqrt(nW1/(1 - nu^(iBatch+1))) + W1_eps));
% 
%     % W2
%     mW2 = nadam_prs.mW2;
%     nW2 = nadam_prs.nW2;
%     W2_rate = nadam_prs.W12_rate;
%     W2_eps = nadam_prs.W12_eps;
%     gradW2 = -(W1'*(Z - lambda4*WL))' + lambda3*sign(W2);
%     mW2 = mu*mW2 + (1 - mu)*gradW2;
%     nW2 = nu*nW2 + (1 - nu)*gradW2.^2;
%     W2 = W2 - W2_rate*(((((1 - mu)/(1 - mu^(iBatch+1)))*gradW2) + ((mu/(1 - mu^(iBatch+2)))*mW2))./(sqrt(nW2/(1 - nu^(iBatch+1))) + W2_eps));

    % D2
    mD2 = nadam_prs.mD2;
    nD2 = nadam_prs.nD2;
    D2_rate = nadam_prs.D2_rate;
    D2_eps = nadam_prs.D2_eps;
    gradD2 = 2*D2.*RX;
    mD2 = mu*mD2 + (1 - mu)*gradD2;
    nD2 = nu*nD2 + (1 - nu)*gradD2.^2;
    D2 = D2 - D2_rate*(((((1 - mu)/(1 - mu^(iBatch+1)))*gradD2) + ((mu/(1 - mu^(iBatch+2)))*mD2))./(sqrt(nD2/(1 - nu^(iBatch+1))) + D2_eps));

    model_prs.xi = xi;
    model_prs.W = W;
%     model_prs.W1 = W1;
%     model_prs.W2 = W2;
    model_prs.D2 = D2;

    nadam_prs.mxi = mxi;
    nadam_prs.nxi = nxi;
    nadam_prs.mW = mW;
    nadam_prs.nW = nW;
%     nadam_prs.mW1 = mW1;
%     nadam_prs.nW1 = nW1;
%     nadam_prs.mW2 = mW2;
%     nadam_prs.nW2 = nW2;
    nadam_prs.mD2 = mD2;
    nadam_prs.nD2 = nD2;
end