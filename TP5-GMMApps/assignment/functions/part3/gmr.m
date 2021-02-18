function [y_est, var_est] = gmr(Priors, Mu, Sigma, X, in, out)
%GMR This function performs Gaussian Mixture Regression (GMR), using the 
% parameters of a Gaussian Mixture Model (GMM) for a D-dimensional dataset,
% for D= N+P, where N is the dimensionality of the inputs and P the 
% dimensionality of the outputs.
%
% Inputs -----------------------------------------------------------------
%   o Priors:  1 x K array representing the prior probabilities of the K GMM 
%              components.
%   o Mu:      D x K array representing the centers of the K GMM components.
%   o Sigma:   D x D x K array representing the covariance matrices of the 
%              K GMM components.
%   o X:       N x M array representing M datapoints of N dimensions.
%   o in:      1 x N array representing the dimensions of the GMM parameters
%                to consider as inputs.
%   o out:     1 x P array representing the dimensions of the GMM parameters
%                to consider as outputs. 
% Outputs ----------------------------------------------------------------
%   o y_est:     P x M array representing the retrieved M datapoints of 
%                P dimensions, i.e. expected means.
%   o var_est:   P x P x M array representing the M expected covariance 
%                matrices retrieved. 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Auxiliary Variables
[N,M] = size(X);
P = length(out);
K = length(Priors);

% Auxiliary Matrixs
Mu_x = Mu(in,:);
Mu_y = Mu(out,:);
Sigma_xx = Sigma(in,in,:);
Sigma_yx = Sigma(out,in,:);
Sigma_xy = Sigma(in,out,:);
Sigma_yy = Sigma(out,out,:);

%% Compute y_est
y_est = zeros(P,M);

% Compute density: p(x|mu_x,sigma_xx)
p_x = zeros(K,M);
for i = 1:K
    p_x(i,:) = gaussPDF(X, Mu_x(:,i), Sigma_xx(:,:,i));
end

% Compute Beta_k
Beta_k = (Priors'.*p_x)./(Priors*p_x);

% Compute Means
for i = 1:K
    Mu_bar_k = Mu_y(:,i) + (Sigma_yx(:,:,i)/Sigma_xx(:,:,i))*(X - Mu_x(:,i));
    y_est = y_est + Beta_k(i,:).*Mu_bar_k;
end

%% Compute var_est
% Compute Sigma_bar_k
Sigma_bar_k = zeros(P,P,K);
for i = 1:K
    Sigma_bar_k(:,:,i) = Sigma_yy(:,:,i)...
        -(Sigma_yx(:,:,i)/Sigma_xx(:,:,i))*Sigma_xy(:,:,i);
end

% Compute variances
var_est = zeros(P,P,M);
for i = 1:M
%     var_est(:,:,i) = var_est(:,:,i) - y_est(:,i)*y_est(:,i)';
    %Assume x,y has only one dimension, i.e. P=N=1
    var_est(:,:,i) = var_est(:,:,i) - y_est(:,i)^2; 
    for j = 1:K
        Mu_bar_kx = Mu_y(:,j) + (Sigma_yx(:,:,j)/Sigma_xx(:,:,j))*(X(:,i) - Mu_x(:,j));
%         var_est(:,:,i) = var_est(:,:,i)...
%             + Beta_k(j,i).*(Mu_bar_kx*Mu_bar_kx'+ Sigma_bar_k(:,:,j));
        var_est(:,:,i) = var_est(:,:,i)...
            + Beta_k(j,i).*(Mu_bar_kx^2+ Sigma_bar_k(:,:,j));
    end
end

end

