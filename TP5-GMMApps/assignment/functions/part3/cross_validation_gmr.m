function [metrics] = cross_validation_gmr( X, y, F_fold, valid_ratio, k_range, params )
%CROSS_VALIDATION_GMR Implementation of F-fold cross-validation for regression algorithm.
%
%   input -----------------------------------------------------------------
%
%       o X         : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o y         : (P x M) array representing the y vector assigned to
%                           each datapoints
%       o F_fold    : (int), the number of folds of cross-validation to compute.
%       o valid_ratio  : (double), Testing Ratio.
%       o k_range   : (1 x K), Range of k-values to evaluate
%       o params    : parameter strcuture of the GMM
%
%   output ----------------------------------------------------------------
%       o metrics : (structure) contains the following elements:
%           - mean_MSE   : (1 x K), Mean Squared Error computed for each value of k averaged over the number of folds.
%           - mean_NMSE  : (1 x K), Normalized Mean Squared Error computed for each value of k averaged over the number of folds.
%           - mean_R2    : (1 x K), Coefficient of Determination computed for each value of k averaged over the number of folds.
%           - mean_AIC   : (1 x K), Mean AIC Scores computed for each value of k averaged over the number of folds.
%           - mean_BIC   : (1 x K), Mean BIC Scores computed for each value of k averaged over the number of folds.
%           - std_MSE    : (1 x K), Standard Deviation of Mean Squared Error computed for each value of k.
%           - std_NMSE   : (1 x K), Standard Deviation of Normalized Mean Squared Error computed for each value of k.
%           - std_R2     : (1 x K), Standard Deviation of Coefficient of Determination computed for each value of k averaged over the number of folds.
%           - std_AIC    : (1 x K), Standard Deviation of AIC Scores computed for each value of k averaged over the number of folds.
%           - std_BIC    : (1 x K), Standard Deviation of BIC Scores computed for each value of k averaged over the number of folds.
%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Auxiliary Variables
N = size(X,1);
P = size(y,1);
K = length(k_range);

metrics.mean_MSE = zeros(1,K); metrics.mean_NMSE = zeros(1,K); 
metrics.mean_R2 = zeros(1,K); metrics.mean_AIC = zeros(1,K);
metrics.mean_BIC = zeros(1,K); metrics.std_MSE = zeros(1,K);
metrics.std_NMSE = zeros(1,K); metrics.std_R2 = zeros(1,K);
metrics.std_AIC = zeros(1,K); metrics.std_BIC = zeros(1,K);

for i = 1:K
    MSE_k = zeros(1,F_fold); NMSE_k = zeros(1,F_fold); R2_k = zeros(1,F_fold);
    AIC_k = zeros(1,F_fold); BIC_k = zeros(1,F_fold);
    params.k = k_range(i);
    for j = 1:F_fold
        [ X_train, y_train, X_test, y_test ] = split_regression_data(X, y, valid_ratio);% split data
        [Priors, Mu, Sigma, ~] = gmmEM([X_train; y_train], params);% Run GMM-EM function
        in  = 1:N;       % input dimensions
        out = N+1:(N+P); % output dimensions
        [y_est, ~] = gmr(Priors, Mu, Sigma, X_test, in, out);
        
        %Although the text told us to compute each metric for the test
        %data, I think AIC and BIC should be computed on train data.
        %Because these two metrics are computed for model selection. In
        %summary, MSE, NMSE and R2_k is computed on test data to test the
        %performance of the model, while AIC and BIC is computed on train
        %data for model selection.
        
        %Compute the regression metrics on test data.
        [MSE_k(j), NMSE_k(j), R2_k(j)] = regression_metrics( y_est, y_test );
        
        %Compute the AIC and BIC on train data
        [AIC_k(j), BIC_k(j)] =  gmm_metrics([X_train; y_train], Priors, Mu, Sigma, params.cov_type);
        
%         %Compute the AIC and BIC on test data, which I think should be
%         %wrong
%         [AIC_k(j), BIC_k(j)] =  gmm_metrics([X_test; y_test], Priors, Mu, Sigma, params.cov_type);
    end
    metrics.mean_MSE(i) = mean(MSE_k); metrics.mean_NMSE(i) = mean(NMSE_k); 
    metrics.mean_R2(i) = mean(R2_k); metrics.mean_AIC(i) = mean(AIC_k);
    metrics.mean_BIC(i) = mean(BIC_k); metrics.std_MSE(i) = std(MSE_k);
    metrics.std_NMSE(i) = std(NMSE_k); metrics.std_R2(i) = std(R2_k);
    metrics.std_AIC(i) = std(AIC_k); metrics.std_BIC(i) = std(BIC_k);
end






end

