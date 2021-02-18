function [Yest] = gmm_classifier(Xtest, models, labels)
%GMM_CLASSIFIER Classifies datapoints of X_test using ML Discriminant Rule
%   input------------------------------------------------------------------
%
%       o Xtest    : (N x M_test), a data set with M_test samples each being of
%                           dimension N, each column corresponds to a datapoint.
%       o models    : (1 x N_classes) struct array with fields:
%                   | o Priors : (1 x K), the set of priors (or mixing weights) for each
%                   |            k-th Gaussian component
%                   | o Mu     : (N x K), an NxK matrix corresponding to the centroids
%                   |            mu = {mu^1,...mu^K}
%                   | o Sigma  : (N x N x K), an NxNxK matrix corresponding to the
%                   |            Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
%       o labels    : (1 x N_classes) unique labels of X_test.
%   output ----------------------------------------------------------------
%       o Yest  :  (1 x M_test), a vector with estimated labels y \in {0,...,N_classes}
%                   corresponding to X_test.
%%

% Auxiliary Variables
[~,M_test] = size(Xtest);
N_classes = length(labels);

% Compute the the conditional densities to belong to class i: p(x'|y=i)
p_x = zeros(N_classes,M_test);
for i = 1:N_classes
    Priors_temp = models(i).Priors;
    Mu_temp = models(i).Mu;
    Sigma_temp = models(i).Sigma;
    K = length(models(1).Priors);
    prob_temp = zeros(K,M_test);
    for j = 1:K
        prob_temp(j,:) = gaussPDF(Xtest, Mu_temp(:,j), Sigma_temp(:,:,j));
    end
    p_x(i,:) = Priors_temp*prob_temp;
end

% Implement the Gaussian Maximum Likelihood Discriminant Rule
[~,idx] = max(p_x);
Yest = labels(idx);




end