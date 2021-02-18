function [ logl ] = gmmLogLik(X, Priors, Mu, Sigma)
%MY_GMMLOGLIK Compute the likelihood of a set of parameters for a GMM
%given a dataset X
%
%   input------------------------------------------------------------------
%
%       o X      : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o Priors : (1 x K), the set of priors (or mixing weights) for each
%                           k-th Gaussian component
%       o Mu     : (N x K), an NxK matrix corresponding to the centroids mu = {mu^1,...mu^K}
%       o Sigma  : (N x N x K), an NxNxK matrix corresponding to the 
%                    Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
%
%   output ----------------------------------------------------------------
%
%      o logl       : (1 x 1) , loglikelihood
%%
K = length(Priors);
M = size(X,2);

prob = zeros(K,M);
for j = 1:K
    %Calculate the pdf for each set of parameters
    prob(j,:) = gaussPDF(X, Mu(:,j), Sigma(:,:,j));
end

inner_sum = log(Priors*prob); %compute the inner sum with the priors and log() the value

logl = sum(inner_sum);%sum all the  log-ed probabilities


end

