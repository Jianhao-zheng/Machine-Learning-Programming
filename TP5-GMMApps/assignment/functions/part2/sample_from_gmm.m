function [XNew] = sample_from_gmm(gmm, nbSamples)
%SAMPLE_FROM_GMM Generate new samples from a learned GMM
%
%   input------------------------------------------------------------------
%       o gmm    : (structure), Contains the following fields
%                   | o Priors : (1 x K), the set of priors (or mixing weights) for each
%                   |            k-th Gaussian component
%                   | o Mu     : (N x K), an NxK matrix corresponding to the centroids
%                   |            mu = {mu^1,...mu^K}
%                   | o Sigma  : (N x N x K), an NxNxK matrix corresponding to the
%                   |            Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
%       o nbSamples    : (int) Number of samples to generate.
%   output ----------------------------------------------------------------
%       o XNew  :  (N x nbSamples), Newly generated set of samples.
%%
[N,K] = size(gmm.Mu);

% Select which Gaussian  k  to draw a sample from
chosen_gaussains = randsrc(nbSamples,1,[1:K; gmm.Priors]);

% Draw same number of datapoints from a Gaussian as the frequency it was choosed. 
XNew = [];
for i = 1:K
    Mu_temp = gmm.Mu(:,i);
    Sigma_temp = gmm.Sigma(:,:,i);
    num_choosed = length(chosen_gaussains(chosen_gaussains==i));
    XNew = [XNew, mvnrnd(Mu_temp,Sigma_temp,num_choosed)'];
end


% Another way to compute XNew.
% XNew = zeros(N,nbSamples);
% for i = 1:nbSamples
%     Mu_temp = gmm.Mu(:,chosen_gaussains(i));
%     Sigma_temp = gmm.Sigma(:,:,chosen_gaussains(i));
%     XNew(:,i) = mvnrnd(Mu_temp,Sigma_temp,1)';
% end

end

