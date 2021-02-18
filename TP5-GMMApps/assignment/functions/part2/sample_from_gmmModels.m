function [XNew] = sample_from_gmmModels(models, nbSamplesPerClass, desiredClass)
%SAMPLE_FROM_GMMMODELS Generate new samples from a set of GMM
%   input------------------------------------------------------------------
%       o models : (structure array), Contains the following fields
%                   | o Priors : (1 x K), the set of priors (or mixing weights) for each
%                   |            k-th Gaussian component
%                   | o Mu     : (N x K), an NxK matrix corresponding to the centroids
%                   |            mu = {mu^1,...mu^K}
%                   | o Sigma  : (N x N x K), an NxNxK matrix corresponding to the
%                   |            Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
%       o nbSamplesPerClass    : (int) Number of samples per class to generate.
%       o desiredClass : [optional] (int) Desired class to generate samples for
%   output ----------------------------------------------------------------
%       o XNew  :  (N x nbSamples), Newly generated set of samples.
%       nbSamples depends if the optional argument is provided or not. If
%       not nbSamples = nbSamplesPerClass * nbClasses, if yes nbSamples = nbSamplesPerClass
%%
% Check number of arguments
if nargin ~=2 && nargin~=3
    error('wrong number of arguments')
end

if nargin == 3
    % Add this because our label is start from 0
    % Or we can simply use the commands below it to replace this one if we
    % have Ytrain as our input
    idx = desiredClass + 1;
    % labels = unique(Ytrain); idx = 1:length(labels);
    % idx =idx(labels == desiredClass);
    gmm_temp.Priors = models(idx).Priors;
    gmm_temp.Mu = models(idx).Mu;
    gmm_temp.Sigma = models(idx).Sigma;
    XNew = sample_from_gmm(gmm_temp, nbSamplesPerClass);
else
    XNew = [];
    for i = 1:length(models)
        gmm_temp.Priors = models(i).Priors;
        gmm_temp.Mu = models(i).Mu;
        gmm_temp.Sigma = models(i).Sigma;
        XNew = [XNew, sample_from_gmm(gmm_temp, nbSamplesPerClass)];
    end
end
    
    




end
