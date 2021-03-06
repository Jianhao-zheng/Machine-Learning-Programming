function [RSS_curve, AIC_curve, BIC_curve] =  kmeans_eval(X, K_range,  repeats, init, type, MaxIter)
%KMEANS_EVAL Implementation of the k-means evaluation with clustering
%metrics.
%
%   input -----------------------------------------------------------------
%   
%       o X           : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o repeats     : (1 X 1), # times to repeat k-means
%       o K_range     : (1 X K_range), Range of k-values to evaluate
%       o init        : (string), type of initialization {'sample','range'}
%       o type        : (string), type of distance {'L1','L2','LInf'}
%       o MaxIter     : (int), maximum number of iterations
%
%   output ----------------------------------------------------------------
%       o RSS_curve  : (1 X K_range), RSS values for each value of K in K_range
%       o AIC_curve  : (1 X K_range), AIC values for each value of K in K_range
%       o BIC_curve  : (1 X K_range), BIC values for each value of K in K_range
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
RSS_curve = zeros(repeats,length(K_range));
AIC_curve = zeros(repeats,length(K_range));
BIC_curve = zeros(repeats,length(K_range));
for i = 1:length(K_range)
    K=K_range(i);
    for j = 1:repeats
        [labels, Mu, ~, ~] =  kmeans(X,K,init,type,MaxIter,0);
        [RSS_curve(j,i), AIC_curve(j,i), BIC_curve(j,i)] =  compute_metrics(X, labels, Mu);
    end
end
RSS_curve=sum(RSS_curve,1)./repeats;
AIC_curve=sum(AIC_curve,1)./repeats;
BIC_curve=sum(BIC_curve,1)./repeats;

end