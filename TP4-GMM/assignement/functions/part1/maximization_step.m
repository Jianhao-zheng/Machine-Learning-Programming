function [Priors,Mu,Sigma] = maximization_step(X, Pk_x, params)
%MAXIMISATION_STEP Compute the maximization step of the EM algorithm
%   input------------------------------------------------------------------
%       o X         : (N x M), a data set with M samples each being of 
%       o Pk_x      : (K, M) a KxM matrix containing the posterior probabilty
%                     that a k Gaussian is responsible for generating a point
%                     m in the dataset, output of the expectation step
%       o params    : The hyperparameters structure that contains k, the number of Gaussians
%                     and cov_type the coviariance type
%   output ----------------------------------------------------------------
%       o Priors    : (1 x K), the set of updated priors (or mixing weights) for each
%                           k-th Gaussian component
%       o Mu        : (N x K), an NxK matrix corresponding to the updated centroids 
%                           mu = {mu^1,...mu^K}
%       o Sigma     : (N x N x K), an NxNxK matrix corresponding to the
%                   updated Covariance matrices  Sigma = {Sigma^1,...,Sigma^K}
%%
[N,M] = size(X);

%update the priors
Priors = mean(Pk_x,2);
Priors = Priors';

%update the means
Mu = (X*Pk_x')./sum(Pk_x,2)';

%update the covariance matices

Sigma = zeros(N,N,params.k);
switch params.cov_type
    case 'full'
        for i = 1:params.k
            X_temp = X - Mu(:,i);%store the zero-mean data
            for j = 1:M
                Sigma(:,:,i) = Sigma(:,:,i) + Pk_x(i,j).*X_temp(:,j)*X_temp(:,j)';
            end
            Sigma(:,:,i) = Sigma(:,:,i)./sum(Pk_x(i,:));
        end
        
    case 'diag'
        for i = 1:params.k
            X_temp = X - Mu(:,i);%store the zero-mean data
            for j = 1:M
                Sigma(:,:,i) = Sigma(:,:,i) + Pk_x(i,j).*X_temp(:,j)*X_temp(:,j)';
            end
            Sigma(:,:,i) = Sigma(:,:,i)./sum(Pk_x(i,:));
            Sigma(:,:,i) = Sigma(:,:,i).*eye(N);%extract only the diagonal values
        end
        
    case 'iso'
        for i = 1:params.k
            iso = 0;
            for j = 1:M
                iso = iso + Pk_x(i,j)*(norm(X(:,j)-Mu(:,i),2))^2;
            end
            iso = iso/(sum(Pk_x(i,:))*N);
            Sigma(:,:,i) = iso.*eye(N);
        end
        
end

%add a tiny variance to avoid numerical instability
epsilon = 1e-5;
for i = 1:params.k
    Sigma(:,:,i) = Sigma(:,:,i) + epsilon.*eye(N);
end

end

