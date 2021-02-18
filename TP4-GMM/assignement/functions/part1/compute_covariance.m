function [ Sigma ] = compute_covariance( X, X_bar, type )
%MY_COVARIANCE computes the covariance matrix of X given a covariance type.
%
% Inputs -----------------------------------------------------------------
%       o X     : (N x M), a data set with M samples each being of dimension N.
%                          each column corresponds to a datapoint
%       o X_bar : (N x 1), an Nx1 matrix corresponding to mean of data X
%       o type  : string , type={'full', 'diag', 'iso'} of Covariance matrix
%
% Outputs ----------------------------------------------------------------
%       o Sigma : (N x N), an NxN matrix representing the covariance matrix of the 
%                          Gaussian function
%%
[N,M] = size(X);
switch type
    case 'full'
        X = X - X_bar;%zero-mean data
        Sigma = X*X'./(M-1);
        
    case 'diag'
        X = X - X_bar;%zero-mean data
        Sigma = X*X'./(M-1);
        Sigma = Sigma.*eye(N);%extract only the diagonal values
        
    case 'iso'
        %Computing the isotropic variance by summing the squared distance
        %between each points and the mean
        Sigma_iso = 0;
        for i = 1:M
            Sigma_iso = Sigma_iso + (norm(X(:,i)-X_bar,2))^2;
        end
        Sigma_iso = Sigma_iso/(M*N);
        %replicate the variance in a diagonal matrix
        Sigma = Sigma_iso.*eye(N);
end
        


end

