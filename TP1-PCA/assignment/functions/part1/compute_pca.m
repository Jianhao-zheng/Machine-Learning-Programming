function [Mu, C, EigenVectors, EigenValues] = compute_pca(X)
%COMPUTE_PCA Step-by-step implementation of Principal Component Analysis
%   In this function, the student should implement the Principal Component 
%   Algorithm
%
%   input -----------------------------------------------------------------
%   
%       o X      : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%
%   output ----------------------------------------------------------------
%
%       o Mu              : (N x 1), Mean Vector of Dataset
%       o C               : (N x N), Covariance matrix of the dataset
%       o EigenVectors    : (N x N), Eigenvectors of Covariance Matrix.
%       o EigenValues     : (N x 1), Eigenvalues of Covariance Matrix
Mu=mean(X,2);
X_temp=X-Mu;
C=X_temp*X_temp'./(length(X)-1);

[EigenVectors,D] = eig(C);
[EigenValues,ind] = sort(diag(D),'descend');
EigenVectors = EigenVectors(:,ind);


end

