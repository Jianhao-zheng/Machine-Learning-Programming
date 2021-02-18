function prob = gaussPDF(X, Mu, Sigma)
%MY_GAUSSPDF computes the Probability Density Function (PDF) of a
% multivariate Gaussian represented by a mean and covariance matrix.
%
% Inputs -----------------------------------------------------------------
%       o X     : (N x M), a data set with M samples each being of dimension N.
%                          each column corresponds to a datapoint
%       o Mu    : (N x 1), an Nx1 vector corresponding to the mean of the 
%							Gaussian function
%       o Sigma : (N x N), an NxN matrix representing the covariance matrix 
%						   of the Gaussian function
% Outputs ----------------------------------------------------------------
%       o prob  : (1 x M),  a 1xM vector representing the probabilities for each 
%                           M datapoints given Mu and Sigma    
%%
[N,M] = size(X);

det_Sigma = det(Sigma);
factor = 1/sqrt((2*pi)^N*det_Sigma);
prob = zeros(1,M);

for i = 1:M
%     Calculate the pdf by definition
    prob(i) = factor*exp(-(X(:,i)-Mu)'/Sigma*(X(:,i)-Mu)/2);
%     Or we can use the matlab function mvnpdf:
%     prob(i) = mvnpdf(X(:,i),Mu,Sigma);
end



end