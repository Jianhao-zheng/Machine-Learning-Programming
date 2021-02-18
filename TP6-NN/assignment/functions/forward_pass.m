function [Y, A, Z] = forward_pass(X, W, W0, Sigmas)
%FORWARD_PASS This function calculate the forward pass of the network with
%   inputs:
%       o X (NxM) The input data
%       o W {Lx1} cell array containing the weight matrices for all the layers 
%       o W0 {Lx1} cell array containing the bias matrices for all the layers
%       o Sigmas {Lx1} cell array containing the type of the activation
%       functions for all the layers
%
%   outputs:
%       o Y (PxM) The output of the network, result of the activation
%       function of the last layer
%       o A {L+1x1} cell array containing the results of the activation functions
%       at each layer. Also contain the input layer A0
%       o Z {Lx1} cell array containing the Z values at each layer

L = length(W);

A = cell(L+1,1);
Z = cell(L,1);

A{1} = X;
for i = 1:L
    % Here should be Z{i} = W{i}*A{i} + W0{i};
    % I write W0{i}(:,1) instead of W0{i} is because the input W0{i}
    % sometimes is a S*S matrix with repeat column during the test.
    % So, I simply take the first column. Thus, when W0{i} is S*1, it will
    % still take W0{i}. When W0{i} is S*S with repeat column, it will take
    % the first column.
    Z{i} = W{i}*A{i} + W0{i}(:,1);
    A{i+1} = forward_activation(Z{i},Sigmas{i});
end

Y = A{L+1};


end

