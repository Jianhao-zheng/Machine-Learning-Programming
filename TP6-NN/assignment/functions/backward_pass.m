function [dZ, dW, dW0] = backward_pass(dE, W, A, Z, Sigmas)
%BACKWARD_PASS This function calculate the backward pass of the network with
%   inputs:
%       o dE (PxM) The derivative dE/dZL
%       o W {Lx1} cell array containing the weight matrices for all the layers 
%       o b {Lx1} cell array containing the bias matrices for all the layers
%       o A {L+1x1} cell array containing the results of the activation functions
%       at each layer. Also contain the input layer A0
%       o Z {Lx1} cell array containing the Z values at each layer
%       o Sigmas {Lx1} cell array containing the type of the activation
%       functions for all the layers
%
%   outputs:
%       o dZ {Lx1} cell array containing the derivatives dE/dZl values at each layer
%       o dW {Lx1} cell array containing the derivatives of the weights at
%       each layer
%       o dW0 {Lx1} cell array containing the derivatives of the bias at each layer
[~,M] = size(dE);
L = length(W);

dZ = cell(L,1);
dW = cell(L,1);
dW0 = cell(L,1);

dZ{L} = dE;
dW{L} = dZ{L}*A{L}'./M; % Here, A{L} actually represents A(L-1) since we start with X as A(0)
dW0{L} = sum(dZ{L},2)./M;
for i = L-1:-1:1
    dZ{i} = (W{i+1}'*dZ{i+1}).*backward_activation(Z{i}, Sigmas{i});
    dW{i} = dZ{i}*A{i}'./M;% Same as stated above, A{i} means A(i-1)
    dW0{i} = sum(dZ{i},2)./M;
end

end

