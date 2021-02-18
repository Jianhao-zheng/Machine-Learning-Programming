function [A] = forward_activation(Z, Sigma)
%FORWARD_ACTIVATION Compute the value A of the activation function given Z
%   inputs:
%       o Z (NxM) Z value, input of the activation function. The size N
%       depends of the number of neurons at the considered layer but is
%       irrelevant here.
%       o Sigma (string) type of the activation to use
%
%   outputs:
%       o A (NXM) value of the activation function
switch Sigma
    case 'sigmoid'
        A = 1./(1+exp(-Z));
    case 'tanh'
        A = (exp(Z)-exp(-Z))./(exp(Z)+exp(-Z));
    case 'relu'
        A = max(0,Z);
    case 'leakyrelu'
        k = 0.01; %usually chosen value
        A = max(k.*Z,Z);
    case 'softmax'
        del = max(Z);
        denom = sum(exp(Z-del)); % compute the Denominator of softmax
        A = exp(Z-del)./denom;
end

end

