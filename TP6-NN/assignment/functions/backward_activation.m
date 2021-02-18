function [dZ] = backward_activation(Z, Sigma)
%BACKWARD_ACTIVATION Compute the derivative of the activation function
%evaluated in Z
%   inputs:
%       o Z (NxM) Z value, input of the activation function. The size N
%       depends of the number of neurons at the considered layer but is
%       irrelevant here.
%       o Sigma (string) type of the activation to use
%   outputs:
%       o dZ (NxM) derivative of the activation function
switch Sigma
    case 'sigmoid'
        dZ = exp(-Z)./((1+exp(-Z)).^2);
    case 'tanh'
        dZ = 4.*exp(-2.*Z)./((1+exp(-2.*Z)).^2);
    case 'relu'
        % In this method, I consider 0 as the positive case when compute
        % the derivative
        dZ = Z;
        dZ(Z>=0) =1;
        dZ(Z<0) =1;
    case 'leakyrelu'
        % In this method, I consider 0 as the positive case when compute
        % the derivative
        k = 0.01; %usually chosen value
        dZ = Z;
        dZ(Z>=0) = 1;
        dZ(Z<0) = k;
end

end

