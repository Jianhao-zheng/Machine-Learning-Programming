function [dZ] = cost_derivative(Y, Yd, typeCost, typeLayer)
%COST_DERIVATIVE compute the derivative of the cost function w.r.t to the Z
%value of the last layer
%   inputs:
%       o Y (PxM) Output of the last layer of the network, should match
%       Yd
%       o Yd (PxM) Ground truth
%       o typeCost (string) type of the cost evaluation function
%       o typeLayer (string) type of the last layer
%   outputs:
%       o dZ (PxM) The derivative dE/dZL

[P, M] = size(Y);
%Compute dZ_A, which means dAL/dZL
switch typeLayer
    case 'sigmoid'
        dZ_A = Y.*(1-Y);
    case 'tanh'
        dZ_A = 1 - Y.^2;
    case 'softmax'
        dZ_A = cell(P,1);
        % dZ_A{i} means dAi/dZ, which is the derivative of the ith
        % demension of A
        for i = 1:P
            % Note that, dAi/dZj is -Yi*Yj when i is not equal to j,
            % while dAi/dZj is (1-Yi)*Yj = -Yi*Yi+Yi when i is equal to j
            dZ_A{i} = - Y(i,:).*Y;
            dZ_A{i}(i,:) = dZ_A{i}(i,:) + Y(i,:);
        end
end

switch typeCost
    case 'LogLoss'
        % in this case, assume P = 1    
        dZ = (-(Yd./Y+(1-Yd)./(Y-1))).*dZ_A;
    case 'CrossEntropy'
        dZ = zeros(P,M);
        % dE/dZi should be the sum of (dE/dAj)*(dAj/dZi),
        % where (dE/dAj) is -Yd(i,:)./Y(i,:)
        for i = 1:P
            dZ = dZ - (Yd(i,:)./Y(i,:)).*dZ_A{i};
        end
end


end

