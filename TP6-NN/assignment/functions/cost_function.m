function [E] = cost_function(Y, Yd, type)
%COST_FUNCTION compute the error between Yd and Y
%   inputs:
%       o Y (PxM) Output of the last layer of the network, should match
%       Y
%       o Yd (PxM) Ground truth
%       o type (string) type of the cost evaluation function
%   outputs:
%       o E (scalar) The error
[~,M] = size(Y);

switch type
    case 'LogLoss'
        % in this case, assume P = 1 since it's for the binary
        % classification
        E = -sum(Yd.*log(Y)+(1-Yd).*log(1-Y))/M;
    case 'CrossEntropy'
        E = -sum(sum(Yd.*log(Y)))/M;
end
        
end

