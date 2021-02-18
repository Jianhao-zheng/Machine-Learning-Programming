function [MSE, NMSE, Rsquared] = regression_metrics( yest, y )
%REGRESSION_METRICS Computes the metrics (MSE, NMSE, R squared) for 
%   regression evaluation
%
%   input -----------------------------------------------------------------
%   
%       o yest  : (P x M), representing the estimated outputs of P-dimension
%       of the regressor corresponding to the M points of the dataset
%       o y     : (P x M), representing the M continuous labels of the M 
%       points. Each label has P dimensions.
%
%   output ----------------------------------------------------------------
%
%       o MSE       : (1 x 1), Mean Squared Error
%       o NMSE      : (1 x 1), Normalized Mean Squared Error
%       o R squared : (1 x 1), Coefficent of determination
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Auxiliary Variables
[~,M] = size(y);

MSE = 0;
Var = 0;
R_Num = 0; %Numerator of R squared
R_Den_part = 0; %Second factor of Denominator of R squared

y_bar = mean(y,2);
y_bar_est = mean(yest,2);

%Assume x,y has only one dimension, i.e. P=N=1
for i = 1:M
    MSE = MSE + norm(yest(:,i)-y(:,i))^2;
    Var = Var + norm(y(:,i)-y_bar)^2;
    R_Num = R_Num + (y(:,i)-y_bar)'*(yest(:,i)-y_bar_est);
    R_Den_part = R_Den_part + norm(yest(:,i)-y_bar_est)^2;
end

MSE = MSE/M;
Var = Var/(M-1);

NMSE = MSE/Var;

R_Num = R_Num^2;
Rsquared = R_Num/((M-1)*Var*R_Den_part);

% prevent the Rsquared from being NAh when R_Num and R_Den are both 0
if R_Num == 0 && Var*R_Den_part==0
    Rsquared = 0;
end
end

