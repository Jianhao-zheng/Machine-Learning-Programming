function [X, param1, param2] = normalize(data, normalization, param1, param2)
%NORMALIZE Normalize the data wrt to the normalization technique passed in
%parameter. If param1 and param2 are given, use them during the
%normalization step
%
%   input -----------------------------------------------------------------
%   
%       o data : (N x M), a dataset of M sample points of N features
%       o normalization : String indicating which normalization technique
%                         to use among minmax, zscore and none
%       o param1 : (optional) first parameter of the normalization to be
%                  used instead of being recalculated if provided
%       o param2 : (optional) second parameter of the normalization to be
%                  used instead of being recalculated if provided
%
%   output ----------------------------------------------------------------
%
%       o X : (N x M), normalized data
%       o param1 : first parameter of the normalization
%       o param2 : second parameter of the normalization
X_temp=data';
if normalization == "minmax"
    if nargin==2
        param1=min(X_temp)';
        param2=max(X_temp)';
    end
    X=(data-param1)./(param2-param1);
elseif normalization == "zscore"
    if nargin==2
        param1=mean(X_temp)';
        param2=std(X_temp)';
    end
    X=(data-param1)./param2;
elseif normalization == "none" 
    if nargin==2
        param1=zeros(size(data,1),1);
        param2=zeros(size(data,1),1);
    end
    X=data;
end

end

