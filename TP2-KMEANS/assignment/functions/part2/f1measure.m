function [F1_overall, P, R, F1] =  f1measure(cluster_labels, class_labels)
%MY_F1MEASURE Computes the f1-measure for semi-supervised clustering
%
%   input -----------------------------------------------------------------
%   
%       o class_labels     : (1 x M),  M-dimensional vector with true class
%                                       labels for each data point
%       o cluster_labels   : (1 x M),  M-dimensional vector with predicted 
%                                       cluster labels for each data point
%   output ----------------------------------------------------------------
%
%       o F1_overall      : (1 x 1)     f1-measure for the clustered labels
%       o P               : (nClusters x nClasses)  Precision values
%       o R               : (nClusters x nClasses)  Recall values
%       o F1              : (nClusters x nClasses)  F1 values
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%In case the two inputs are n by 1 vectors instead of 1 by n.                   
%In my test, if one input is n by 1 and the other is 1 by n, the solution
%will be wrong
cluster_labels = reshape(cluster_labels,1,length(cluster_labels));
class_labels = reshape(class_labels,1,length(class_labels));

n_cluster = length(unique(cluster_labels));
n_class = length(unique(class_labels));
P = zeros(n_cluster,n_class);
R = zeros(n_cluster,n_class);
F1 = zeros(n_cluster,n_class);
F1_overall = 0;
for j = 1:n_class
    num_class_j = length(find(class_labels == j));
    for i = 1:n_cluster
        num_cluster_i = length(find(cluster_labels == i));
        L = length(find (class_labels == j & cluster_labels == i));
        P(i,j) = L/num_cluster_i;
        R(i,j) = L/num_class_j;
        if P(i,j) == 0 & R(i,j) == 0
            F1(i,j) = 0;
        else
            F1(i,j) = (2*P(i,j)*R(i,j))/(P(i,j)+R(i,j));
        end
    end
    F1_overall = F1_overall + max(F1(:,j))*num_class_j;
end
F1_overall = F1_overall/length(class_labels);






end
