function [labels, Mu, Mu_init, iter] =  kmeans(X,K,init,type,MaxIter,plot_iter)
%MY_KMEANS Implementation of the k-means algorithm
%   for clustering.
%
%   input -----------------------------------------------------------------
%   
%       o X        : (N x M), a data set with M samples each being of dimension N.
%                           each column corresponds to a datapoint
%       o K        : (int), chosen K clusters
%       o init     : (string), type of initialization {'sample','range'}
%       o type     : (string), type of distance {'L1','L2','LInf'}
%       o MaxIter  : (int), maximum number of iterations
%       o plot_iter: (bool), boolean to plot iterations or not (only works with 2d)
%
%   output ----------------------------------------------------------------
%
%       o labels   : (1 x M), a vector with predicted labels labels \in {1,..,k} 
%                   corresponding to the k-clusters for each points.
%       o Mu       : (N x k), an Nxk matrix where the k-th column corresponds
%                          to the k-th centroid mu_k \in R^N 
%       o Mu_init  : (N x k), same as above, corresponds to the centroids used
%                            to initialize the algorithm
%       o iter     : (int), iteration where algorithm stopped
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% TEMPLATE CODE (DO NOT MODIFY)
% Auxiliary Variable
[D, N] = size(X);
d_i    = zeros(K,N);
k_i    = zeros(1,N);
r_i    = zeros(K,N);
if plot_iter == [];plot_iter = 0;end
tolerance = 1e-6;
MaxTolIter = 10;

% Output Variables
Mu     = zeros(D, K);
labels = zeros(1,N);


%% INSERT CODE HERE
%make sure every cluster is not empty in the beginning
while (~check_empty (r_i))  
    iter = 0;
    tol_iter = 0;
    Mu =  kmeans_init(X, K, init);
    Mu_init = Mu;
    %% TEMPLATE CODE (DO NOT MODIFY)
    % Visualize Initial Centroids if N=2 and plot_iter active
    colors     = hsv(K);
    if (D==2 && plot_iter)
        options.title       = sprintf('Initial Mu with %s method', init);
        ml_plot_data(X',options); hold on;
        ml_plot_centroids(Mu_init',colors);
    end

    %% INSERT CODE HERE
    r_i = eye(K,N);
    has_converged = false;
    while (~has_converged)
        if (~check_empty (r_i))
            break
        end
        Mu_previous = Mu;
        d_i =  distance_to_centroids(X, Mu, type);
        r_i = compute_responsibility(d_i);
        sum_r = repmat(sum(r_i,2),1,D)';
        Mu = (r_i*X')'./sum_r;
        iter = iter+1;
        [has_converged, tol_iter] = check_convergence(Mu, Mu_previous, iter, tol_iter, MaxIter, MaxTolIter, tolerance);
    end
end
[labels,~] = find(r_i==1);
labels = labels';

%% TEMPLATE CODE (DO NOT MODIFY)
if (D==2 && plot_iter)
    options.labels      = labels;
    options.class_names = {};
    options.title       = sprintf('Mu and labels after %d iter', iter);
    ml_plot_data(X',options); hold on;    
    ml_plot_centroids(Mu',colors);
end


end