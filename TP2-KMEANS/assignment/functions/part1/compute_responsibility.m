function r_i = compute_responsibility(d)
K = size(d,1);
N = size(d,2);
r_i = zeros(K,N);
r_count = zeros(K,1);
for i = 1:N
    min_d = min(d(:,i));
    min_index = find(d(:,i)==min_d);
    
%% Assign the data point to the centroid with the smallest winning cluster size if a tie happens
%     if length(min_index) > 1
%         [~,index] = min(r_count(min_index));
%         min_index = min_index(index);
%     end
%%

    r_i(min_index(1),i) = 1;
    r_count(min_index) = r_count(min_index)+1;
end
end