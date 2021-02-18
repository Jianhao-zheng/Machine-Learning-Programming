function no_empty_cluster = check_empty (r_i)
sum_r = sum(r_i,2);
no_empty_cluster = isempty(find(sum_r == 0,1));
end