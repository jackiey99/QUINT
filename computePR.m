function [PR_1, PR_2] = computePR(real, predict)
% real is {0,1} vector
% predict is [0,1] vector


[t,ind] = sort(predict,'descend');
rsize = length(real);
real = real(ind);
tmp = ((1:rsize)'-1)./rsize;
PR_1 = sum(real.*tmp);
PR_2 = sum(real);

end

