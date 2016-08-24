function [Ru, Ru_max] = computeHLU(real, predict)
% real is {0,1} vector
% predict is [0,1] vector


[t,ind] = sort(predict,'descend');
rsize = length(real);
real = real(ind);

tmp = ((1:rsize)'-1)./4;
tmp = 2.^tmp;
Ru = sum(real./tmp);

isize = sum(real);
if isize == 0
    Ru_max = 0;
else
    tmp = ((1:isize)-1)/4;
    tmp = 2.^tmp;
    Ru_max = sum(1./tmp);
end

end

