function AP = computeAP(real, predict)
% real is {0,1} vector
% predict is [0,1] vector
% topN is 10

isize = sum(real);
if isize == 0
    AP = 0;
    return;
end

[t,ind] = sort(predict,'descend');
rsize = length(real);

real = real(ind);
hits = cumsum(real);
prec = hits./(1:rsize)';
prec = sum(prec.*real);

AP = prec/isize;

end

