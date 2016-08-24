function pK = precisionAtK(actual, predict, K)
    [~, I] = sort(predict, 'descend');
    nHits = sum(actual(I(1:K)));
    pK = nHits/K;
end