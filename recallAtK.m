function rK = recallAtK(actual, predict, K)
    if K>sum(actual)
        rK = 0;
    else
        [~, I] = sort(predict, 'descend');
        hits = cumsum(actual(I));
        L = find(hits == K, 1);
        rK = 1/L;
    end
end