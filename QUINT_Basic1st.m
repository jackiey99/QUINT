clear all;

load Astro50
load Astro_seeds5p5n_negposnonbr

m = 10000;
train = train(1:m, 1:m);
test = test(1:m, 1:m);

n = size(train,1);
A0 = train;
c = 0.5*1/eigs(A0, 1);

count = 0;
AUC = 0;
MAP = 0;
MPR = 0;
precAt10 = 0;
precAt20 = 0;
recallAt1 = 0;
recallAt5 = 0;
HLU1 = 0;
HLU2 = 0;
MPR1 = 0;
MPR2 = 0;

for i =1:min(200, length(seeds))
    s = seeds{i}.s;
    Pos = seeds{i}.Pos;
    Neg = seeds{i}.Neg;
    topKitems = [Pos, Neg];
    
    testSet  = setdiff(1:n, [find(train(s, :)), topKitems]);
    actual = test(s, testSet)';
    
    if any(actual) == 0
        continue;
    end
    
    A = A0;
    % add noise
    A(s, Neg) = 1;
    A(Neg, s) = 1;
    Nbr_s = find(train(s, :));
    Nbr_n = [];
    for nn = 1:length(Neg)
        Nbr_n = [Nbr_n , find(train(Neg(nn), :))];
    end
    A(Nbr_s, Nbr_n) = A(Nbr_s, Nbr_n) + 0.1;
    A(Nbr_n, Nbr_s) = A(Nbr_n, Nbr_s) + 0.1;
    
    % begin update
    diff_Qdl = sparse(1, size(A0,1));
    b = 1;
    for pp = 1:length(Pos)
        for nn = 1:length(Neg)
            xd = Pos(pp);
            xl = Neg(nn);
            [qs, qd, ql] = extractQCol1st(A, s, xd, xl, c);
            x = qs(xl) - qs(xd);
            hx = 1/(1 + exp(-x/b));
            
            % compute gradient
            diff_Qdl = diff_Qdl + (1/b) * hx * (1-hx) * c * (ql' - qd');
        end
    end
    diff_Qdl = qs * diff_Qdl;
    
    % only update the neighborhoods
    Nbr_s = [s, find(train(s, :))];
    for pp = 1:length(Pos)
        xd = Pos(pp);
        Nbr_p = [xd, find(train(xd, :))];
        sq = diff_Qdl(Nbr_s, Nbr_p);
        
        t = 1/abs(min(sq(:)));
        A(Nbr_s, Nbr_p)  = A(Nbr_s, Nbr_p) - t * sq;
        A(Nbr_p, Nbr_s) = A(Nbr_p, Nbr_s) - t * sq';
    end
  
    for nn = 1:length(Neg)
        xl = Neg(nn);
        Nbr_n = [xl, find(train(xl, :))];
        sq = diff_Qdl(Nbr_s, Nbr_n);

        t = 10^floor(abs(log10(min(sq(:)))));
        A(Nbr_s, Nbr_n)  = A(Nbr_s, Nbr_n) -  t * sq;
        A(Nbr_n, Nbr_s) = A(Nbr_n, Nbr_s) - t  * sq';
    end
    
      A = max(A, 0);
 %     A(A > 0) = 1;
    
    % evaluate
    cc = 0.8;
    P = BLin_W2P(A, 0);
    
    rs = sparse(n, 1);
    es = sparse(n, 1);
    es(s) = 1;
    for j = 1:100
        rs = cc * P * rs + (1-cc) * es;
    end
    
    if any(actual)
        count = count + 1;
        predict = rs(testSet);
        
        AUC = AUC + computeAUC(actual, predict);
        MAP = MAP + computeAP(actual, predict);
                
        precAt10 = precAt10 + precisionAtK(actual, predict, 10);
        precAt20 = precAt20 + precisionAtK(actual, predict, 20);
        recallAt1 = recallAt1 + recallAtK(actual, predict, 1);
        recallAt5 = recallAt5 + recallAtK(actual, predict, 5);
        
        [H1, H2] = computeHLU(actual, predict);
        HLU1 = HLU1 + H1;
        HLU2 = HLU2 + H2;
        [PR1, PR2] = computePR(actual, predict);
        MPR1 = MPR1 + PR1;
        MPR2 = MPR2 + PR2;
    end
end

MAP = MAP / count;
AUC = AUC / count;
precAt10 = precAt10 / count;
precAt20 = precAt20 / count;
recallAt1 = recallAt1 /count;
recallAt5 = recallAt5 /count;
HLU = 100 * HLU1/HLU2;
MPR = MPR1 / MPR2;

fprintf('MAP = %f, MPR = %f, HLU %f, AUC = %f, P10 = %f, P20 = %f, R1 = %f, R5 = %f\n', MAP, MPR, HLU, AUC, precAt10, precAt20, recallAt1, recallAt5);
