%clear all;
% load Grqc50
% load Grqc_seeds5p5n_negposnonbr

load Astro50
load Astro_seeds5p5n_negposnonbr
m = 10000;
train = train(1:m, 1:m);
test = test(1:m, 1:m);

% load Hepth50
% load Hepth_seeds5p5n_negposnonbr

% load Hepph50
% load Hepph_seeds5p5n_negposnbr
% m = 10000;
% train = train(1:m, 1:m);
% test = test(1:m, 1:m);

% load lastfm50
% load lastfm_seeds

% load protein50
% load protein_seeds

% load airport50
% load airport_seeds

% load oregon50
% load oregon_seeds

% load email50
% load email_seeds
% m = 10000;
% train = train(1:m, 1:m);
% test = test(1:m, 1:m);

% load nba50
% load nba_seeds


% load gene50
% load gene_seeds
% m = 10000;
% train = train(1:m, 1:m);
% test = test(1:m, 1:m);


n = size(train,1);
c = 0.8;

A0 = train;

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
running_time = 0;

ttest_map_prosin = zeros(1, min(200, length(seeds)));


for i =1:min(200, length(seeds))
    i
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
    
    
    tic
    [~, A] = ProSIN(A, s, Pos, Neg, 1, c);
    running_time = running_time + toc
    %P = BLin_W2P(A, 0);
    P = A;
    rs = sparse(n, 1);
    es = sparse(n, 1);
    es(s) = 1;
    for j = 1:100
        rs = c * P * rs + (1-c) * es;
    end
    
    
    if any(actual)
        count = count + 1;
        predict = rs(testSet);
        
        AUC = AUC + computeAUC(actual, predict);
        MAP = MAP + computeAP(actual, predict);
        
        ttest_map_prosin(i) = computeAP(actual, predict);
         
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

fprintf('Total Time: %f\n', running_time);
