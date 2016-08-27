function [rs, tA] = ProSIN(A, s, P, N, k, c)

% Hanghang's ProSIN algorithm, see Measuring Proximity on Graphs with Side
% Information, ICDM 08'
% A: adjacency matrix, column normalized
% s: sournce node
% P: positive set
% N: negative set
% k: neighborhood size
% c: rwr parameter

A = BLin_W2P(A', 0);
n = size(A,1);

tA = A;
ns = length(find(A(:,s)));

nP = length(P);
% adjust according to Positive set
if nP>0
    tA(:,s) = (ns/(ns + nP)) * tA(:,s);
    
    for i=1:nP
        x = P(i);
        tA(x,s) = tA(x,s) + 1/(ns + nP);
    end
end

nN = length(N);
% adjust according to Negative Set
if nN>0
    for i= 1:nN
        y = N(i);
        tA = processNegNode(tA, y, k, c);
    end
end

max_iter = 100;
rs = ones(n,1)/n;
es = zeros(n,1);
es(s) = 1;
for i=1:max_iter
    rs = c*tA*rs + (1-c)*es;
end

end

function tA = processNegNode(A, y, k, c)

n = size(A, 1);
[It, Jt, Vt] = find(A);
tA = sparse(It, Jt, Vt,n+1, n+1); 

max_iter = 100;
ry = ones(n,1)/n;
ey = zeros(n,1);
ey(y) = 1;
for i=1:max_iter
    ry = c*A*ry + (1-c)*ey;
end

[~, sortInd] = sort(ry, 'descend');
maxKind = sortInd(1:k);

for i =1:k
    node_i = maxKind(i);
    tA(n+1,node_i) = ry(node_i)/ry(y);
    tA(1:n, node_i) = (1-ry(node_i)/ry(y)) * tA(1:n, node_i);
end

tA = tA(1:n,1:n);
tA = sparse(tA);
end
