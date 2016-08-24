function  [qs, qd, ql] = extractQCol1st(A, s, xd, xl, c)

% get the s, xd, xl-th columns of inv(I - cA) fast
% first order Taylor expansion

n = size(A, 1);

%Q = speye(n) + c * A;
qs = c*A(:, s);
qs(s) = qs(s) + 1;
qd = c*A(:, xd);
qd(xd) = qd(xd) + 1;
ql = c*A(:, xl);
ql(xl) = ql(xl) + 1;

end
