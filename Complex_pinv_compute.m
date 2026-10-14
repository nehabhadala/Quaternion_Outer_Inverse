function X = Complex_pinv_compute(A1,A2,tol)
% Computes X = A'*(A'*A*A')^(1)*A'.
% tol applies to W = A'*A*A', not A.

if nargin < 3
    tol = [];
end

% Complex components of A'.
S11 = A1';
S12 = -A2.';

X = Complex_X_compute(A1,A2,S11,S12,S11,S12,tol);

end