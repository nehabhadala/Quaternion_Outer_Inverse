function X = QTFM_pinv_compute(A,tol)
% Computes X = A'*(A'*A*A')^(1)*A'.
% tol applies to W = A'*A*A', not A.

if nargin < 2
    tol = [];
end

X = QTFM_X_compute(A,A',A',tol);

end