function X = QTFM_X_compute(A,S1,T1,tol)
% Computes X = S1*(T1*A*S1)^(1)*T1.
% tol applies to W = T1*A*S1.

if nargin < 4
    tol = [];
end

W = T1*A*S1;

G = QTFM_1_inverse(W,tol);

X = S1*G*T1;

end