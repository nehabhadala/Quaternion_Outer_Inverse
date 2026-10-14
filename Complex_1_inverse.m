function G = Complex_1_inverse(A1,A2,tol)
% Computes a {1}-inverse using the structure-preserving SVD.
% The free blocks K, L, M are zero.
% Omit tol or use [] for a scale-aware cutoff.

if nargin < 3
    tol = [];
end

if ~isempty(tol)
    if ~isnumeric(tol) || ~isscalar(tol) || ~isreal(tol) || ...
            ~isfinite(tol) || tol <= 0
        error('tol must be a positive finite number.');
    end
end

[m,n] = size(A1);

if norm(A1,'fro') == 0 && norm(A2,'fro') == 0
    G = zerosq(n,m);
    return;
end

if min(m,n) < 2
    error('This implementation requires at least two rows and columns.');
end

[U,S,V] = CSbidiagQ2(A1,A2);
s = diag(S);

if isempty(tol)
    tol = max(m,n)*eps(class(s))*max(s);
end

D = zeros(n,m);

for k = 1:length(s)
    if s(k) > tol
        D(k,k) = 1/s(k);
    end
end

G = V*D*U';

end