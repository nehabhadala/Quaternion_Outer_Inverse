function G = QTFM_1_inverse(A,tol)
% Computes a {1}-inverse using QTFM SVD.
% The free blocks K, L, M are zero.
% Omit tol or use [] for a scale-aware cutoff.

if nargin < 2
    tol = [];
end

if ~isa(A,'quaternion')
    error('A must be a QTFM quaternion matrix.');
end

if ~isempty(tol)
    if ~isnumeric(tol) || ~isscalar(tol) || ~isreal(tol) || ...
            ~isfinite(tol) || tol <= 0
        error('tol must be a positive finite number.');
    end
end

[m,n] = size(A);

if norm(abs(A),'fro') == 0
    G = zerosq(n,m);
    return;
end

% Scalar quaternion case.
if m == 1 && n == 1
    s = abs(A);

    if isempty(tol)
        tol = eps(class(s))*s;
    end

    G = zerosq(1,1);

    if s > tol
        G = conj(A/s)/s;
    end
    return;
end

[U,S,V] = svd(A);

% Extract the diagonal correctly for rectangular and vector inputs.
d = min(m,n);
s = S(sub2ind([m,n],1:d,1:d));
s = s(:);

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