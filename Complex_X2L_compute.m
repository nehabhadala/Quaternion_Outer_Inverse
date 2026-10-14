function [X,t] = Complex_X2L_compute(A1,A2,W21,W22,tolPivot,tolReduced)
%COMPLEX_X2L_COMPUTE Theorem 3.12 using Algorithm 4's complex arithmetic.
%   A = A1 + A2*j (m-by-n); W2 = W21 + W22*j (n-by-m).
%   W2 approximately equals T2*S2; X = T2*(S2*A*T2)^(-1)*S2.
%   tolPivot: absolute cutoff passed to Complex_FR; [] uses its default.
%   tolReduced: absolute singular-value cutoff; [] uses
%               t*eps(class(s))*max(s), where t is quaternion rank.
%   X is a QTFM quaternion matrix; t is the accepted pivot count.
%   QTFM is needed only to construct the quaternion output.

if nargin < 5, tolPivot = []; end
if nargin < 6, tolReduced = []; end

parts = {A1,A2,W21,W22};
for k = 1:4
    P = parts{k};
    if ~isfloat(P) || ~ismatrix(P) || ...
            ~strcmp(class(P),class(A1)) || any(~isfinite(P(:)))
        error('Complex_X2L_compute:InputType', ...
              'All components must be finite floating-point matrices of the same class.');
    end
end
[m,n] = size(A1);
if ~isequal(size(A2),[m,n]) || ...
        ~isequal(size(W21),[n,m]) || ~isequal(size(W22),[n,m])
    error('Complex_X2L_compute:Dimensions', ...
          'A1,A2 must be m-by-n and W21,W22 must be n-by-m.');
end
if ~isempty(tolReduced) && (~isnumeric(tolReduced) || ...
        ~isscalar(tolReduced) || ~isreal(tolReduced) || ...
        ~isfinite(tolReduced) || tolReduced < 0)
    error('Complex_X2L_compute:InvalidTolerance', ...
          'tolReduced must be [] or a nonnegative finite real scalar.');
end

% Full-rank factors: T2 is n-by-t and S2 is t-by-m.
[T21,T22,S21,S22,t] = Complex_FR(W21,W22,tolPivot);
if t == 0
    Z = zeros(n,m,'like',real(A1));
    X = quaternion(Z,Z,Z,Z);
    return;
end

% Full complex representations, using the paper's sign convention.
Ac = [A1 A2; -conj(A2) conj(A1)];
Sc = [S21 S22; -conj(S22) conj(S21)];
Tc = [T21 T22; -conj(T22) conj(T21)];
Wc = Sc*Ac*Tc;
if any(~isfinite(Wc(:)))
    error('Complex_X2L_compute:ArithmeticFailure', ...
          'The reduced matrix contains nonfinite values.');
end

% Each quaternion singular value appears twice in the complex matrix.
s = svd(Wc);
if any(~isfinite(s(:)))
    error('Complex_X2L_compute:ArithmeticFailure', ...
          'The reduced matrix has nonfinite singular values.');
end
if isempty(tolReduced)
    tolReduced = t*eps(class(s))*max(s);
end
if min(s) <= tolReduced
    error('Complex_X2L_compute:SingularReducedMatrix', ...
          'S2*A*T2 is singular at the selected tolerance.');
end

% Solve Wc*Yc = Sc, then use only the first row block of T2^C.
Xr = [T21 T22]*(Wc\Sc);
if any(~isfinite(Xr(:)))
    error('Complex_X2L_compute:ArithmeticFailure', ...
          'The computed X contains nonfinite values.');
end
X1 = Xr(:,1:m);
X2 = Xr(:,m+1:2*m);
X = quaternion(real(X1),imag(X1),real(X2),imag(X2));
end
