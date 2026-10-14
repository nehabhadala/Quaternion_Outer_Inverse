function [X,t] = QTFM_X2L_compute(A,W2,tolPivot,tolReduced)
%QTFM_X2L_COMPUTE Left outer inverse from Theorem 3.12.
%   X = QTFM_X2L_compute(A,W2)
%   [X,t] = QTFM_X2L_compute(A,W2,tolPivot,tolReduced)
%
%   A is m-by-n; W2 is n-by-m (QTFM quaternion matrices).
%   Factor W2 approximately as T2*S2, then compute
%       X = T2*inv(S2*A*T2)*S2.
%   t is the accepted pivot count returned by QTFM_FR.
%   tolPivot: absolute pivot cutoff; [] uses QTFM_FR's default.
%   tolReduced: absolute singular-value cutoff for S2*A*T2;
%               [] uses t*eps(class(s))*max(s).
%   This computes a prescribed outer inverse, not generally pinv(A).
%   Requires QTFM and the supplied QTFM_FR.m.

if nargin < 3, tolPivot = []; end
if nargin < 4, tolReduced = []; end

if ~isa(A,'quaternion') || ~isa(W2,'quaternion')
    error('QTFM_X2L_compute:InputType', ...
          'A and W2 must be QTFM quaternion matrices.');
end
if numel(size(A)) > 2 || numel(size(W2)) > 2
    error('QTFM_X2L_compute:InputSize', ...
          'A and W2 must be two-dimensional.');
end
[m,n] = size(A);
if ~isequal(size(W2),[n,m])
    error('QTFM_X2L_compute:Dimensions', ...
          'For m-by-n A, W2 must have size n-by-m.');
end
magA = abs(A);
if ~isreal(magA) || any(~isfinite(magA(:)))
    error('QTFM_X2L_compute:InvalidInput', ...
          'A must have finite real quaternion magnitudes.');
end
if ~isempty(tolReduced) && (~isnumeric(tolReduced) || ...
        ~isscalar(tolReduced) || ~isreal(tolReduced) || ...
        ~isfinite(tolReduced) || tolReduced < 0)
    error('QTFM_X2L_compute:InvalidTolerance', ...
          'tolReduced must be [] or a nonnegative finite real scalar.');
end

% QTFM_FR also validates W2 and tolPivot.
% T2 is n-by-t and S2 is t-by-m.
[T2,S2,t] = QTFM_FR(W2,tolPivot);
if t == 0
    Z = zeros(n,m,'like',magA);
    X = quaternion(Z,Z,Z,Z);
    return;
end

% Form the t-by-t reduced matrix, preserving multiplication order.
W = S2*A*T2;
magW = abs(W);
if any(~isfinite(magW(:)))
    error('QTFM_X2L_compute:ArithmeticFailure', ...
          'The reduced matrix contains nonfinite values.');
end

% A scalar quaternion has one singular value: its magnitude.
if t == 1
    s = abs(W);
else
    s = svd(W);
end
s = s(:);
if any(~isfinite(s))
    error('QTFM_X2L_compute:ArithmeticFailure', ...
          'The reduced matrix has nonfinite singular values.');
end
if isempty(tolReduced)
    tolReduced = t*eps(class(s))*max(s);
end
if min(s) <= tolReduced
    error('QTFM_X2L_compute:SingularReducedMatrix', ...
          'S2*A*T2 is singular at the selected tolerance.');
end

% Direct inverse formula of Theorem 3.12.
% Keep the existing QTFM inverse operation: matrix left-division support
% in the installed toolbox has not been verified.
X = T2*inv(W)*S2;
magX = abs(X);
if any(~isfinite(magX(:)))
    error('QTFM_X2L_compute:ArithmeticFailure', ...
          'The computed X contains nonfinite values.');
end
end
