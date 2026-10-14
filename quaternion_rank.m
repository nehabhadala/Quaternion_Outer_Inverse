function [r,tolUsed,s] = quaternion_rank(A,tol)
%QUATERNION_RANK SVD-based numerical rank of a QTFM quaternion matrix.
%   r = quaternion_rank(A)
%   r = quaternion_rank(A,tol)
%   [r,tolUsed,s] = quaternion_rank(A,tol)
%   tol is an absolute singular-value cutoff. [] uses
%   max(size(A))*eps(class(s))*max(s). Singular values strictly greater
%   than the cutoff are counted. This is an SVD-based rank estimate and
%   is not the same criterion as the pivot rank returned by QTFM_FR.

if ~isa(A,'quaternion') || numel(size(A)) > 2
    error('quaternion_rank:InputType', ...
          'A must be a two-dimensional QTFM quaternion matrix.');
end
if nargin < 2, tol = []; end
if ~isempty(tol) && (~isnumeric(tol) || ~isscalar(tol) || ...
        ~isreal(tol) || ~isfinite(tol) || tol < 0)
    error('quaternion_rank:InvalidTolerance', ...
          'tol must be [] or a nonnegative finite real scalar.');
end

[m,n] = size(A);
magA = abs(A);
if any(~isfinite(magA(:)))
    error('quaternion_rank:InvalidInput', ...
          'A must have finite quaternion magnitudes.');
end

if isempty(A) || all(magA(:) == 0)
    s = zeros(0,1);
    tolUsed = 0;
    r = 0;
    return;
end

if isscalar(A)
    s = abs(A);
else
    s = svd(A);
    s = s(:);
end
if ~isreal(s) || any(~isfinite(s(:)))
    error('quaternion_rank:ArithmeticFailure', ...
          'The SVD returned invalid singular values.');
end

if isempty(tol)
    tolUsed = max(m,n)*eps(class(s))*max(s(:));
else
    tolUsed = tol;
end
r = sum(s(:) > tolUsed);
end
