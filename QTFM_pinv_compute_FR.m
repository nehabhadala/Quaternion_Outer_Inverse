function [X,r] = QTFM_pinv_compute_FR(A,tolPivot,tolReduced)
%QTFM_PINV_COMPUTE_FR Pseudoinverse from full-rank factors of A'.
%   [X,r] = QTFM_pinv_compute_FR(A)
%   [X,r] = QTFM_pinv_compute_FR(A,tolPivot,tolReduced)
%   A is m-by-n; X is n-by-m; r is the accepted pivot count.
%   tolPivot: absolute pivot cutoff; [] uses QTFM_FR's default.
%   tolReduced: absolute reduced singular-value cutoff; [] uses
%               r*eps(class(s))*max(s).
%   Requires QTFM_X2L_compute.m, QTFM_FR.m and QTFM.

if nargin < 2, tolPivot = []; end
if nargin < 3, tolReduced = []; end
if ~isa(A,'quaternion')
    error('QTFM_pinv_compute_FR:InputType', ...
          'A must be a QTFM quaternion matrix.');
end
if numel(size(A)) > 2
    error('QTFM_pinv_compute_FR:InputSize', ...
          'A must be two-dimensional.');
end

% Choosing W2 = A' gives the Moore-Penrose inverse.
% Internally: A' approximately equals S1*T1, and
% X = S1*inv(T1*A*S1)*T1.
% Reuse the same validation, zero-rank case and reduced-matrix checks.
[X,r] = QTFM_X2L_compute(A,A',tolPivot,tolReduced);
end
