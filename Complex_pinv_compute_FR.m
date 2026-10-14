function [X,r] = Complex_pinv_compute_FR(A1,A2,tolPivot,tolReduced)
%COMPLEX_PINV_COMPUTE_FR Pseudoinverse using complex full-rank factors.
%   A = A1 + A2*j. Factor A* and apply the reduced-inverse formula.
%   X is a QTFM quaternion matrix; r is the accepted pivot count.
%   tolPivot and tolReduced are separate absolute thresholds; [] uses
%   the defaults in Complex_FR and Complex_X2L_compute, respectively.
%   Requires Complex_X2L_compute.m, Complex_FR.m and QTFM for output.

if nargin < 3, tolPivot = []; end
if nargin < 4, tolReduced = []; end
if ~isfloat(A1) || ~isfloat(A2) || ...
        ~ismatrix(A1) || ~ismatrix(A2) || ...
        ~isequal(size(A1),size(A2)) || ~strcmp(class(A1),class(A2))
    error('Complex_pinv_compute_FR:InputType', ...
          'A1 and A2 must be same-sized floating-point matrices of the same class.');
end

% (A1 + A2*j)* = A1' - A2.'*j.
% Choosing W2 = A* makes the prescribed outer inverse the pseudoinverse.
% The called function validates finiteness, factorizes A*, checks its
% reduced matrix, solves the reduced system and constructs X.
[X,r] = Complex_X2L_compute(A1,A2,A1',-A2.',tolPivot,tolReduced);
end
