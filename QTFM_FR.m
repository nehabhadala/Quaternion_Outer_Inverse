function [B,C,r] = QTFM_FR(A,tolPivot)
%QTFM_FR Numerical full-rank decomposition using QTFM.
% Computes A approximately B*C using partial row pivoting.
% B is m-by-r; C is r-by-n.
% r is the number of accepted pivots.
%
% Omit tolPivot or use [] for a scale-aware pivot cutoff.
% tolPivot=0 accepts every strictly nonzero computed pivot.
% This is a pivot cutoff, not a singular-value cutoff.

if ~isa(A,'quaternion')
    error('QTFM_FR:InputType', ...
        'A must be a QTFM quaternion matrix.');
end

sz = size(A);

if numel(sz) > 2
    error('QTFM_FR:InputSize','A must be two-dimensional.');
end

[m,n] = size(A);

magA = abs(A);

if ~isreal(magA) || any(~isfinite(magA(:)))
    error('QTFM_FR:InvalidInput', ...
        'A must have finite, real quaternion magnitudes.');
end

if isempty(magA)
    scaleA = 0;
else
    scaleA = max(magA(:));
end

if nargin < 2 || isempty(tolPivot)
    if isa(magA,'single')
        unitRoundoff = eps('single');
    else
        unitRoundoff = eps('double');
    end

    tolPivot = (max(m,n)*unitRoundoff)*scaleA;
end

if ~isnumeric(tolPivot) || ~isscalar(tolPivot) || ...
        ~isreal(tolPivot) || ~isfinite(tolPivot) || tolPivot < 0
    error('QTFM_FR:InvalidTolerance', ...
        'tolPivot must be a nonnegative finite real scalar.');
end

M = A;

Z = zeros(m,min(m,n),'like',magA);
F = quaternion(Z,Z,Z,Z);

p = 1:m;
r = 0;

for j = 1:n

    k = r+1;

    if k > m
        break;
    end

    % Select the largest-magnitude pivot in the active column.
    d = abs(M(k:m,j));
    [pivotMag,iLocal] = max(d);

    if pivotMag <= tolPivot
        continue;
    end

    i = k+iLocal-1;

    % Exchange rows and previously computed multipliers.
    if i ~= k
        M([k i],:) = M([i k],:);
        p([k i]) = p([i k]);

        if r > 0
            F([k i],1:r) = F([i k],1:r);
        end
    end

    F(k,k) = quaternion(1,0,0,0);

    if k < m

        a = M(k,j);
        aNorm = abs(a);

        % Compute entry*a^(-1) using scaled quantities.
        % Quaternion multiplication order is essential.
        pivotUnit = a/aNorm;
        scaledEntries = M(k+1:m,j)/aNorm;
        multipliers = scaledEntries*conj(pivotUnit);

        multiplierMag = abs(multipliers);

        if any(~isfinite(multiplierMag(:)))
            error('QTFM_FR:ArithmeticFailure', ...
                'Nonfinite multipliers. Rescale A and retry.');
        end

        F(k+1:m,k) = multipliers;

        % Update whole rows, including previously skipped columns.
        updatedRows = M(k+1:m,:) - multipliers*M(k,:);
        updatedMag = abs(updatedRows);

        if any(~isfinite(updatedMag(:)))
            error('QTFM_FR:ArithmeticFailure', ...
                'Nonfinite elimination result. Rescale A and retry.');
        end

        M(k+1:m,:) = updatedRows;

        Z = zeros(m-k,1,'like',magA);
        M(k+1:m,j) = quaternion(Z,Z,Z,Z);
    end

    r = k;
end

% Restore the original row order.
Z = zeros(m,r,'like',magA);
B = quaternion(Z,Z,Z,Z);

B(p,:) = F(:,1:r);
C = M(1:r,:);

end