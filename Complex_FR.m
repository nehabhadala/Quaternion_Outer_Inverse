function [B1,B2,C1,C2,r] = Complex_FR(A1,A2,tolPivot)
%COMPLEX_FR Quaternion full-rank decomposition using complex components.
% A = A1 + A2*j, B = B1 + B2*j, C = C1 + C2*j.
% Computes A approximately B*C.
% Omit tolPivot or use [] for a scale-aware pivot cutoff.
% r is the number of accepted pivots.

if ~isfloat(A1) || ~isfloat(A2) || ~ismatrix(A1) || ~ismatrix(A2) || ...
        ~isequal(size(A1),size(A2)) || ~strcmp(class(A1),class(A2))
    error('Complex_FR:InvalidInput', ...
        'A1 and A2 must be same-sized floating-point matrices of the same class.');
end

[m,n] = size(A1);

magA = hypot(abs(A1),abs(A2));

if any(~isfinite(magA(:)))
    error('Complex_FR:InvalidInput', ...
        'Input quaternion magnitudes must be finite.');
end

if isempty(magA)
    scaleA = 0;
else
    scaleA = max(magA(:));
end

if nargin < 3 || isempty(tolPivot)
    tolPivot = max(m,n)*eps(class(magA))*scaleA;
end

if ~isnumeric(tolPivot) || ~isscalar(tolPivot) || ...
        ~isreal(tolPivot) || ~isfinite(tolPivot) || tolPivot < 0
    error('Complex_FR:InvalidTolerance', ...
        'tolPivot must be a nonnegative finite real scalar.');
end

q = min(m,n);
M = [A1 A2];
F = zeros(m,2*q,'like',M);
p = 1:m;
r = 0;

for j = 1:n

    k = r+1;

    if k > m
        break;
    end

    % Select the largest-magnitude pivot in the active column.
    d = hypot(abs(M(k:m,j)),abs(M(k:m,j+n)));
    [pivotMag,iLocal] = max(d);

    if pivotMag <= tolPivot
        continue;
    end

    i = k+iLocal-1;

    % Exchange rows and previously computed multipliers.
    if i ~= k
        M([k i],:) = M([i k],:);

        if r > 0
            previous = [1:r, q+1:q+r];
            F([k i],previous) = F([i k],previous);
        end

        p([k i]) = p([i k]);
    end

    F(k,k) = 1;

    if k < m

        a = M(k,j);
        b = M(k,j+n);
        aNorm = hypot(abs(a),abs(b));

        % Compute entry*pivot^(-1) using scaled components.
        aUnit = complex(real(a)/aNorm,imag(a)/aNorm);
        bUnit = complex(real(b)/aNorm,imag(b)/aNorm);

        inverseUnitBlock = ...
            [conj(aUnit) -bUnit; conj(bUnit) aUnit];

        entries = M(k+1:m,[j j+n]);
        scaledEntries = ...
            complex(real(entries)/aNorm,imag(entries)/aNorm);

        multipliers = scaledEntries*inverseUnitBlock;

        multiplierMag = ...
            hypot(abs(multipliers(:,1)),abs(multipliers(:,2)));

        if any(~isfinite(multiplierMag(:)))
            error('Complex_FR:ArithmeticFailure', ...
                'Nonfinite multipliers. Rescale A and retry.');
        end

        F(k+1:m,[k k+q]) = multipliers;

        row1 = M(k,1:n);
        row2 = M(k,n+1:2*n);
        rowBlock = [row1 row2; -conj(row2) conj(row1)];

        updatedRows = M(k+1:m,:)-multipliers*rowBlock;

        updatedMag = hypot(abs(updatedRows(:,1:n)), ...
                           abs(updatedRows(:,n+1:2*n)));

        if any(~isfinite(updatedMag(:)))
            error('Complex_FR:ArithmeticFailure', ...
                'Nonfinite elimination result. Rescale A and retry.');
        end

        M(k+1:m,:) = updatedRows;
        M(k+1:m,[j j+n]) = 0;
    end

    r = r+1;
end

% Restore the original row order.
B1 = zeros(m,r,'like',M);
B2 = zeros(m,r,'like',M);

B1(p,:) = F(:,1:r);
B2(p,:) = F(:,q+1:q+r);

C1 = M(1:r,1:n);
C2 = M(1:r,n+1:2*n);

end