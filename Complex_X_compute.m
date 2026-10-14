function X = Complex_X_compute(A1,A2,S11,S12,T11,T12,tol)
% Computes X = S1*(T1*A*S1)^(1)*T1.
% tol applies to W = T1*A*S1.

if nargin < 7
    tol = [];
end

m = size(A1,1);
p = size(S11,2);

Ac = [A1 A2; -conj(A2) conj(A1)];
Sc = [S11 S12; -conj(S12) conj(S11)];
Tc = [T11 T12; -conj(T12) conj(T11)];

% Compute W = T1*A*S1 through its complex row representation.
Wr = [T11 T12]*Ac*Sc;

G = Complex_1_inverse(Wr(:,1:p),Wr(:,p+1:end),tol);

G1 = complex(G.s,G.x);
G2 = complex(G.y,G.z);
Gc = [G1 G2; -conj(G2) conj(G1)];

% Compute X = S1*G*T1.
Xr = [S11 S12]*Gc*Tc;

X1 = Xr(:,1:m);
X2 = Xr(:,m+1:end);

X = quaternion(real(X1),imag(X1),real(X2),imag(X2));

end