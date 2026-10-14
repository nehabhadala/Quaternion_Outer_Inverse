clc; clear; close all;
rng(222,'twister');             % New fixed seed for reproducible noise.
tol = 1e-10;
sigma = 10; beta = 8/3; rho = 28;
T = 50; dt = 0.06;
tspan = 0:dt:T;
y0 = [1 1 1];
lorenz = @(t,y) [sigma*(y(2)-y(1)); y(1)*(rho-y(3))-y(2); y(1)*y(2)-beta*y(3)];
options = odeset('RelTol',1e-3,'AbsTol',1e-6);
[t,Y] = ode45(lorenz,tspan,y0,options);
noise = 30*randn(length(t)-1,3);
input = Y(1:end-1,:)+noise;
Xq = quaternion(zeros(size(input,1),1),input(:,1),input(:,2),input(:,3));
Yq = quaternion(zeros(size(Y,1),1),Y(:,1),Y(:,2),Y(:,3));
m = floor((length(t)-1)/2)+1;
N = length(t)-m;
A = zerosq(N,m);
for i = 1:N
    A(i,:) = Xq(i+m-1:-1:i).';
end
y = Yq(m+1:m+N);
A1 = complex(A.s,A.x); A2 = complex(A.y,A.z);
w = Complex_pinv_compute(A1,A2,tol)*y;
yhat = A*w;
relativeError = norm(yhat-y,'fro')/norm(y,'fro');
fprintf('Training relative residual: %.6e\n',relativeError);

noisy = input(m:end,:);
clean = [y.x y.y y.z];
filtered = [yhat.x yhat.y yhat.z];
times = t(m+1:end);
figure;
for channel = 1:3
    subplot(3,3,(channel-1)*3+1); plot(times,noisy(:,channel)); grid on;
    title('Noisy delayed input'); xlabel('t');
    subplot(3,3,(channel-1)*3+2); plot(times,clean(:,channel)); grid on;
    title('Target'); xlabel('t');
    subplot(3,3,(channel-1)*3+3); plot(times,filtered(:,channel)); grid on;
    title('Filtered'); xlabel('t');
end
saveas(gcf,'Figure4.png');
