% %% result Xw=y, size of X is 417*417, w is 417*1, and y is 417*1
%% ne is  Absolute error          = 3.7486e-07
%% Relative error   = 6.6076e-10


clc; clear; close all;
rng(321,'twister');  % Fixed seed for reproducible Gaussian noise.

output_folder = 'signal_images_final';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% === Global Style Defaults ===
set(0, 'DefaultAxesFontSize', 14, ...                  % tick labels
       'DefaultAxesLabelFontSizeMultiplier', 1.5, ...  % 18*1.2 ≈ 22 for axis labels
       'DefaultAxesTitleFontSizeMultiplier', 1.5, ...  % titles slightly larger
       'DefaultAxesFontWeight', 'normal', ...
       'DefaultAxesLineWidth', 1.0, ...
       'DefaultAxesBox', 'on', ...
       'DefaultAxesXColor', 'k', ...
       'DefaultAxesYColor', 'k', ...
       'DefaultAxesZColor', 'k', ...
       'DefaultLineLineWidth', 1.5);



%% Step 1: Solve the Lorenz system using ODE45
sigma = 10;  
beta = 8/3;  
rho = 28;  
T = 50;  
dt = 0.06;  
tspan = 0:dt:T;  
y0 = [1 1 1];  % Initial conditions

lorenz_eq = @(t, y) [sigma * (y(2) - y(1));  
                      y(1) * (rho - y(3)) - y(2);  
                      y(1) * y(2) - beta * y(3)];

[t, Y] = ode45(lorenz_eq, tspan, y0);


%% 0. 3D Lorenz Attractor
%fs = 10; % Font size for labels and ticks

figure;
plot3(Y(:, 1), Y(:, 2), Y(:, 3), 'b');
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$y$', 'Interpreter', 'latex');
zlabel('$z$', 'Interpreter', 'latex');
set(gca, 'XTick', -20:10:25, ...
         'YTick', -30:10:30, ...
         'ZTick', 0:10:50);
grid on; view(45,25);
box on;    % ✅ Ensure full box


    % Save figure
%saveas(gcf, fullfile(output_folder, 'lorentzsystem.png'));
print(fullfile(output_folder, 'lorentzsystem'), '-depsc');

xr = Y(:,1);  
xg = Y(:,2);  
xb = Y(:,3);
L = length(t)
l = floor((L - 1) / 2);  % Ensure integer N

%% Step 2: Construct quaternion representation using MATLAB's quaternion function
Y_quat = quaternion(zeros(size(xr)), xr, xg, xb);  % Y_quat = 0 + xr*i + xg*j + xb*k
disp('Size of Y_quat:');
disp(size(Y_quat));


 %% Step 3: Create the input signal (delayed version + noise)
 noise_level = 30;
 n_t = noise_level * randn(length(t)-1, 3); 
 disp('Size of n_t:');
 disp(size(n_t));
 
X_quat = quaternion(zeros(size(xr(1:end-1))), xr(1:end-1) + n_t(:,1), xg(1:end-1) + n_t(:,2), xb(1:end-1) + n_t(:,3));
 disp('Size of X_quat:');
 disp(size(X_quat));

%% Step 4: Construct the quaternion linear system
m = l+1 % Filter length
N = length(Y_quat) - m 

% Corrected: Initialize quaternion matrix properly
X_matrix = repmat(quaternion(0, 0, 0, 0), N, m);  

for idx = 1:N
    X_matrix(idx, :) = X_quat(idx+m-1:-1:idx);  % Store past values as rows
end

% Corrected: Ensure y_target has the same number of elements as rows in X_matrix
y_target = Y_quat(m+1:m+N);  % Take only the first N elements

% Initialize w (quaternion filter vector)
w = repmat(quaternion(0, 0, 0, 0), m, 1);  % Size (m,1)

X_matrix1=complex(X_matrix.s,X_matrix.x);
X_matrix2=complex(X_matrix.y,X_matrix.z);
w = Complex_pinv_compute(X_matrix1,X_matrix2)*y_target;

 
y_est = X_matrix*w;
 
e = y_est - y_target;
 
ne=norm(e)

ny = norm(y_target);       % norm of target signal
rel_err = ne / ny        % relative error

% Common time vector aligned to target signal
t_common = t(m+1:end);

% Extract components
xr_noisy = X_quat.x;
xg_noisy = X_quat.y;
xb_noisy = X_quat.z;

xr_clean = y_target.x;
xg_clean = y_target.y;
xb_clean = y_target.z;

xr_est = y_est.x;
xg_est = y_est.y;
xb_est = y_est.z;
%% 1. Noisy & Delayed Input Signal
figure;
subplot(3,1,1);
plot(t_common, xr_noisy(m:end), 'r');
ylabel('$x$', 'Interpreter', 'latex');
%set(gca, 'FontSize', fs, 'LineWidth', 0.5, 'Box', 'on', 'XColor', 'k', 'YColor', 'k');
grid on;

subplot(3,1,2);
plot(t_common, xg_noisy(m:end), 'r');
ylabel('$y$', 'Interpreter', 'latex');
%set(gca, 'FontSize', fs, 'LineWidth', 0.5, 'Box', 'on', 'XColor', 'k', 'YColor', 'k');
grid on;

subplot(3,1,3);
plot(t_common, xb_noisy(m:end), 'r');
ylabel('$z$', 'Interpreter', 'latex');
xlabel('$t$', 'Interpreter', 'latex');
%set(gca, 'FontSize', fs, 'LineWidth', 0.5, 'Box', 'on', 'XColor', 'k', 'YColor', 'k');
grid on;


% Save figure
%saveas(gcf, fullfile(output_folder, 'NoisyDelayedInput.png'));
print(fullfile(output_folder, 'NoisyDelayedInput'), '-depsc');
%% 2. Target (Clean) Signal
%% 2. Target (Clean) Signal


figure;
subplot(3,1,1);
plot(t_common, xr_clean, 'b');
ylabel('$x$', 'Interpreter', 'latex');
%set(gca, 'FontSize', fs, 'LineWidth', 0.5, 'Box', 'on', 'XColor', 'k', 'YColor', 'k');
grid on;

subplot(3,1,2);
plot(t_common, xg_clean, 'b');
ylabel('$y$', 'Interpreter', 'latex');
%set(gca, 'FontSize', fs, 'LineWidth', 0.5, 'Box', 'on', 'XColor', 'k', 'YColor', 'k');
grid on;

subplot(3,1,3);
plot(t_common, xb_clean, 'b');
ylabel('$z$', 'Interpreter', 'latex');
xlabel('$t$', 'Interpreter', 'latex');
%set(gca, 'FontSize', fs, 'LineWidth', 0.5, 'Box', 'on', 'XColor', 'k', 'YColor', 'k');
grid on;


%saveas(gcf, fullfile(output_folder, 'targetsignal.png'));
print(fullfile(output_folder, 'targetsignal'), '-depsc');
%% 3. Estimated Signal after Filtering
fs = 10; % Font size for labels and ticks

% Custom color
customColor = [0.4940 0.1840 0.5560];

figure;
subplot(3,1,1);
plot(t_common, xr_est, 'Color', customColor);
ylabel('$x$', 'Interpreter', 'latex');
%set(gca, 'FontSize', fs, 'LineWidth', 0.5, 'Box', 'on', 'XColor', 'k', 'YColor', 'k');
grid on;

subplot(3,1,2);
plot(t_common, xg_est, 'Color', customColor);
ylabel('$y$', 'Interpreter', 'latex');
%set(gca, 'FontSize', fs, 'LineWidth', 0.5, 'Box', 'on', 'XColor', 'k', 'YColor', 'k');
grid on;

subplot(3,1,3);
plot(t_common, xb_est, 'Color', customColor);
ylabel('$z$', 'Interpreter', 'latex');
xlabel('$t$', 'Interpreter', 'latex');
%set(gca, 'FontSize', fs, 'LineWidth', 0.5, 'Box', 'on', 'XColor', 'k', 'YColor', 'k');
grid on;



% Save figure
%saveas(gcf, fullfile(output_folder, 'filtersignal.png'));
print(fullfile(output_folder, 'filtersignal'), '-depsc');

fprintf('Absolute error          = %.4e\n', ne);
fprintf('Relative error   = %.4e\n', rel_err);



