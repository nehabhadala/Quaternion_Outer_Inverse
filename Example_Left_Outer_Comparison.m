% LEFT OUTER-INVERSE COMPARISON: COMPLEX AND QTFM
%
% Purpose:
%   Compare Complex_X2L_compute and QTFM_X2L_compute.
%   Each method computes a full-rank factorization
%       W2 approximately equals T2*S2,
%   and constructs
%       X = T2*(S2*A*T2)^(-1)*S2.
%
% Input matrices:
%   For k=5:5:100, set m=3k and n=2k.
%   Generate quaternion matrices:
%       A: m-by-n, W2: n-by-m.
%   Their four real components are generated using rand.
%   Both methods receive exactly the same input matrices.
%   Set rng(321,'twister') once before matrix generation.
%   One matrix pair is generated per k and reused for all trials.
%
% Tolerance policy:
%   1. Compute one common pivot cutoff outside timing:
%        tolPivot = max(m,n)*eps*max_ij |W2_ij|.
%      Pass this identical value to both FR implementations.
%      Accept a pivot only when its magnitude exceeds tolPivot.
%      This is a pivot cutoff, not a singular-value cutoff.
%
%   2. Set tolReduced=[] so that each inverse function computes:
%        tolReduced = r*eps*sigma_max(R),
%      where R=S2*A*T2 and r is its accepted quaternion pivot count.
%      Both methods use the same formula, but the numerical cutoff
%      values may differ because the computed factors may differ.
%      For r>0, reject R if sigma_min(R)<=tolReduced.
%      This is an invertibility check, not singular-value truncation.
%
%   3. Compute independent numerical ranks of A and W2 using
%      quaternion_rank with its default scale-aware SVD cutoff:
%        tolM = max(size(M))*eps*sigma_max(M).
%      These ranks are diagnostic and do not determine the FR pivots.
%      Differences between rank(W2) and the FR pivot counts generate
%      a warning; results are not automatically discarded.
%
%   eps denotes the machine spacing at unity for the working precision.
%
% Timing:
%   Run both inverse functions once before the timing loop.
%   Perform 50 repetitions on the same matrix pair.
%   Measure complete inverse-function calls using tic/toc.
%   Report the arithmetic mean of the elapsed execution times.
%   Internal factorization and reduced-matrix checks are included.

%
%   Matrix generation, external rank diagnostics, residual evaluation,
%   plotting and file output are excluded from timing.

%
% Verification:
%   Using the final computed outputs, report:
%       Outer = ||X*A*X-X||_F / ||X||_F
%       Range = ||W2*A*X-W2||_F / ||W2||_F
%       Null  = ||X*A*W2-W2||_F / ||W2||_F
%   These are relative equation residuals.
%   Quaternion Frobenius norms are computed as norm(abs(Q),'fro').
%   Denominators are bounded below by realmin.
%   An error raised by either inverse function stops the script.
%
% Output:
%   Columns are ordered as Complex, QTFM.
%   Save elapsed times, residuals, rank diagnostics and the shared
%   pivot cutoff in TXT, CSV and MAT files.
%   Save four separate plots with linear axes in PNG and EPS:
%   time, outer-inverse residual, range residual and null residual.
%   Output folder: Example_Left_Outer_Comparison_output.
%
% Requirements:
%   QTFM, quaternion_rank, Complex_FR, QTFM_FR,
%   Complex_X2L_compute and QTFM_X2L_compute.

clc;
clear;
close all;
rng(321,'twister');



%% 1. Settings
k_values = 5:5:100;
num_trials = 50;

outdir = fullfile(fileparts(mfilename('fullpath')), ...
    'Example_Left_Outer_Comparison_output');

if ~exist(outdir,'dir')
    mkdir(outdir);
end

N = numel(k_values);
methodNames = {'Complex','QTFM'};

% Columns: Complex, QTFM.
Time = zeros(N,2);
OuterError = zeros(N,2);
RangeError = zeros(N,2);
NullError = zeros(N,2);

% Columns: rank(A), rank(W2), pivot count Complex, pivot count QTFM.
Ranks = zeros(N,4);
TolPivot = zeros(N,1);

%% 2. Computation
for idx = 1:N

    k = k_values(idx);
    m = 3*k;
    n = 2*k;

    % Generate A: m-by-n.
    A0 = rand(m,n);
    Ai = rand(m,n);
    Aj = rand(m,n);
    Ak = rand(m,n);

    A11 = complex(A0,Ai);
    A12 = complex(Aj,Ak);
    A = quaternion(A0,Ai,Aj,Ak);

    % Generate W2: n-by-m.
    W0 = rand(n,m);
    Wi = rand(n,m);
    Wj = rand(n,m);
    Wk = rand(n,m);

    W21 = complex(W0,Wi);
    W22 = complex(Wj,Wk);
    W2 = quaternion(W0,Wi,Wj,Wk);

    % Independent rank diagnostics, outside timing.
    rankA = quaternion_rank(A,[]);
    rankW2 = quaternion_rank(W2,[]);

    % One common pivot cutoff for both FR implementations.
    magW2 = hypot(abs(W21),abs(W22));
    tolPivot = max(m,n)*eps(class(magW2))*max(magW2(:));

    % Each inverse function computes its own default reduced cutoff.
    tolReduced = [];

    TolPivot(idx) = tolPivot;

    %% 3. Warm-up, excluded from timing
    % Each function checks whether its reduced matrix is invertible.
    [~,rC] = Complex_X2L_compute( ...
        A11,A12,W21,W22,tolPivot,tolReduced);

    [~,rQ] = QTFM_X2L_compute( ...
        A,W2,tolPivot,tolReduced);

    Ranks(idx,:) = [rankA,rankW2,rC,rQ];

    % Record disagreements without treating SVD rank as a target rank.
    if rC ~= rankW2 || rQ ~= rankW2
        warning('Example_Left:RankDifference', ...
            ['k=%d: SVD rank(W2)=%d, Complex pivot count=%d, ' ...
             'QTFM pivot count=%d.'],k,rankW2,rC,rQ);
    end

    %% 4. Original timing procedure
    total_time_complex = 0;
    total_time_qtfm = 0;

    for trial = 1:num_trials

        tic;
        Xc = Complex_X2L_compute( ...
            A11,A12,W21,W22,tolPivot,tolReduced);
        total_time_complex = total_time_complex + toc;

        clear functions;

        tic;
        Xq = QTFM_X2L_compute( ...
            A,W2,tolPivot,tolReduced);
        total_time_qtfm = total_time_qtfm + toc;

        clear functions;
    end

    Time(idx,1) = total_time_complex/num_trials;
    Time(idx,2) = total_time_qtfm/num_trials;

    %% 5. Relative residuals, outside timing
    normW2 = max(norm(abs(W2),'fro'),realmin);
    Xvalues = {Xc,Xq};

    for method = 1:2

        X = Xvalues{method};
        normX = max(norm(abs(X),'fro'),realmin);

        % Outer-inverse equation: X*A*X = X.
        OuterError(idx,method) = ...
            norm(abs(X*A*X-X),'fro')/normX;

        % Left range equation: W2*A*X = W2.
        RangeError(idx,method) = ...
            norm(abs(W2*A*X-W2),'fro')/normW2;

        % Left null-space equation: X*A*W2 = W2.
        NullError(idx,method) = ...
            norm(abs(X*A*W2-W2),'fro')/normW2;
    end

    fprintf('\nk=%d: A is %d-by-%d\n',k,m,n);

    fprintf('%-10s %14s %14s %14s %14s\n', ...
        'Method','Time(s)','Outer','Range','Null');

    for method = 1:2
        fprintf('%-10s %14.6e %14.6e %14.6e %14.6e\n', ...
            methodNames{method},Time(idx,method), ...
            OuterError(idx,method),RangeError(idx,method), ...
            NullError(idx,method));
    end
end

%% 6. Save an aligned text report
fid = fopen(fullfile(outdir,'Left_Outer_results.txt'),'w');

if fid < 0
    error('Cannot open the results file.');
end

fprintf(fid,'LEFT OUTER INVERSE: COMPLEX AND QTFM\n');
fprintf(fid,'Seed: 321 (twister); repetitions: %d\n',num_trials);
fprintf(fid,'Dimensions: A is 3k-by-2k; W2 is 2k-by-3k.\n');
fprintf(fid,'Both methods receive the same pivot cutoff.\n');
fprintf(fid,'Reduced cutoffs use each function''s scale-aware default.\n');
fprintf(fid,'Time: mean tic/toc elapsed seconds.\n');
fprintf(fid,'Original clear-functions timing procedure retained.\n');
fprintf(fid,'Rank diagnostics and residuals excluded from timing.\n\n');

fprintf(fid,'Outer = ||X*A*X-X||_F / ||X||_F\n');
fprintf(fid,'Range = ||W2*A*X-W2||_F / ||W2||_F\n');
fprintf(fid,'Null  = ||X*A*W2-W2||_F / ||W2||_F\n\n');

fprintf(fid,'%5s %-10s %14s %14s %14s %14s\n', ...
    'k','Method','Time(s)','Outer','Range','Null');

fprintf(fid,'%s\n',repmat('-',1,80));

for idx = 1:N
    for method = 1:2
        fprintf(fid,'%5d %-10s %14.6e %14.6e %14.6e %14.6e\n', ...
            k_values(idx),methodNames{method},Time(idx,method), ...
            OuterError(idx,method),RangeError(idx,method), ...
            NullError(idx,method));
    end
end

fprintf(fid,'\nRANK DIAGNOSTICS AND PIVOT CUTOFF\n');

fprintf(fid,'%5s %10s %10s %10s %10s %16s\n', ...
    'k','rankA','rankW2','rComplex','rQTFM','tolPivot');

for idx = 1:N
    fprintf(fid,'%5d %10d %10d %10d %10d %16.8e\n', ...
        k_values(idx),Ranks(idx,:),TolPivot(idx));
end

fprintf(fid,'\nMATLAB: %s\nPlatform: %s\n',version,computer);
fclose(fid);

%% 7. Save numerical data for separate plotting
Results = array2table( ...
    [k_values(:),Time,OuterError,RangeError,NullError,Ranks,TolPivot], ...
    'VariableNames', { ...
    'k','Time_Complex_s','Time_QTFM_s', ...
    'Outer_Complex','Outer_QTFM', ...
    'Range_Complex','Range_QTFM', ...
    'Null_Complex','Null_QTFM', ...
    'RankA','RankW2','PivotRank_Complex','PivotRank_QTFM', ...
    'TolPivot'});

writetable(Results,fullfile(outdir,'Left_Outer_results.csv'));

save(fullfile(outdir,'Left_Outer_results.mat'), ...
    'Results','k_values','num_trials','methodNames', ...
    'Time','OuterError','RangeError','NullError','Ranks','TolPivot');

%% 8. Four separate plots: left outer inverse

methodNames = {'Complex','QTFM'};

plotData = {Time,OuterError,RangeError,NullError};

yLabels = { ...
    'Mean elapsed time (s)', ...
    '$\epsilon_{\mathrm{out}}$', ...
    '$\epsilon_{\mathrm{range}}$', ...
    '$\epsilon_{\mathrm{null}}$'};

fileNames = { ...
    'Left_CPU_time', ...
    'Left_outer_error', ...
    'Left_range_error', ...
    'Left_null_error'};

% Appearance settings: identical in both examples.
figureWidthCm = 12;
figureHeightCm = 9;

axisNumberFontSize = 14;
axisNameFontSize = 20;
legendFontSize = 14;

curveLineWidth = 2;
markerSize = 7;
pngResolution = 600;

colors = [0.000 0.447 0.741; ...
          0.850 0.325 0.098];

markers = {'d','o'};  % Complex: diamond; QTFM: circle.

for plotIndex = 1:numel(plotData)

    Y = plotData{plotIndex};

    fig = figure('Color','w', ...
        'Units','centimeters', ...
        'Position',[2 2 figureWidthCm figureHeightCm], ...
        'PaperUnits','centimeters', ...
        'PaperPosition',[0 0 figureWidthCm figureHeightCm], ...
        'PaperSize',[figureWidthCm figureHeightCm]);

    ax = axes(fig);
    hold(ax,'on');

    for method = 1:2

        plot(ax,k_values,Y(:,method), ...
            'LineStyle','-', ...
            'Marker',markers{method}, ...
            'MarkerIndices',1:numel(k_values), ...
            'Color',colors(method,:), ...
            'MarkerFaceColor',colors(method,:), ...
            'MarkerEdgeColor',colors(method,:), ...
            'LineWidth',curveLineWidth, ...
            'MarkerSize',markerSize, ...
            'DisplayName',methodNames{method});

    end

    xlabel(ax,'$k$', ...
        'Interpreter','latex', ...
        'FontSize',axisNameFontSize);

    ylabel(ax,yLabels{plotIndex}, ...
        'Interpreter','latex', ...
        'FontSize',axisNameFontSize);

    legend(ax,'Location','northwest', ...
        'FontSize',legendFontSize, ...
        'FontWeight','normal', ...
        'Box','on');

    set(ax, ...
        'FontName','Times New Roman', ...
        'FontSize',axisNumberFontSize, ...
        'FontWeight','normal', ...
        'LineWidth',1.2, ...
        'TickDir','in', ...
        'Box','on', ...
        'XScale','linear', ...
        'YScale','linear', ...
        'XLim',[0 max(k_values)], ...
        'XTick',20:20:100, ...
        'XMinorTick','off', ...
        'YMinorTick','off');

    % Major grid lines only.
    grid(ax,'on');
    ax.XMinorGrid = 'off';
    ax.YMinorGrid = 'off';
    ax.GridAlpha = 0.18;
    ax.GridLineStyle = '-';

    % Hide the interactive axes toolbar before export.
    if isprop(ax,'Toolbar') && ~isempty(ax.Toolbar)
        ax.Toolbar.Visible = 'off';
    end

    hold(ax,'off');
    drawnow;

    % Save high-resolution PNG.
    exportgraphics(fig, ...
        fullfile(outdir,[fileNames{plotIndex} '.png']), ...
        'Resolution',pngResolution, ...
        'BackgroundColor','white');

    % Save vector EPS.
    print(fig, ...
        fullfile(outdir,[fileNames{plotIndex} '.eps']), ...
        '-depsc','-painters');

end

fprintf('\nLeft outer-inverse PNG and EPS plots saved in:\n%s\n',outdir);