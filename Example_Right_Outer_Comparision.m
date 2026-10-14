% RIGHT OUTER-INVERSE COMPARISON: COMPLEX AND QTFM
%
% Purpose:
%   Compare the computation of X = S*(T*A*S)^(1)*T using
%   Complex_X_compute and QTFM_X_compute.
%   S and T denote S1 and T1 in the paper.
%   Both implementations use the free blocks K=L=M=0.
%
% Input matrices:
%   For k=5:5:100, set m=3k, n=2k and p=q=k.
%   Generate quaternion matrices:
%       A: m-by-n, S: n-by-p, T: q-by-m.
%   Their four real components are generated using rand.
%   Both implementations receive the same matrices.
%   The actual random seed in this script is rng(42,'twister').
%   One matrix triple is generated per k and reused for all trials.
%
% Tolerance policy:
%   Form W=T*A*S before performing the rank checks.
%   For each M in {A,S,T,W}, quaternion_rank uses the scale-aware
%   singular-value cutoff
%       tolM = max(size(M))*eps*sigma_max(M).
%   Here eps is the machine spacing at unity for the working precision.
%
%   Compute tolW once outside timing and pass it unchanged to both
%   inverse implementations. Both should retain singular values
%   strictly greater than tolW.
%   tolA, tolS and tolT are used only for the external rank diagnostics.
%
% Rank conditions:
%   Check the numerical condition rank(W)=rank(S)=rank(T).
%   The script stops if this condition fails.
%   No additional requirement that these ranks equal p is imposed.
%   Save all four numerical ranks and their associated cutoffs.
%
% Timing:
%   Run each inverse function once before the timing loop.
%   Perform 50 repetitions on the same matrix triple.
%   Measure each complete inverse-function call using tic/toc.
%   Report the arithmetic mean of the elapsed execution times.
%

%
%   Matrix generation, external rank checks, residual evaluation,
%   plotting and file output are excluded from timing.
%   The repetitions measure timing variability for one input triple;
%   they are not 50 independent random-matrix experiments.
%
% Accuracy:
%   Using the final computed outputs, evaluate:
%       Outer = ||X*A*X-X||_F / ||X||_F
%       Range = ||X*A*S-S||_F / ||S||_F
%       Null  = ||T*A*X-T||_F / ||T||_F
%   These are relative equation residuals.
%   Denominators are bounded below by realmin.
%   Frobenius norms are evaluated using QTFM's norm implementation.
%
% Output:
%   Columns are ordered as Complex, QTFM.
%   Save elapsed times, residuals, ranks and tolerances in
%   TXT, CSV and MAT files.
%   Save four separate PNG plots with linear axes:
%   time, outer-inverse residual, range residual and null residual.
%   Output folder: Example_Right_Outer_Comparision_output.
%


% Requirements:
%   QTFM, quaternion_rank, Complex_X_compute, QTFM_X_compute,
%   and their supporting functions.


clc;
clear;
close all;
rng(42,'twister');

k_values = 5:5:100;
num_trials = 50;

outdir = fullfile(fileparts(mfilename('fullpath')), ...
    'Example_Right_Outer_Comparision_output');

if ~exist(outdir,'dir')
    mkdir(outdir);
end

N = numel(k_values);

Time = zeros(N,2);       % Column 1: Complex; column 2: QTFM.
OuterError = zeros(N,2);
RangeError = zeros(N,2);
NullError = zeros(N,2);
Ranks = zeros(N,4);

% Columns: tolA, tolS, tolT, tolW.
Tolerances = zeros(N,4);

for idx = 1:N

    k = k_values(idx);
    m = 3*k;
    n = 2*k;
    p = k;
    q = k;

    %% 1. Generate the same data for both methods
    A0 = rand(m,n);
    A1 = rand(m,n);
    A2 = rand(m,n);
    A3 = rand(m,n);

    A11 = complex(A0,A1);
    A12 = complex(A2,A3);
    A = quaternion(A0,A1,A2,A3);

    S0 = rand(n,p);
    S1 = rand(n,p);
    S2 = rand(n,p);
    S3 = rand(n,p);

    S11 = complex(S0,S1);
    S12 = complex(S2,S3);
    S = quaternion(S0,S1,S2,S3);

    T0 = rand(q,m);
    T1 = rand(q,m);
    T2 = rand(q,m);
    T3 = rand(q,m);

    T11 = complex(T0,T1);
    T12 = complex(T2,T3);
    T = quaternion(T0,T1,T2,T3);

    %% 2. Rank checks and shared tolerance, outside timing
    W = T*A*S;

    [ra,tolA] = quaternion_rank(A,[]);
    [rs,tolS] = quaternion_rank(S,[]);
    [rt,tolT] = quaternion_rank(T,[]);
    [rw,tolW] = quaternion_rank(W,[]);

    if rw ~= rs || rw ~= rt
        error(['Rank conditions fail at k=%d: rank(W)=%d, ' ...
               'rank(S)=%d, rank(T)=%d.'], ...
               k,rw,rs,rt);
    end

    Ranks(idx,:) = [ra,rs,rt,rw];
    Tolerances(idx,:) = [tolA,tolS,tolT,tolW];

    %% 3. Original timing procedure
    total_time_complex = 0;
    total_time_qtfm = 0;

    % Warm-up using the same explicit cutoff.
    Complex_X_compute(A11,A12,S11,S12,T11,T12,tolW);
    QTFM_X_compute(A,S,T,tolW);

    for trial = 1:num_trials

        % Complex method.
        tic;
        Xc = Complex_X_compute(A11,A12,S11,S12,T11,T12,tolW);
        total_time_complex = total_time_complex + toc;

        clear functions;

        % QTFM method.
        tic;
        Xq = QTFM_X_compute(A,S,T,tolW);
        total_time_qtfm = total_time_qtfm + toc;

        clear functions;
    end

    avg_time_complex = total_time_complex/num_trials;
    avg_time_qtfm = total_time_qtfm/num_trials;

    Time(idx,:) = [avg_time_complex,avg_time_qtfm];

    %% 4. Relative residuals, outside timing
    % XAX=X: outer inverse; XAS=S: range; TAX=T: null space.
    nc = max(norm(Xc,'fro'),realmin);
    nq = max(norm(Xq,'fro'),realmin);
    ns = max(norm(S,'fro'),realmin);
    nt = max(norm(T,'fro'),realmin);

    OuterError(idx,:) = [ ...
        norm(Xc*A*Xc-Xc,'fro')/nc, ...
        norm(Xq*A*Xq-Xq,'fro')/nq];

    RangeError(idx,:) = [ ...
        norm(Xc*A*S-S,'fro')/ns, ...
        norm(Xq*A*S-S,'fro')/ns];

    NullError(idx,:) = [ ...
        norm(T*A*Xc-T,'fro')/nt, ...
        norm(T*A*Xq-T,'fro')/nt];

    fprintf('k=%3d completed: Complex %.6f s, QTFM %.6f s\n', ...
        k,avg_time_complex,avg_time_qtfm);
end

%% 5. Save an aligned text table
fid = fopen(fullfile(outdir, ...
    'Example_Right_Outer_Comparision_output.txt'),'w');

if fid < 0
    error('Cannot open results file.');
end

fprintf(fid,'RIGHT OUTER INVERSE: COMPLEX AND QTFM\n');
fprintf(fid,'Seed: 42 (twister) | Trials: %d\n',num_trials);
fprintf(fid,'Dimensions: m=3k, n=2k, p=q=k\n');

fprintf(fid,'Tolerance policy:\n');
fprintf(fid,'Each rank check uses tolM=max(size(M))*eps*sigma_max(M).\n');
fprintf(fid,'A single tolW is computed from W=T*A*S outside timing.\n');
fprintf(fid,'Both inverse methods receive exactly the same tolW.\n');
fprintf(fid,'Both inverse methods retain singular values using s>tolW.\n');

fprintf(fid,['Time: mean tic/toc elapsed seconds; ' ...
             'original clear-functions procedure.\n']);

fprintf(fid,'Outer_error = ||XAX-X||_F / ||X||_F\n');
fprintf(fid,'Range_error = ||XAS-S||_F / ||S||_F\n');
fprintf(fid,'Null_error  = ||TAX-T||_F / ||T||_F\n\n');

line = repmat('-',1,88);

fprintf(fid,'%s\n',line);
fprintf(fid,'%5s  %-9s  %14s  %14s  %14s  %14s\n', ...
    'k','Method','Time(s)','Outer_error','Range_error','Null_error');
fprintf(fid,'%s\n',line);

for idx = 1:N

    fprintf(fid,'%5d  %-9s  %14.6f  %14.6e  %14.6e  %14.6e\n', ...
        k_values(idx),'Complex',Time(idx,1), ...
        OuterError(idx,1),RangeError(idx,1),NullError(idx,1));

    fprintf(fid,'%5d  %-9s  %14.6f  %14.6e  %14.6e  %14.6e\n', ...
        k_values(idx),'QTFM',Time(idx,2), ...
        OuterError(idx,2),RangeError(idx,2),NullError(idx,2));

end

fprintf(fid,'%s\n\nRANK CHECKS\n',line);
fprintf(fid,'%5s  %10s  %10s  %10s  %10s\n', ...
    'k','rank(A)','rank(S)','rank(T)','rank(TAS)');
fprintf(fid,'%s\n',repmat('-',1,53));

for idx = 1:N
    fprintf(fid,'%5d  %10d  %10d  %10d  %10d\n', ...
        k_values(idx),Ranks(idx,:));
end

fprintf(fid,'\nTOLERANCES\n');
fprintf(fid,'%5s  %16s  %16s  %16s  %16s\n', ...
    'k','tolA','tolS','tolT','tolW');
fprintf(fid,'%s\n',repmat('-',1,77));

for idx = 1:N
    fprintf(fid,'%5d  %16.8e  %16.8e  %16.8e  %16.8e\n', ...
        k_values(idx),Tolerances(idx,:));
end

fprintf(fid,'\nMATLAB: %s\nPlatform: %s\n',version,computer);
fclose(fid);

% Retain numerical data for later plotting.
Results = array2table( ...
    [k_values(:),Time,OuterError,RangeError,NullError,Ranks,Tolerances], ...
    'VariableNames', { ...
    'k','Time_Complex_s','Time_QTFM_s', ...
    'Outer_Complex','Outer_QTFM', ...
    'Range_Complex','Range_QTFM', ...
    'Null_Complex','Null_QTFM', ...
    'RankA','RankS','RankT','RankW', ...
    'TolA','TolS','TolT','TolW'});

writetable(Results,fullfile(outdir,'Example_results.csv'));

save(fullfile(outdir,'Example_results.mat'), ...
    'Results','Time','OuterError','RangeError','NullError', ...
    'Ranks','Tolerances','k_values','num_trials');

%% 6. Four separate plots: right outer inverse

methodNames = {'Complex','QTFM'};

plotData = {Time,OuterError,RangeError,NullError};

yLabels = { ...
    'Mean elapsed time (s)', ...
    '$\epsilon_{\mathrm{out}}$', ...
    '$\epsilon_{\mathrm{range}}$', ...
    '$\epsilon_{\mathrm{null}}$'};

fileNames = { ...
    'Right_CPU_time', ...
    'Right_outer_error', ...
    'Right_range_error', ...
    'Right_null_error'};

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

fprintf('\nRight outer-inverse PNG and EPS plots saved in:\n%s\n',outdir);