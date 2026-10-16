% QUATERNION PSEUDOINVERSE COMPARISON: FIVE METHODS
%
% Purpose:
%   Compare five implementations on the same quaternion matrix:
%       1. Complex SVD construction: Algorithm 2
%       2. QTFM SVD construction: Algorithm 1
%       3. Complex FR construction: Algorithm 4
%       4. QTFM FR construction: Algorithm 3
%       5. QTFM toolbox pinv
%
% Input matrices:
%   For k=20:20:100, generate A of size m-by-n,
%   with m=3k and n=2k.
%   Generate the four real quaternion components using rand.
%   Fix rng(111,'twister') once before matrix generation.
%   All five methods receive the same matrix.
%   One matrix is generated per k and reused for all timing trials.
%
% Tolerance policy:
%   1. The two SVD constructions receive the SAME explicit cutoff:
%        tolW = max(m,n)*eps*sigma_max(W), W=A'*A*A'.
%      Compute tolW using quaternion_rank(W,[]), outside timing.
%      This cutoff applies to W, not directly to A.
%
%   2. Both FR methods receive the SAME explicit pivot cutoff:
%        tolPivot = max(m,n)*eps*max_ij |A_ij|.
%      This controls pivot acceptance in the factorization of A'.
%      A and A' have the same maximum entry magnitude and
%      maximum dimension.
%      tolPivot is not a singular-value cutoff.
%
%   3. tolReduced=[] lets each FR method use its own default:
%        tolReduced = r*eps*sigma_max(R),
%      where A' approximately equals S1*T1, R=T1*A*S1,
%      and r is the accepted quaternion pivot count.
%      The formula is shared, but numerical values may differ.
%      This checks invertibility of R; it does not truncate R.
%
%   4. QTFM pinv receives the explicit cutoff:
%        tolA = max(m,n)*eps*sigma_max(A).
%      Compute tolA using quaternion_rank(A,[]), outside timing.
%
%   eps denotes the machine spacing at unity for the working precision.
%   Cutoffs are shared within corresponding method pairs.
%   They apply to different matrices and operations, so they are not
%   identical across all five methods.
%
% Timing:
%   Make one preliminary call to every method.
%   Perform 50 repeated evaluations of each method on the same A.

%
%   Report mean elapsed time.
%   Matrix generation, external tolerance preparation, residual
%   evaluation, plotting and file output are excluded from timing.
%   The repetitions are not independent random-matrix experiments.
%
% Accuracy:
%   Evaluate four RELATIVE Moore-Penrose residuals:
%       MP1 = ||A*X*A-A||_F / ||A||_F
%       MP2 = ||X*A*X-X||_F / ||X||_F
%       MP3 = ||(A*X)'-A*X||_F / ||A*X||_F
%       MP4 = ||(X*A)'-X*A||_F / ||X*A||_F
%
%   Use the outputs from the final timed calls.
%   Quaternion Frobenius norms are computed as norm(abs(Q),'fro').
%   Denominators are floored at realmin to prevent division by zero.
%   This safeguard does not affect the nonzero norms in this example.
%
% Output:
%   Save five aligned tables and tolerance values in a TXT file.
%   Save corresponding CSV files and a MAT file.
%   Save five separate plots in PNG, EPS and FIG formats.
%   All plots use linear axes and major grid lines only.
%   Err1, Err2, Err3 and Err4  store relative residuals.
%
% Output folder:
%   Example_Pseudoinverse_Comparison_output
%
% Requires:
%   QTFM, quaternion_rank, and the four custom pseudoinverse functions.
%   An error raised by any method stops the script.

clc;
clear;
close all;

seed = 111;
rng(seed,'twister');

%% 1. Settings and output folder

k_values = 20:20:100;
num_trials = 50;
nk = numel(k_values);

scriptDir = fileparts(mfilename('fullpath'));
outdir = fullfile(scriptDir, ...
    'Example_Pseudoinverse_Comparison_output');

if ~exist(outdir,'dir')
    mkdir(outdir);
end

% Column order is used consistently throughout the example.
methodNames = { ...
    'Complex', ...
    'QTFM', ...
    'Complex_FR', ...
    'QTFM_FR', ...
    'QTFM_pinv'};

num_methods = numel(methodNames);

Time = zeros(nk,num_methods);
Err1 = zeros(nk,num_methods);
Err2 = zeros(nk,num_methods);
Err3 = zeros(nk,num_methods);
Err4 = zeros(nk,num_methods);

% Columns: tolA, tolW, tolPivot.
Tolerances = zeros(nk,3);

%% 2. Computation

for idx = 1:nk

    k = k_values(idx);
    m = 3*k;
    n = 2*k;

    % Generate the same quaternion matrix for all methods.
    A0 = rand(m,n);
    A1 = rand(m,n);
    A2 = rand(m,n);
    A3 = rand(m,n);

    A11 = complex(A0,A1);
    A12 = complex(A2,A3);

    A = quaternion(A0,A1,A2,A3);

    %% 3. Tolerances: outside timing

    % Explicit cutoff for QTFM pinv(A).
    [~,tolA] = quaternion_rank(A,[]);

    % Shared cutoff for the two SVD constructions.
    W = A'*A*A';
    [~,tolW] = quaternion_rank(W,[]);

    % Shared pivot cutoff for the two FR methods.
    magA = hypot(abs(A11),abs(A12));

    tolPivot = max(m,n)*eps(class(magA))*max(magA(:));

    % Each FR method uses its own default reduced-matrix cutoff.
    tolReduced = [];

    Tolerances(idx,:) = [tolA,tolW,tolPivot];

    %% 4. Preliminary calls: outside timing

    Complex_pinv_compute(A11,A12,tolW);
    QTFM_pinv_compute(A,tolW);

    Complex_pinv_compute_FR(A11,A12,tolPivot,tolReduced);
    QTFM_pinv_compute_FR(A,tolPivot,tolReduced);

    pinv(A,tolA);

    %% 5. Original timing procedure

    T = zeros(1,num_methods);

    for trial = 1:num_trials

        tic;
        X1 = Complex_pinv_compute(A11,A12,tolW);
        T(1) = T(1) + toc;
        clear functions;

        tic;
        X2 = QTFM_pinv_compute(A,tolW);
        T(2) = T(2) + toc;
        clear functions;

        tic;
        X3 = Complex_pinv_compute_FR( ...
            A11,A12,tolPivot,tolReduced);
        T(3) = T(3) + toc;
        clear functions;

        tic;
        X4 = QTFM_pinv_compute_FR(A,tolPivot,tolReduced);
        T(4) = T(4) + toc;
        clear functions;

        tic;
        X5 = pinv(A,tolA);
        T(5) = T(5) + toc;
        clear functions;

    end

    Time(idx,:) = T/num_trials;

    %% 6. Relative Moore-Penrose residuals: outside timing

    Xvalues = {X1,X2,X3,X4,X5};

    % Common denominator for MP1.
    tiny = realmin(class(magA));
    normA = max(norm(abs(A),'fro'),tiny);

    for method = 1:num_methods

        X = Xvalues{method};

        AX = A*X;
        XA = X*A;

        % Method-dependent normalization factors.
        normX = max(norm(abs(X),'fro'),tiny);
        normAX = max(norm(abs(AX),'fro'),tiny);
        normXA = max(norm(abs(XA),'fro'),tiny);

        Err1(idx,method) = norm(abs(AX*A-A),'fro')/normA;
        Err2(idx,method) = norm(abs(XA*X-X),'fro')/normX;
        Err3(idx,method) = norm(abs(AX'-AX),'fro')/normAX;
        Err4(idx,method) = norm(abs(XA'-XA),'fro')/normXA;

    end

    %% 7. Display results for this size

    fprintf('\nCompleted k=%d: A is %d-by-%d.\n',k,m,n);

    fprintf('%-12s %14s %14s %14s %14s %14s\n', ...
        'Method','Time(s)','Relative MP1','Relative MP2', ...
        'Relative MP3','Relative MP4');

    for method = 1:num_methods
        fprintf('%-12s %14.6e %14.6e %14.6e %14.6e %14.6e\n', ...
            methodNames{method},Time(idx,method), ...
            Err1(idx,method),Err2(idx,method), ...
            Err3(idx,method),Err4(idx,method));
    end

end

%% 8. Save five aligned tables in one text file

output_file = fullfile(outdir,'Pseudoinverse_results.txt');
fileID = fopen(output_file,'w');

if fileID < 0
    error('Cannot open the results text file.');
end

fprintf(fileID,'QUATERNION PSEUDOINVERSE: FIVE METHODS\n');
fprintf(fileID,'Seed: %d (twister); repetitions: %d\n', ...
    seed,num_trials);
fprintf(fileID,'Dimensions: m=3k, n=2k.\n');
fprintf(fileID,'One random matrix per size, reused across repetitions.\n\n');

fprintf(fileID,'TOLERANCE POLICY\n');
fprintf(fileID,'tolA = max(m,n)*eps*sigma_max(A).\n');
fprintf(fileID,'tolW = max(m,n)*eps*sigma_max(W), W=A''*A*A''.\n');
fprintf(fileID,'Complex and QTFM SVD constructions share tolW.\n');
fprintf(fileID,'tolPivot = max(m,n)*eps*max_ij |A_ij|.\n');
fprintf(fileID,'Both FR methods share tolPivot.\n');
fprintf(fileID,'Each FR method uses its default reduced cutoff:\n');
fprintf(fileID,'r*eps*sigma_max(R), evaluated from its own R.\n');
fprintf(fileID,'QTFM pinv receives tolA explicitly.\n\n');

fprintf(fileID,'TIMING AND ERRORS\n');
fprintf(fileID,'Time is mean elapsed time measured using tic/toc.\n');
fprintf(fileID,'Fixed method order; clear functions after each timed call.\n');
fprintf(fileID,'These are not consistently warmed execution timings.\n');
fprintf(fileID,'External tolerance preparation and residuals are not timed.\n');
fprintf(fileID,'Errors are relative Frobenius residuals.\n');
fprintf(fileID,'MP1 denominator: ||A||_F.\n');
fprintf(fileID,'MP2 denominator: ||X||_F.\n');
fprintf(fileID,'MP3 denominator: ||A*X||_F.\n');
fprintf(fileID,'MP4 denominator: ||X*A||_F.\n');
fprintf(fileID,'Denominators are floored at realmin.\n\n');

tableTitles = {
    'TABLE 1: MEAN ELAPSED TIME (seconds)'
    'TABLE 2: RELATIVE MP1 ||A*X*A-A||_F / ||A||_F'
    'TABLE 3: RELATIVE MP2 ||X*A*X-X||_F / ||X||_F'
    'TABLE 4: RELATIVE MP3 ||(A*X)''-A*X||_F / ||A*X||_F'
    'TABLE 5: RELATIVE MP4 ||(X*A)''-X*A||_F / ||X*A||_F'
};

tableData = {Time,Err1,Err2,Err3,Err4};

csvNames = { ...
    'Pseudoinverse_time.csv', ...
    'Pseudoinverse_MP1.csv', ...
    'Pseudoinverse_MP2.csv', ...
    'Pseudoinverse_MP3.csv', ...
    'Pseudoinverse_MP4.csv'};

sep = repmat('-',1,90);

for tableIndex = 1:numel(tableData)

    values = tableData{tableIndex};

    fprintf(fileID,'%s\n',tableTitles{tableIndex});

    fprintf(fileID,'%5s %16s %16s %16s %16s %16s\n', ...
        'k',methodNames{:});

    fprintf(fileID,'%s\n',sep);

    for idx = 1:nk
        fprintf(fileID,'%5d %16.6e %16.6e %16.6e %16.6e %16.6e\n', ...
            k_values(idx),values(idx,:));
    end

    fprintf(fileID,'\n');

    % Save the same table in CSV format.
    ResultTable = array2table([k_values(:),values], ...
        'VariableNames',[{'k'},methodNames]);

    writetable(ResultTable,fullfile(outdir,csvNames{tableIndex}));

end

%% 9. Save tolerance values and numerical data

fprintf(fileID,'EXPLICIT TOLERANCES\n');

fprintf(fileID,'%5s %18s %18s %18s\n', ...
    'k','tolA','tolW','tolPivot');

fprintf(fileID,'%s\n',repmat('-',1,62));

for idx = 1:nk
    fprintf(fileID,'%5d %18.10e %18.10e %18.10e\n', ...
        k_values(idx),Tolerances(idx,:));
end

matlabVersion = version;
platform = computer;

fprintf(fileID,'\nMATLAB: %s\nPlatform: %s\n', ...
    matlabVersion,platform);

fclose(fileID);

ToleranceTable = array2table( ...
    [k_values(:),Tolerances], ...
    'VariableNames',{'k','tolA','tolW','tolPivot'});

writetable(ToleranceTable, ...
    fullfile(outdir,'Pseudoinverse_tolerances.csv'));

% Err1--Err4 contain relative residuals.
save(fullfile(outdir,'Pseudoinverse_results.mat'), ...
    'seed','k_values','num_trials','methodNames', ...
    'Time','Err1','Err2','Err3','Err4','Tolerances', ...
    'matlabVersion','platform');

%% 10. Five separate plots: pseudoinverse comparison

% Appearance settings: consistent with the outer-inverse examples.
figureWidthCm = 12;
figureHeightCm = 9;

axisNumberFontSize = 14;
axisNameFontSize = 20;
legendFontSize = 14;

curveLineWidth = 2;
markerSize = 7;
pngResolution = 600;

% Column order:
% Complex SVD, QTFM SVD, Complex FR, QTFM FR, QTFM pinv.
colors = [ ...
    0.000 0.447 0.741; ...  % Algorithm 2: blue
    0.850 0.325 0.098; ...  % Algorithm 1: orange
    0.466 0.674 0.188; ...  % Algorithm 4: green
    0.494 0.184 0.556; ...  % Algorithm 3: purple
    0.000 0.000 0.000];     % QTFM pinv: black

markers = {'d','o','s','^','v'};
lineStyles = {'-','-','--','--',':'};

legendNames = { ...
    'Algorithm 2', ...
    'Algorithm 1', ...
    'Algorithm 4', ...
    'Algorithm 3', ...
    'QTFM pinv'};

plotData = {Time,Err1,Err2,Err3,Err4};

yLabels = { ...
    'Mean elapsed time (s)', ...
    '$\|AXA-A\|_F/\|A\|_F$', ...
    '$\|XAX-X\|_F/\|X\|_F$', ...
    '$\|(AX)^*-AX\|_F/\|AX\|_F$', ...
    '$\|(XA)^*-XA\|_F/\|XA\|_F$'};

fileNames = { ...
    'Pseudoinverse_time', ...
    'Pseudoinverse_MP1', ...
    'Pseudoinverse_MP2', ...
    'Pseudoinverse_MP3', ...
    'Pseudoinverse_MP4'};

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

    plotOrder = [2 1 4 3 5];

    for method = plotOrder

        plot(ax,k_values,Y(:,method), ...
            'LineStyle',lineStyles{method}, ...
            'Marker',markers{method}, ...
            'MarkerIndices',1:numel(k_values), ...
            'Color',colors(method,:), ...
            'MarkerFaceColor',colors(method,:), ...
            'MarkerEdgeColor',colors(method,:), ...
            'LineWidth',curveLineWidth, ...
            'MarkerSize',markerSize, ...
            'DisplayName',legendNames{method});

    end

    xlabel(ax,'$k$', ...
        'Interpreter','latex', ...
        'FontSize',axisNameFontSize);

    ylabel(ax,yLabels{plotIndex}, ...
        'Interpreter','latex', ...
        'FontSize',axisNameFontSize);

    legend(ax,'Location','northwest', ...
        'Interpreter','none', ...
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
        'XLim',[20 max(k_values)], ...
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

    % Save editable MATLAB figure.
    savefig(fig, ...
        fullfile(outdir,[fileNames{plotIndex} '.fig']));

end

fprintf('\nResults and PNG, EPS and FIG plots saved in:\n%s\n', ...
    outdir);