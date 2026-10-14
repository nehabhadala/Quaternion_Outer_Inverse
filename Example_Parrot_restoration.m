% Parrot image restoration using five pseudoinverse implementations.
%
% All methods solve the same noise-free problem Bq = Aq*Xq.

%
% Tolerance policy:
%   QTFM pinv: explicit tolA.
%   QTFM SVD and Complex SVD: shared tolW for W = Aq'*Aq*Aq'.
%   QTFM FR and Complex FR: shared tolPivot.
%   FR reduced matrices: individual default cutoffs, tolReduced = [].
%
% Reported measures:
%   RR    : relative RGB reconstruction error BEFORE clipping.
%   PSNR  : pooled RGB PSNR AFTER clipping to [0,1], peak value 1.
%   SSIM  : average channel-wise SSIM AFTER clipping, dynamic range 1.
%   Time  : elapsed time for pseudoinverse computation and G*Bq.
%
% Timing: one execution per method, fixed order, no warm-up.
% Tolerance preparation, metrics and image saving are outside timing.
%
% Output folder: Parrot_Image_Restoration

clc;
clear;
close all;

%% 1. File locations

scriptDir = fileparts(mfilename('fullpath'));

inputFile = fullfile(scriptDir,'Input_images','parrot512.png');
outdir = fullfile(scriptDir,'Parrot_Image_Restoration');

if ~exist(outdir,'dir')
    mkdir(outdir);
end

%% 2. Read the original image

Original = im2double(imread(inputFile));
[height,width,channels] = size(Original);

if channels ~= 3
    error('The input must be an RGB image.');
end

normOriginal = norm(Original(:));

if normOriginal == 0
    error('RR is undefined for an identically zero original image.');
end

imwrite(Original,fullfile(outdir,'Parrot_original.png'));

%% 3. Construct the blurring matrix

sigma = 3;
r = 3;
s = 3;
p = 32;
q = 16;

if height ~= p*q
    error('Image height must equal p*q = %d.',p*q);
end

% Keep the original kernel definitions unchanged.
B1 = generateB1(p,sigma,r);
B2 = generateB2(q,s);

H = kron(B1,B2);

Ai = H;
Aj = -0.5*H;
Ak = -0.5*H;

% Original pure quaternion image.
Xq = quaternion(zeros(height,width), ...
    Original(:,:,1),Original(:,:,2),Original(:,:,3));

% Quaternion blurring matrix and its complex components.
Aq = quaternion(zeros(size(H)),Ai,Aj,Ak);

Ac1 = complex(zeros(size(H)),Ai);
Ac2 = complex(Aj,Ak);

% Noise-free observation. Retain all four quaternion components.
Bq = Aq*Xq;

% RGB preview only; the solver uses the full, unclipped Bq.
BlurredRaw = cat(3,Bq.x,Bq.y,Bq.z);
BlurredDisplay = max(0,min(1,BlurredRaw));

imwrite(BlurredDisplay,fullfile(outdir,'Parrot_blur.png'));

%% 4. Tolerances: computed outside timing

[rankA,tolA] = quaternion_rank(Aq,[]);

W = Aq'*Aq*Aq';
[rankW,tolW] = quaternion_rank(W,[]);

magA = hypot(abs(Ac1),abs(Ac2));

tolPivot = max(size(Aq))*eps(class(magA))*max(magA(:));
tolReduced = [];

%% 5. Storage

methodNames = {
    'QTFM_pinv'
    'QTFM_SVD'
    'Complex_SVD'
    'QTFM_FR'
    'Complex_FR'
};

imageNames = {
    'Parrot_pinv.png'
    'Parrot_QSVD.png'
    'Parrot_CSVD.png'
    'Parrot_QFR.png'
    'Parrot_CFR.png'
};

num_methods = numel(methodNames);
num_trials = 1;

Elapsed = NaN(num_methods,1);
PSNR_RGB = NaN(num_methods,1);
SSIM_RGB = NaN(num_methods,1);
RR = NaN(num_methods,1);

TrialTimes = NaN(num_trials,num_methods);

RestoredImages = cell(num_methods,1);
RawRGB = cell(num_methods,1);

Status = repmat({'not run'},num_methods,1);
Failure = repmat({''},num_methods,1);

%% 6. Compute the five restorations

for method = 1:num_methods

    fprintf('\nRunning %s...\n',methodNames{method});

    try
        for trial = 1:num_trials

            timer = tic;

            switch method

                case 1
                    G = pinv(Aq,tolA);

                case 2
                    G = QTFM_pinv_compute(Aq,tolW);

                case 3
                    G = Complex_pinv_compute(Ac1,Ac2,tolW);

                case 4
                    G = QTFM_pinv_compute_FR( ...
                        Aq,tolPivot,tolReduced);

                case 5
                    G = Complex_pinv_compute_FR( ...
                        Ac1,Ac2,tolPivot,tolReduced);

            end

            Xrec = G*Bq;

            TrialTimes(trial,method) = toc(timer);

        end

        Elapsed(method) = mean(TrialTimes(:,method));

        % Extract the reconstructed RGB channels.
        raw = cat(3,Xrec.x,Xrec.y,Xrec.z);
        scalarPart = Xrec.s;

        if any(~isfinite(raw(:))) || any(~isfinite(scalarPart(:)))
            error('The reconstruction contains nonfinite values.');
        end

        % RR: relative RGB reconstruction error BEFORE clipping.
        rawDifference = raw-Original;
        RR(method) = norm(rawDifference(:))/normOriginal;

        % Clip RGB values for PSNR, SSIM and image display.
        restored = max(0,min(1,raw));

        % PSNR: pooled RGB MSE, peak intensity 1.
        difference = restored-Original;
        mseRGB = mean(difference(:).^2);

        if mseRGB == 0
            PSNR_RGB(method) = Inf;
        else
            PSNR_RGB(method) = 10*log10(1/mseRGB);
        end

        % SSIM: mean of the three channel-wise values.
        channelSSIM = zeros(1,3);

        for channel = 1:3
            channelSSIM(channel) = ssim( ...
                restored(:,:,channel), ...
                Original(:,:,channel), ...
                'DynamicRange',1);
        end

        SSIM_RGB(method) = mean(channelSSIM);

        RawRGB{method} = raw;
        RestoredImages{method} = restored;

        imwrite(restored,fullfile(outdir,imageNames{method}));

        Status{method} = 'completed';

    catch ME

        Status{method} = 'failed';
        Failure{method} = ME.message;

        % Do not report partially calculated results as successful.
        Elapsed(method) = NaN;
        PSNR_RGB(method) = NaN;
        SSIM_RGB(method) = NaN;
        RR(method) = NaN;

        warning('ParrotRestoration:MethodFailure', ...
            '%s failed: %s',methodNames{method},ME.message);

    end
end

%% 7. Display and save the four reported measures

Results = table(methodNames,Elapsed,PSNR_RGB,SSIM_RGB,RR, ...
    'VariableNames', ...
    {'Method','Elapsed_s','PSNR_dB','SSIM','RR'});

disp(Results);

writetable(Results,fullfile(outdir,'Parrot_results.csv'));

% Preserve numerical results and the full observation for later use.
ObservedScalar = Bq.s;

save(fullfile(outdir,'Parrot_results.mat'), ...
    'Results','TrialTimes','num_trials','methodNames', ...
    'Original','BlurredRaw','ObservedScalar', ...
    'RestoredImages','RawRGB','Status','Failure', ...
    'tolA','tolW','tolPivot','tolReduced', ...
    'rankA','rankW','p','q','sigma','r','s');

%% 8. Save an aligned text table

fid = fopen(fullfile(outdir,'Parrot_results.txt'),'w');

if fid < 0
    error('Cannot open the results text file.');
end

fprintf(fid,'PARROT IMAGE RESTORATION: FIVE METHODS\n');
fprintf(fid,'Image size: %d-by-%d\n',height,width);
fprintf(fid,'Noise-free observation: Bq = Aq*Xq.\n');
fprintf(fid,'p=%d, q=%d, sigma=%g, r=%d, s=%d\n', ...
    p,q,sigma,r,s);

fprintf(fid,'\nTolerance policy:\n');
fprintf(fid,'QTFM pinv tolA: %.12e\n',tolA);
fprintf(fid,'Shared SVD tolW: %.12e\n',tolW);
fprintf(fid,'Shared FR tolPivot: %.12e\n',tolPivot);
fprintf(fid,'FR reduced cutoffs: individual defaults.\n');

fprintf(fid,'\nTiming includes pseudoinverse computation and G*Bq.\n');
fprintf(fid,'Repetitions: %d; fixed order; no warm-up.\n',num_trials);

fprintf(fid,'RR: relative RGB error before clipping.\n');
fprintf(fid,'PSNR: pooled RGB MSE after clipping; peak intensity 1.\n');
fprintf(fid,'SSIM: mean channel-wise SSIM after clipping.\n\n');

fprintf(fid,'%-16s %16s %16s %16s %16s\n', ...
    'Method','Elapsed(s)','PSNR(dB)','SSIM','RR');

fprintf(fid,'%s\n',repmat('-',1,84));

for method = 1:num_methods
    fprintf(fid,'%-16s %16.6e %16.6f %16.6f %16.6e\n', ...
        methodNames{method}, ...
        Elapsed(method), ...
        PSNR_RGB(method), ...
        SSIM_RGB(method), ...
        RR(method));
end

for method = 1:num_methods
    if strcmp(Status{method},'failed')
        fprintf(fid,'\nFAILED: %s\n%s\n', ...
            methodNames{method},Failure{method});
    end
end

fprintf(fid,'\nMATLAB: %s\nPlatform: %s\n',version,computer);

fclose(fid);

%% 9. Display the images together

figure('Color','w');

tiledlayout(2,4,'TileSpacing','compact','Padding','compact');

nexttile;
imshow(Original);
title('Original');

nexttile;
imshow(BlurredDisplay);
title('Observation RGB preview');

for method = 1:num_methods

    nexttile;

    if strcmp(Status{method},'completed')
        imshow(RestoredImages{method});
    else
        axis off;
        text(0.1,0.5,'Method failed');
    end

    title(methodNames{method},'Interpreter','none');

end

fprintf('\nResults and images saved in:\n%s\n',outdir);

%% Local functions: original blur kernels

function B1 = generateB1(p,sigma,r)

B1 = zeros(p,p);

for i = 1:p
    for j = 1:p
        if abs(i-j) <= r
            B1(i,j) = exp(-(i-j)^2/(2*sigma^2)) ...
                /(sigma*sqrt(2*pi));
        end
    end
end

end

function B2 = generateB2(q,s)

B2 = zeros(q,q);

for i = 1:q
    for j = 1:q
        if abs(i-j) <= s
            B2(i,j) = 1/(2*s-1);
        end
    end
end

end