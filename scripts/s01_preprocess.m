%% Step 1 - Preprocessing
% Runs preprocessLesion on every image and saves:
%   results/preprocessed/<id>.mat      struct P (all stages + mask) for later steps
%   results/preprocessed/<id>_pre.png  final color image (P.smoothed)
%   results/image_quality.csv          per-image quality descriptors
%   results/figures/preprocessing_stages.png   step-by-step examples
%   results/figures/before_after.png   all images, raw vs preprocessed

setup;
T = loadDataset(cfg);
outDir = fullfile(cfg.resultsDir, 'preprocessed');
figDir = fullfile(cfg.resultsDir, 'figures');
if ~isfolder(outDir), mkdir(outDir); end
if ~isfolder(figDir), mkdir(figDir); end

n = height(T);
Q = cell(n, 1);
thumbsRaw = cell(n, 1); thumbsPre = cell(n, 1);
fprintf('Preprocessing %d images...\n', n);
for i = 1:n
    [I, M] = readSample(T, i);
    P = preprocessLesion(I, M, cfg);
    P.id = T.id(i); P.label = T.label(i);

    save(fullfile(outDir, T.id(i) + ".mat"), 'P', '-v7');
    imwrite(P.smoothed, fullfile(outDir, T.id(i) + "_pre.png"));

    Q{i} = struct2table(P.quality);
    thumbsRaw{i} = imresize(P.resized,  [150 200]);
    thumbsPre{i} = imresize(P.smoothed, [150 200]);
    fprintf('  %2d/%d %s  hair=%.1f%%  vignetting=%.1f\n', i, n, T.id(i), ...
        100*P.quality.hairFraction, P.quality.vignetting);
end

Q = [T(:, {'id', 'label'}) vertcat(Q{:})];
writetable(Q, fullfile(cfg.resultsDir, 'image_quality.csv'));

%% Figure: stages for a few example images (most hair, most vignetting, ...)
[~, iHair] = max(Q.hairFraction);
[~, iVig]  = max(Q.vignetting);
[~, iCast] = max(Q.colorCastAB);
ex = unique([iHair iVig iCast find(T.label == 'malignant', 1)], 'stable');

stageNames = {'resized', 'hairRemoved', 'illumCorrected', 'colorNorm', 'enhanced'};
rows = cell(numel(ex), 1);
for r = 1:numel(ex)
    S = load(fullfile(outDir, T.id(ex(r)) + ".mat")); P = S.P;
    tiles = cellfun(@(f) imresize(P.(f), [240 320]), stageNames, 'UniformOutput', false);
    tiles{end+1} = imresize(imoverlay(P.smoothed, bwperim(imdilate(P.mask, strel('disk', 2))), [0 1 0]), [240 320]); %#ok<SAGROW>
    tiles{end+1} = imresize(repmat(double(P.hairMask), 1, 1, 3), [240 320]); %#ok<SAGROW>
    rows{r} = imtile(tiles, 'GridSize', [1 numel(tiles)], 'BorderSize', 3, 'BackgroundColor', 'w');
end
stagesImg = imtile(rows, 'GridSize', [numel(rows) 1]);
imwrite(stagesImg, fullfile(figDir, 'preprocessing_stages.png'));

%% Figure: before/after for all images
ba = cellfun(@(a, b) [a; b], thumbsRaw, thumbsPre, 'UniformOutput', false);
imwrite(imtile(ba, 'GridSize', [5 13], 'BorderSize', 3, 'BackgroundColor', 'w'), ...
    fullfile(figDir, 'before_after.png'));

fprintf('\nDone. Columns in preprocessing_stages.png:\n  %s | mask overlay | hair mask\n', ...
    strjoin(stageNames, ' | '));
disp(groupsummary(Q, 'label', 'mean', {'brightness', 'sharpness', 'vignetting', 'hairFraction'}));

if usejava('desktop')
    figure('Name', 'Preprocessing stages'); imshow(stagesImg);
end
