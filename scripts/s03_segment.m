%% Step 3 - Fully automatic pipeline (no hand-drawn masks)
% Re-runs preprocessing with M = [] so the lesion is found automatically,
% scores the automatic masks against the hand-drawn ones, and extracts the
% same 63 features from the automatic masks. Saves:
%   results/preprocessed_auto/<id>.mat     struct P (P.mask = automatic mask)
%   results/segmentation_scores.csv        Dice / Jaccard / ... per image
%   results/features_auto.csv              features from automatic masks
%   results/figures/segmentation_auto_vs_manual.png
%       green = hand-drawn outline, magenta = automatic outline

setup;
T = loadDataset(cfg);
outDir = fullfile(cfg.resultsDir, 'preprocessed_auto');
figDir = fullfile(cfg.resultsDir, 'figures');
if ~isfolder(outDir), mkdir(outDir); end

n = height(T);
scores = cell(n, 1); feats = cell(n, 1); tiles = cell(n, 1);
fprintf('Automatic segmentation of %d images...\n', n);
for i = 1:n
    [I, Mgt] = readSample(T, i);
    P = preprocessLesion(I, [], cfg);
    P.id = T.id(i); P.label = T.label(i);
    P.maskManual = cleanMask(imresize(Mgt, size(P.mask), 'nearest'));
    save(fullfile(outDir, T.id(i) + ".mat"), 'P', '-v7');

    s = segmentationScores(P.mask, P.maskManual);
    scores{i} = struct2table(s);
    feats{i} = struct2table(extractFeatures(P));

    ov = imoverlay(P.smoothed, imdilate(bwperim(P.maskManual), strel('disk', 2)), [0 1 0]);
    ov = imoverlay(ov, imdilate(bwperim(P.mask), strel('disk', 2)), [1 0 1]);
    tiles{i} = imresize(ov, [150 200]);
    fprintf('  %2d/%d %s  Dice = %.3f\n', i, n, T.id(i), s.dice);
end

S = [T(:, {'id', 'label'}) vertcat(scores{:})];
writetable(S, fullfile(cfg.resultsDir, 'segmentation_scores.csv'));
F = [T(:, {'id', 'label'}) vertcat(feats{:})];
writetable(F, fullfile(cfg.resultsDir, 'features_auto.csv'));

[~, order] = sort(S.dice);                       % worst first, easier to inspect
imwrite(imtile(tiles(order), 'GridSize', [5 13], 'BorderSize', 3, 'BackgroundColor', 'w'), ...
    fullfile(figDir, 'segmentation_auto_vs_manual.png'));

fprintf('\nDice: mean %.3f, median %.3f, min %.3f  (>0.8 is good for dermoscopy)\n', ...
    mean(S.dice), median(S.dice), min(S.dice));
disp(groupsummary(S, 'label', 'mean', {'dice', 'jaccard', 'sensitivity', 'specificity', 'areaRatio'}));
fprintf('Worst 5:\n'); disp(S(order(1:5), {'id', 'label', 'dice', 'areaRatio'}));
