%% Step 2 - Handcrafted feature extraction
% Needs step 1 (results/preprocessed/*.mat). Saves:
%   results/features.csv          one row per image: id, label, ~70 features
%   results/feature_ranking.csv   per-feature AUC + Wilcoxon p-value (exploratory)
%   results/figures/top_features_boxplots.png
%
% NOTE: the ranking uses ALL images, so it is for description/the report
% only. Any feature selection used by a classifier must happen inside the
% cross-validation training folds (step 3), otherwise results are inflated.

setup;
T = loadDataset(cfg);
preDir = fullfile(cfg.resultsDir, 'preprocessed');
figDir = fullfile(cfg.resultsDir, 'figures');
if ~isfolder(figDir), mkdir(figDir); end

n = height(T);
rows = cell(n, 1);
fprintf('Extracting features from %d images...\n', n);
for i = 1:n
    S = load(fullfile(preDir, T.id(i) + ".mat"));
    rows{i} = struct2table(extractFeatures(S.P));
end
F = [T(:, {'id', 'label'}) vertcat(rows{:})];
writetable(F, fullfile(cfg.resultsDir, 'features.csv'));
featNames = F.Properties.VariableNames(3:end);
fprintf('%d features x %d images -> results/features.csv\n', numel(featNames), n);

bad = featNames(any(~isfinite(F{:, featNames}), 1));
if ~isempty(bad), warning('Non-finite values in: %s', strjoin(bad, ', ')); end

%% Univariate ranking (exploratory)
isMal = F.label == cfg.positiveClass;
nf = numel(featNames);
auc = zeros(nf, 1); pval = zeros(nf, 1);
medBen = zeros(nf, 1); medMal = zeros(nf, 1);
for k = 1:nf
    x = F.(featNames{k});
    [~, ~, ~, auc(k)] = perfcurve(isMal, x, true);
    pval(k) = ranksum(x(isMal), x(~isMal));
    medBen(k) = median(x(~isMal)); medMal(k) = median(x(isMal));
end
R = table(featNames', auc, max(auc, 1 - auc), pval, medBen, medMal, ...
    'VariableNames', {'feature', 'AUC', 'separation', 'pWilcoxon', 'medianBenign', 'medianMalignant'});
R.higherIn = repmat("malignant", nf, 1); R.higherIn(R.AUC < 0.5) = "benign";
R = sortrows(R, 'separation', 'descend');
writetable(R, fullfile(cfg.resultsDir, 'feature_ranking.csv'));

fprintf('\nTop 15 features (AUC: 0.5 = useless, 1 = perfect; separation = max(AUC, 1-AUC))\n');
disp(R(1:15, {'feature', 'separation', 'pWilcoxon', 'higherIn'}));
fprintf('%d of %d features have p < 0.05 (uncorrected; expect ~%d by chance)\n', ...
    sum(R.pWilcoxon < 0.05), nf, round(0.05 * nf));

%% Boxplots of the 9 most separating features
fig = figure('Visible', 'off', 'Position', [100 100 1200 900]);
if isprop(fig, 'Theme'), fig.Theme = 'light'; end   % white background for the report
tiledlayout(3, 3, 'TileSpacing', 'compact');
for k = 1:9
    nexttile;
    boxchart(double(F.label), F.(R.feature{k}));
    xticks([1 2]); xticklabels(categories(F.label));
    title(sprintf('%s (AUC %.2f)', R.feature{k}, R.separation(k)), 'Interpreter', 'none');
end
exportgraphics(fig, fullfile(figDir, 'top_features_boxplots.png'), 'Resolution', 120);
close(fig);
if usejava('desktop')
    figure; imshow(imread(fullfile(figDir, 'top_features_boxplots.png')));
end
