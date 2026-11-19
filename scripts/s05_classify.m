%% Step 5 - Random Forest vs GCN, with manual and automatic masks
% Needs steps 2-4. For each mask source (manual / auto):
%   * Random Forest on the 63 handcrafted features
%   * GCN on the superpixel graphs
% Both use the SAME stratified 5-fold splits, repeated cfg.cvRepeats times.
% Every prediction / explanation is made on images the model never saw
% during training ("out-of-fold").
%
% Saves (per source <src> = manual | auto):
%   results/predictions_<src>.csv         per-image probabilities + quality
%   results/rf_importance_<src>.csv       RF permutation importance
%   results/gcn_region_features_<src>.csv what red (malignant) regions look like
%   results/heatmaps_<src>/<id>.png       GCN region heatmap for every lesion
%   results/figures/gcn_heatmaps_<src>.png  overview grid
%   results/figures/rf_importance_<src>.png
%   results/figures/failure_cases_<src>.png  lesions both models get wrong
% and, across sources:
%   results/classification_summary.csv  all models, mean +- sd over repeats
%   results/significance_tests.csv      exact McNemar: GCN vs RF, GCN vs GCN-without-edges
%   results/quality_vs_errors.csv       image quality of correct vs wrong images
%   results/figures/model_comparison.png
%
% GCN_noEdges is an ABLATION: the same network with the graph edges removed.
% If GCN > GCN_noEdges, the spatial relations between regions help.

setup;
T = loadDataset(cfg);
figDir = fullfile(cfg.resultsDir, 'figures');
sources = {'manual', 'features.csv',      'graphs_manual.mat', 'preprocessed';
           'auto',   'features_auto.csv', 'graphs_auto.mat',   'preprocessed_auto'};
Qual = readtable(fullfile(cfg.resultsDir, 'image_quality.csv'), 'TextType', 'string');
summary = table();
tests = cell(0, 5);
qualityTests = cell(0, 6);

for s = 1:size(sources, 1)
    src = sources{s, 1};
    F = readtable(fullfile(cfg.resultsDir, sources{s, 2}), 'TextType', 'string');
    G = load(fullfile(cfg.resultsDir, sources{s, 3}));
    assert(isequal(F.id, T.id) && isequal(G.ids, T.id), 'Row order mismatch - rerun steps 2-4.');
    featNames = F.Properties.VariableNames(3:end);
    Xf = F{:, featNames};
    y = T.label == cfg.positiveClass;
    n = height(T); nR = cfg.cvRepeats;

    pRF = zeros(n, nR); pGCN = zeros(n, nR); pGCN0 = zeros(n, nR);
    nodeScores = cell(n, 1);                 % from repeat 1 (out-of-fold)
    imp = [];
    fprintf('\n=== %s masks: %d repeats x %d folds ===\n', upper(src), nR, cfg.kFolds);
    for r = 1:nR
        cfgR = cfg; cfgR.seed = cfg.seed + r - 1;
        cv = makeFolds(T.label, cfgR);
        for f = 1:cv.NumTestSets
            tr = training(cv, f); te = test(cv, f);
            rng(cfgR.seed * 100 + f);

            % --- Random Forest ---
            tb = TreeBagger(cfg.rf.numTrees, Xf(tr, :), double(y(tr)), 'Method', 'classification', ...
                'MinLeafSize', cfg.rf.minLeaf, 'OOBPredictorImportance', 'on');
            [~, sc] = predict(tb, Xf(te, :));
            pRF(te, r) = sc(:, strcmp(tb.ClassNames, '1'));
            imp(:, end+1) = tb.OOBPermutedPredictorDeltaError'; %#ok<SAGROW>

            % --- GCN ---
            model = gcnTrain(G.graphs(tr), y(tr), cfg.gcn);
            [pGCN(te, r), ns] = gcnPredict(model, G.graphs(te));
            if r == 1, nodeScores(te) = ns; end

            % --- Ablation: same GCN without edges (does spatial info help?) ---
            rng(cfgR.seed * 100 + f);
            opts0 = cfg.gcn; opts0.useEdges = false;
            model0 = gcnTrain(G.graphs(tr), y(tr), opts0);
            pGCN0(te, r) = gcnPredict(model0, G.graphs(te));
        end
        fprintf('  repeat %d/%d done\n', r, nR);
    end

    % --- Metrics (per repeat, then mean +- sd) ---
    models = {'RandomForest', pRF; 'GCN', pGCN; 'GCN_noEdges', pGCN0};
    for m = 1:size(models, 1)
        P_ = models{m, 2};
        M_ = zeros(nR, 6);
        for r = 1:nR
            pred = categorical(repmat("benign", n, 1), cfg.classNames);
            pred(P_(:, r) >= 0.5) = cfg.positiveClass;
            mt = classificationMetrics(T.label, pred, cfg.positiveClass);
            [~, ~, ~, auc] = perfcurve(y, P_(:, r), true);
            M_(r, :) = [mt.accuracy mt.sensitivity mt.specificity mt.precision mt.f1 auc];
        end
        row = table(string(src), string(models{m, 1}), 'VariableNames', {'masks', 'model'});
        mn = {'accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'auc'};
        for k = 1:6
            row.(mn{k}) = mean(M_(:, k));
            row.([mn{k} '_sd']) = std(M_(:, k));
        end
        summary = [summary; row]; %#ok<AGROW>
    end

    % --- Per-image predictions + image quality (failure analysis) ---
    Pr = T(:, {'id', 'label'});
    Pr.pMal_RF = mean(pRF, 2); Pr.pMal_GCN = mean(pGCN, 2); Pr.pMal_GCNnoEdges = mean(pGCN0, 2);
    Pr.correct_RF  = (Pr.pMal_RF  >= 0.5) == y;
    Pr.correct_GCN = (Pr.pMal_GCN >= 0.5) == y;
    Pr.correct_GCNnoEdges = (Pr.pMal_GCNnoEdges >= 0.5) == y;

    % --- Paired significance (exact McNemar on per-image predictions) ---
    pairs = {'GCN', 'RF'; 'GCN', 'GCNnoEdges'};
    for k = 1:size(pairs, 1)
        a = Pr.("correct_" + pairs{k, 1}); b = Pr.("correct_" + pairs{k, 2});
        pm = mcnemarExact(a, b);
        fprintf('  McNemar %s vs %s (%s masks): %s right & %s wrong on %d, reverse on %d -> p = %.3f\n', ...
            pairs{k, 1}, pairs{k, 2}, src, pairs{k, 1}, pairs{k, 2}, sum(a & ~b), sum(~a & b), pm);
        tests(end+1, :) = {string(src), string(pairs{k,1}) + " vs " + pairs{k,2}, ...
            sum(a & ~b), sum(~a & b), pm}; %#ok<SAGROW>
    end
    Pr = join(Pr, Qual(:, setdiff(Qual.Properties.VariableNames, {'label'})), 'Keys', 'id');
    writetable(Pr, fullfile(cfg.resultsDir, "predictions_" + src + ".csv"));
    % raw out-of-fold outputs, so figures can be redrawn without retraining
    ids = T.id;
    save(fullfile(cfg.resultsDir, "cv_outputs_" + src + ".mat"), ...
        'pRF', 'pGCN', 'pGCN0', 'nodeScores', 'ids', '-v7');

    % --- RF importance ---
    I = table(featNames', mean(imp, 2), std(imp, 0, 2), 'VariableNames', {'feature', 'importance', 'sd'});
    I = sortrows(I, 'importance', 'descend');
    writetable(I, fullfile(cfg.resultsDir, "rf_importance_" + src + ".csv"));
    fig = figure('Visible', 'off', 'Position', [100 100 700 500]);
    if isprop(fig, 'Theme'), fig.Theme = 'light'; end
    top = I(1:15, :);
    barh(flipud(top.importance)); yticks(1:15); yticklabels(flipud(top.feature));
    set(gca, 'TickLabelInterpreter', 'none');
    xlabel('Permutation importance (increase in OOB error)');
    title(sprintf('Random Forest - top 15 features (%s masks)', src));
    exportgraphics(fig, fullfile(figDir, "rf_importance_" + src + ".png"), 'Resolution', 120); close(fig);

    % --- What do "malignant" regions look like? (node score vs node feature) ---
    Xn = vertcat(G.graphs.X); sn = vertcat(nodeScores{:});
    fn = G.graphs(1).featureNames;
    rho = corr(Xn, sn, 'Type', 'Spearman');
    R = table(fn', rho, 'VariableNames', {'regionFeature', 'spearmanWithMalignantScore'});
    R = sortrows(R, 'spearmanWithMalignantScore', 'descend', 'ComparisonMethod', 'abs');
    writetable(R, fullfile(cfg.resultsDir, "gcn_region_features_" + src + ".csv"));

    % --- Heatmaps (out-of-fold, repeat 1) ---
    hmDir = fullfile(cfg.resultsDir, "heatmaps_" + src);
    if ~isfolder(hmDir), mkdir(hmDir); end
    imgs = cell(n, 1);
    for i = 1:n
        P = load(fullfile(cfg.resultsDir, sources{s, 4}, T.id(i) + ".mat")).P;
        H = renderNodeHeatmap(P.smoothed, G.graphs(i).nodeMap, nodeScores{i}, P.mask);
        orig = im2double(imoverlay(P.smoothed, imdilate(bwperim(P.mask), strel('disk', 1)), [0 0 0]));
        imgs{i} = H;
        imwrite([orig H], fullfile(hmDir, T.id(i) + ".png"));
    end
    % overview: 4 confident malignant, 4 confident benign, up to 4 GCN errors
    [~, oM] = sort(Pr.pMal_GCN .* y, 'descend');
    [~, oB] = sort(Pr.pMal_GCN + y, 'ascend');
    wrong = find(~Pr.correct_GCN);
    pick = [oM(1:4); oB(1:4); wrong(1:min(4, end))];
    fig = figure('Visible', 'off', 'Position', [50 50 1600 1250]);
    if isprop(fig, 'Theme'), fig.Theme = 'light'; end
    tl = tiledlayout(3, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
    for k = 1:numel(pick)
        i = pick(k); nexttile;
        P = load(fullfile(cfg.resultsDir, sources{s, 4}, T.id(i) + ".mat")).P;
        [rr, cc] = find(P.mask);
        crop = imgs{i}(max(min(rr)-15,1):min(max(rr)+15,end), max(min(cc)-15,1):min(max(cc)+15,end), :);
        imshow(crop);
        tag = "";
        if ~Pr.correct_GCN(i), tag = "  WRONG"; end
        title(sprintf('%s | true: %s | P(mal)=%.2f%s', T.id(i), char(T.label(i)), Pr.pMal_GCN(i), tag), ...
            'Interpreter', 'none', 'FontSize', 9);
    end
    title(tl, sprintf(['GCN region heatmaps (%s masks) - red: pushes toward malignant, ' ...
        'blue: toward benign. Rows: confident malignant | confident benign | errors'], src));
    exportgraphics(fig, fullfile(figDir, "gcn_heatmaps_" + src + ".png"), 'Resolution', 110); close(fig);

    % --- Failure cases: wrong for BOTH RF and GCN (original | heatmap) ---
    both = find(~Pr.correct_RF & ~Pr.correct_GCN);
    if ~isempty(both)
        both = both(1:min(8, end));
        nc = 4; nrow = ceil(numel(both) / nc);
        fig = figure('Visible', 'off', 'Position', [50 50 1600 420 * nrow]);
        if isprop(fig, 'Theme'), fig.Theme = 'light'; end
        tl = tiledlayout(nrow, nc, 'TileSpacing', 'compact', 'Padding', 'loose');
        for k = 1:numel(both)
            i = both(k); nexttile;
            P = load(fullfile(cfg.resultsDir, sources{s, 4}, T.id(i) + ".mat")).P;
            [rr, cc] = find(P.mask);
            rs = max(min(rr)-15,1):min(max(rr)+15,size(P.mask,1));
            cs = max(min(cc)-15,1):min(max(cc)+15,size(P.mask,2));
            imshow([P.smoothed(rs, cs, :) imgs{i}(rs, cs, :)]);
            title({sprintf('%s  true: %s', T.id(i), char(T.label(i))), ...
                sprintf('P(mal) RF %.2f  GCN %.2f | sharp %.2f  bright %.0f', ...
                Pr.pMal_RF(i), Pr.pMal_GCN(i), Pr.sharpness(i), Pr.brightness(i))}, ...
                'Interpreter', 'none', 'FontSize', 8);
        end
        title(tl, sprintf('Misclassified by BOTH models (%s masks): original | GCN heatmap', src));
        exportgraphics(fig, fullfile(figDir, "failure_cases_" + src + ".png"), 'Resolution', 110); close(fig);
    end

    % --- Does image quality relate to errors? ---
    qv = {'sharpness', 'brightness', 'vignetting', 'hairFraction', 'colorCastAB'};
    for m = ["RF" "GCN"]
        c = Pr.("correct_" + m);
        for k = 1:numel(qv)
            x = Pr.(qv{k});
            qualityTests(end+1, :) = {string(src), m, string(qv{k}), median(x(c)), ...
                median(x(~c)), ranksum(x(c), x(~c))}; %#ok<SAGROW>
        end
    end
end
Q2 = cell2table(qualityTests, 'VariableNames', {'masks', 'model', 'qualityMeasure', ...
    'medianCorrect', 'medianWrong', 'pWilcoxon'});
writetable(Q2, fullfile(cfg.resultsDir, 'quality_vs_errors.csv'));

writetable(summary, fullfile(cfg.resultsDir, 'classification_summary.csv'));
Tt = cell2table(tests, 'VariableNames', {'masks', 'comparison', 'firstRightSecondWrong', 'firstWrongSecondRight', 'pMcNemar'});
writetable(Tt, fullfile(cfg.resultsDir, 'significance_tests.csv'));

%% Model comparison figure (mean +- sd over CV repeats)
mn = {'accuracy', 'sensitivity', 'specificity', 'f1', 'auc'};
fig = figure('Visible', 'off', 'Position', [100 100 1300 450]);
if isprop(fig, 'Theme'), fig.Theme = 'light'; end
tiledlayout(1, 2, 'TileSpacing', 'compact');
for s = 1:size(sources, 1)
    nexttile;
    rows = summary(summary.masks == sources{s, 1}, :);
    vals = rows{:, mn}'; sds = rows{:, strcat(mn, '_sd')}';
    b = bar(vals, 'grouped'); hold on;
    for k = 1:numel(b)
        errorbar(b(k).XEndPoints, vals(:, k), sds(:, k), 'k.', 'LineWidth', 1, 'HandleVisibility', 'off');
    end
    yline(0.5, ':', 'chance', 'HandleVisibility', 'off');
    xticklabels(mn); ylim([0 1]); ylabel('score');
    legend(strrep(rows.model, '_', ' '), 'Location', 'southoutside', 'Orientation', 'horizontal');
    title(sprintf('%s masks', sources{s, 1}));
end
exportgraphics(fig, fullfile(figDir, 'model_comparison.png'), 'Resolution', 120); close(fig);
fprintf('\n===== Results (mean over %d repeats of %d-fold CV) =====\n', cfg.cvRepeats, cfg.kFolds);
disp(summary(:, {'masks', 'model', 'accuracy', 'accuracy_sd', 'sensitivity', 'specificity', 'precision', 'f1', 'auc'}));
disp(Tt);
