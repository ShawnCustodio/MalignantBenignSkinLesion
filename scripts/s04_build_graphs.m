%% Step 4 - Build one superpixel graph per lesion (for the GCN)
% Builds graphs for both mask sources:
%   manual: hand-drawn masks   (results/preprocessed/*.mat, from step 1)
%   auto  : automatic masks    (results/preprocessed_auto/*.mat, from step 3)
% Saves results/graphs_manual.mat and results/graphs_auto.mat
% (variables: graphs (struct array), ids, labels) and an example figure
% results/figures/graph_examples.png (superpixel borders + graph edges).

setup;
T = loadDataset(cfg);
figDir = fullfile(cfg.resultsDir, 'figures');
sources = {'manual', 'preprocessed'; 'auto', 'preprocessed_auto'};

for s = 1:size(sources, 1)
    src = sources{s, 1};
    inDir = fullfile(cfg.resultsDir, sources{s, 2});
    if ~isfolder(inDir)
        warning('%s not found - run the earlier step first. Skipping "%s".', inDir, src);
        continue
    end
    fprintf('Building %s-mask graphs...\n', src);
    graphs = cell(height(T), 1);
    for i = 1:height(T)
        S = load(fullfile(inDir, T.id(i) + ".mat"));
        graphs{i} = buildLesionGraph(S.P, cfg);
    end
    graphs = vertcat(graphs{:});
    ids = T.id; labels = T.label;
    save(fullfile(cfg.resultsDir, "graphs_" + src + ".mat"), 'graphs', 'ids', 'labels', '-v7');
    nn = arrayfun(@(g) size(g.X, 1), graphs);
    ne = arrayfun(@(g) nnz(g.A) / 2, graphs);
    fprintf('  nodes per lesion: median %d (min %d, max %d); edges: median %d\n', ...
        round(median(nn)), min(nn), max(nn), round(median(ne)));
end

%% Example figure: 2 benign + 2 malignant (manual masks)
S = load(fullfile(cfg.resultsDir, 'graphs_manual.mat'));
ex = [find(T.label == 'benign', 2); find(T.label == 'malignant', 2)];
fig = figure('Visible', 'off', 'Position', [50 50 1600 450]);
if isprop(fig, 'Theme'), fig.Theme = 'light'; end
tiledlayout(1, numel(ex), 'TileSpacing', 'compact', 'Padding', 'compact');
for k = 1:numel(ex)
    i = ex(k); G = S.graphs(i);
    P = load(fullfile(cfg.resultsDir, 'preprocessed', T.id(i) + ".mat")).P;
    nexttile;
    imshow(imoverlay(P.smoothed, boundarymask(G.nodeMap) & G.nodeMap > 0, [1 1 1]));
    hold on;
    [a, b] = find(triu(G.A));
    xy = G.centroids;
    plot([xy(a,1) xy(b,1)]', [xy(a,2) xy(b,2)]', '-', 'Color', [0 0.8 1], 'LineWidth', 0.8);
    plot(xy(:,1), xy(:,2), '.', 'Color', [1 0.9 0], 'MarkerSize', 10);
    [r, c] = find(P.mask);
    axis([min(c)-15 max(c)+15 min(r)-15 max(r)+15]);
    title(sprintf('%s  (%s, %d nodes)', T.id(i), char(T.label(i)), size(G.X, 1)), 'Interpreter', 'none');
end
exportgraphics(fig, fullfile(figDir, 'graph_examples.png'), 'Resolution', 110);
close(fig);
