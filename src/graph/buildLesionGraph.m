function G = buildLesionGraph(P, cfg)
%BUILDLESIONGRAPH Region-adjacency graph of a lesion for the GCN.
%   Nodes  = SLIC superpixels lying (mostly) inside the lesion mask.
%   Edges  = superpixels that share a boundary.
%   Node features (all relative to the lesion's own surrounding skin or to
%   the lesion itself, so they are robust to camera / lighting):
%      1-3  dL, da, db     mean Lab minus surrounding-skin Lab
%      4-6  sL, sa, sb     std of L, a, b inside the region (color variation)
%      7    grad           mean gradient magnitude (edges / structure)
%      8    texStd         std of CLAHE gray (local texture)
%      9    radial         distance from lesion center (0 center .. 1 edge)
%      10   isBorder       region touches the lesion border
%      11   relArea        region size relative to the average region
%      12   darkVsLesion   L minus mean lesion L (darker/lighter than rest)
%      13   nbrColorDiff   mean Lab Delta-E to neighboring regions
%                          (how different this region is from its neighbors)
%      14   blueGray       share of blue-gray pixels
%      15   black          share of very dark pixels (L* < 25)
%
%   G fields: X (n x 15), A (sparse adjacency), Ahat (normalized A+I),
%   nodeMap (label image, 0 = not a node, same size as P.mask), centroids.

    M = P.mask;
    [h, w] = size(M);
    lab = rgb2lab(P.smoothed);
    gray = rgb2gray(P.enhanced);
    gmag = imgradient(imgaussfilt(gray, 1));

    % Superpixels on the lesion bounding box, sized so ~nodesPerLesion fall inside
    [r, c] = find(M);
    pad = 10;
    r0 = max(min(r) - pad, 1); r1 = min(max(r) + pad, h);
    c0 = max(min(c) - pad, 1); c1 = min(max(c) + pad, w);
    boxArea = (r1 - r0 + 1) * (c1 - c0 + 1);
    nSp = max(round(cfg.graph.nodesPerLesion * boxArea / nnz(M)), 20);
    spCrop = superpixels(P.smoothed(r0:r1, c0:c1, :), nSp, 'Compactness', cfg.graph.compactness);
    sp = zeros(h, w); sp(r0:r1, c0:c1) = spCrop;

    % Keep superpixels that are > 50% lesion; node pixels = superpixel AND mask
    nAll = max(sp(:));
    inFrac = accumarray(sp(sp > 0), M(sp > 0), [nAll 1], @mean);
    keep = find(inFrac > 0.5);
    remap = zeros(nAll + 1, 1); remap(keep + 1) = 1:numel(keep);
    nodeMap = remap(sp + 1);
    nodeMap(~M) = 0;
    n = numel(keep);

    % Adjacency from horizontally / vertically neighboring pixels
    p1 = [reshape(nodeMap(:, 1:end-1), [], 1); reshape(nodeMap(1:end-1, :), [], 1)];
    p2 = [reshape(nodeMap(:, 2:end),   [], 1); reshape(nodeMap(2:end, :),   [], 1)];
    e = p1 > 0 & p2 > 0 & p1 ~= p2;
    A = sparse([p1(e); p2(e)], [p2(e); p1(e)], 1, n, n) > 0;

    % Reference values
    ring = imdilate(M, strel('disk', 40)) & ~imdilate(M, strel('disk', 10));
    Lc = lab(:,:,1); ac = lab(:,:,2); bc = lab(:,:,3);
    skin = [mean(Lc(ring)) mean(ac(ring)) mean(bc(ring))];
    lesionL = mean(Lc(M));
    [yy, xx] = ndgrid(1:h, 1:w);
    cy = mean(r); cx = mean(c);
    dCenter = hypot(yy - cy, xx - cx);
    dCenter = dCenter / max(dCenter(M));
    border = imdilate(bwperim(M), strel('disk', 2));
    C = hypot(ac, bc); hue = atan2d(bc, ac);
    blueGray = (bc < 3 | hue > 100) & C < 15 & Lc < skin(1) - 5;
    black = Lc < 25;

    idx = nodeMap(nodeMap > 0);
    mean_ = @(img) accumarray(idx, img(nodeMap > 0), [n 1], @mean);
    std_  = @(img) accumarray(idx, img(nodeMap > 0), [n 1], @std);
    npix  = accumarray(idx, 1, [n 1]);

    meanLab = [mean_(Lc) mean_(ac) mean_(bc)];
    X = zeros(n, 15);
    X(:, 1:3)  = meanLab - skin;
    X(:, 4:6)  = [std_(Lc) std_(ac) std_(bc)];
    X(:, 7)    = mean_(gmag);
    X(:, 8)    = std_(gray);
    X(:, 9)    = mean_(dCenter);
    X(:, 10)   = accumarray(idx, border(nodeMap > 0), [n 1], @max);
    X(:, 11)   = npix / mean(npix);
    X(:, 12)   = meanLab(:,1) - lesionL;
    [ii, jj] = find(A);
    dE = vecnorm(meanLab(ii,:) - meanLab(jj,:), 2, 2);
    X(:, 13)   = accumarray(ii, dE, [n 1], @mean, 0);
    X(:, 14)   = mean_(double(blueGray));
    X(:, 15)   = mean_(double(black));

    % Symmetric normalized adjacency with self-loops (Kipf & Welling 2017)
    At = double(A) + speye(n);
    d = full(sum(At, 2));
    Dm = spdiags(1 ./ sqrt(d), 0, n, n);
    G.Ahat = Dm * At * Dm;

    G.X = X;
    G.A = A;
    G.nodeMap = nodeMap;
    G.centroids = [accumarray(idx, xx(nodeMap > 0), [n 1], @mean), ...
                   accumarray(idx, yy(nodeMap > 0), [n 1], @mean)];
    G.featureNames = {'dL', 'da', 'db', 'sL', 'sa', 'sb', 'grad', 'texStd', 'radial', ...
        'isBorder', 'relArea', 'darkVsLesion', 'nbrColorDiff', 'blueGray', 'black'};
end
