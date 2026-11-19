function out = renderNodeHeatmap(I, nodeMap, scores, M, clim)
%RENDERNODEHEATMAP Paint GCN region scores onto the lesion.
%   red  = region pushes toward MALIGNANT, blue = toward BENIGN,
%   white = neutral. clim = score magnitude mapped to full color
%   (default: max |score| in this lesion).
    if nargin < 5 || isempty(clim), clim = max(abs(scores)) + eps; end
    cmap = divergingMap(256);
    S = zeros(size(nodeMap));
    S(nodeMap > 0) = scores(nodeMap(nodeMap > 0));
    k = round((max(min(S / clim, 1), -1) + 1) / 2 * 255) + 1;
    heat = reshape(cmap(k, :), [size(S) 3]);
    alpha = 0.6 * (nodeMap > 0);
    out = I .* (1 - alpha) + heat .* alpha;
    % imoverlay returns uint8; convert back so it can be combined with
    % double images (mixing them silently turns the double image black)
    out = im2double(imoverlay(out, imdilate(bwperim(M), strel('disk', 1)), [0 0 0]));
end

function c = divergingMap(n)
    t = linspace(0, 1, n)';
    blue = [0.13 0.40 0.80]; white = [1 1 1]; red = [0.80 0.12 0.12];
    c = zeros(n, 3);
    lo = t <= 0.5; hi = ~lo;
    c(lo, :) = blue + (white - blue) .* (t(lo) / 0.5);
    c(hi, :) = white + (red - white) .* ((t(hi) - 0.5) / 0.5);
end
