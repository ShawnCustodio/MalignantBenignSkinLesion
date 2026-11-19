function f = asymmetryFeatures(M, lab)
%ASYMMETRYFEATURES Shape and color asymmetry about the lesion's principal axes.
%   The lesion is centered on its centroid and rotated so its major axis is
%   horizontal (PCA of pixel coordinates). It is then mirrored about each
%   axis:
%     shape asymmetry = |lesion XOR mirrored lesion| / |lesion|  (0 = symmetric)
%     color asymmetry = Delta-E between the mean Lab color of the two halves
%   This is the "A" of the dermoscopic ABCD rule.

    [y, x] = find(M);
    x = x - mean(x); y = y - mean(y);
    [V, D] = eig(cov(x, y));
    [~, order] = sort(diag(D), 'descend');
    V = V(:, order);
    uv = [x y] * V;            % u = along major axis, v = along minor axis
    u = uv(:,1); v = uv(:,2);

    % Rasterize the rotated mask on a grid centered at (0,0) so flips are exact
    R = ceil(max(abs([u; v]))) + 2;
    G = false(2*R + 1);
    G(sub2ind(size(G), round(v) + R + 1, round(u) + R + 1)) = true;
    G = imfill(imclose(G, strel('disk', 1)), 'holes');   % fill rounding gaps
    A = nnz(G);

    f.asymShapeMajor = nnz(xor(G, flipud(G))) / A;   % mirrored across major axis
    f.asymShapeMinor = nnz(xor(G, fliplr(G))) / A;   % mirrored across minor axis
    f.asymShapeMean  = (f.asymShapeMajor + f.asymShapeMinor) / 2;

    % Color asymmetry: compare halves on either side of each axis
    idx = find(M);
    pix = reshape(lab, [], 3); pix = pix(idx, :);
    dE = @(a, b) norm(mean(pix(a,:), 1) - mean(pix(b,:), 1));
    f.asymColorMajor = dE(v > 0, v <= 0);
    f.asymColorMinor = dE(u > 0, u <= 0);
    f.asymColorMax   = max(f.asymColorMajor, f.asymColorMinor);
end
