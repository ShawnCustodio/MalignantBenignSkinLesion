function f = textureFeatures(G, M)
%TEXTUREFEATURES GLCM (Haralick) and LBP texture inside the lesion.
%   G is the grayscale CLAHE-enhanced image (from P.enhanced).
%
%   GLCM: pixels outside the lesion are set to NaN (graycomatrix ignores
%   them), 32 gray levels, 4 directions averaged, distances 1/3/5 px.
%   LBP:  rotation-invariant uniform LBP (8 neighbors, radius 1), 10-bin
%   normalized histogram. Implemented here because extractLBPFeatures
%   needs the Computer Vision Toolbox.

    [r, c] = find(M);
    Gc = G(min(r):max(r), min(c):max(c));
    Mc = M(min(r):max(r), min(c):max(c));
    Gn = Gc; Gn(~Mc) = NaN;

    props = {'Contrast', 'Correlation', 'Energy', 'Homogeneity'};
    warnState = warning('off', 'images:graycomatrix:scaledImageContainsNan');
    cleanup = onCleanup(@() warning(warnState));
    for d = [1 3 5]
        offs = [0 d; -d d; -d 0; -d -d];
        glcm = graycomatrix(Gn, 'NumLevels', 32, 'GrayLimits', [0 1], ...
            'Offset', offs, 'Symmetric', true);
        st = graycoprops(glcm, props);
        for p = props
            f.(sprintf('glcm%s_d%d', p{1}, d)) = mean(st.(p{1}), 'omitnan');
        end
        Pn = sum(glcm, 3); Pn = Pn / sum(Pn(:));
        f.(sprintf('glcmEntropy_d%d', d)) = -sum(Pn(Pn > 0) .* log2(Pn(Pn > 0)));
    end

    h = lbpRIU2(Gc, imerode(Mc, strel('disk', 1)));
    for k = 1:numel(h)
        f.(sprintf('lbp%02d', k - 1)) = h(k);
    end
end

function h = lbpRIU2(G, M)
    % 8 neighbors in circular order
    offs = [0 1; -1 1; -1 0; -1 -1; 0 -1; 1 -1; 1 0; 1 1];
    bits = false([size(G) 8]);
    for k = 1:8
        bits(:,:,k) = circshift(G, -offs(k,:)) >= G;
    end
    transitions = sum(bits ~= circshift(bits, 1, 3), 3);
    code = sum(bits, 3);
    code(transitions > 2) = 9;              % non-uniform patterns
    h = accumarray(code(M) + 1, 1, [10 1])';
    h = h / sum(h);
end
