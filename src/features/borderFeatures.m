function f = borderFeatures(M, L)
%BORDERFEATURES Border irregularity and border sharpness ("B" of ABCD).
%   L is the L* (lightness) channel of the preprocessed image.
%
%   radialCV       std/mean of centroid-to-border distance (irregular outline)
%   fractalDim     box-counting dimension of the border (1 = smooth curve)
%   borderGradMean mean lightness gradient on the border band (abrupt edge)
%   borderGradCV   variation of that gradient across 8 angular sectors -
%                  melanomas often have an abrupt cut-off on one side and a
%                  gradual fade on another

    B = bwboundaries(M, 'noholes');
    [~, k] = max(cellfun(@numel, B));
    b = B{k};                                 % [row col] border points
    c = mean(b, 1);
    d = hypot(b(:,1) - c(1), b(:,2) - c(2));
    f.radialCV = std(d) / mean(d);

    f.fractalDim = boxCountDim(bwperim(M));

    gmag = imgradient(imgaussfilt(L, 1));
    band = imdilate(bwperim(M), strel('disk', 3));
    f.borderGradMean = mean(gmag(band));

    [rr, cc] = find(band);
    ang = atan2(rr - c(1), cc - c(2));
    sector = discretize(ang, linspace(-pi, pi, 9));
    g = gmag(band);
    secMean = accumarray(sector, g, [8 1], @mean, NaN);
    f.borderGradCV = std(secMean, 'omitnan') / mean(secMean, 'omitnan');
end

function D = boxCountDim(P)
    [r, c] = find(P);
    sizes = 2.^(1:7);
    counts = zeros(size(sizes));
    for i = 1:numel(sizes)
        counts(i) = size(unique([floor(r / sizes(i)) floor(c / sizes(i))], 'rows'), 1);
    end
    p = polyfit(log(1 ./ sizes), log(counts), 1);
    D = p(1);
end
