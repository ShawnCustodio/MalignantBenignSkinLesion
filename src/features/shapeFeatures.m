function f = shapeFeatures(M)
%SHAPEFEATURES Size and shape descriptors of the lesion mask.
%   Images have different (unknown) magnifications, so absolute pixel area is
%   not comparable between images. We therefore use scale-free ratios, plus
%   areaFraction (lesion area / image area) as the only "size" feature.

    s = regionprops(M, 'Area', 'Perimeter', 'Eccentricity', 'Solidity', 'Extent', ...
        'MajorAxisLength', 'MinorAxisLength', 'ConvexHull');

    f.areaFraction = s.Area / numel(M);
    f.compactness  = s.Perimeter^2 / (4 * pi * s.Area);   % 1 = circle, larger = irregular
    f.solidity     = s.Solidity;                          % area / convex-hull area
    f.extent       = s.Extent;                            % area / bounding-box area
    f.eccentricity = s.Eccentricity;
    f.aspectRatio  = s.MajorAxisLength / max(s.MinorAxisLength, 1);

    hull = s.ConvexHull;
    hullPerim = sum(hypot(diff(hull(:,1)), diff(hull(:,2))));
    f.convexity = hullPerim / s.Perimeter;                % 1 = convex, <1 = indented border
end
