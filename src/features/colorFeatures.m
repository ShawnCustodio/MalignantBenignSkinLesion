function f = colorFeatures(I, M, lab)
%COLORFEATURES Color, discoloration and pigmentation variation ("C" of ABCD).
%   I is the preprocessed RGB image (P.smoothed - NOT the CLAHE image).
%
%   - mean / std / skewness of L*, a*, b* inside the lesion
%   - lesion-vs-surrounding-skin contrast (relative, so robust to camera)
%   - dermoscopic color count: each lesion pixel is assigned by Lab rules to
%     one of six colors (white, red, light brown, dark brown, blue-gray,
%     black); a color "is present" if it covers >= 5% of the lesion.
%     Melanomas typically show 3+ colors.
%   - pigmentation variation: lightness entropy and center-vs-periphery
%     difference

    names = {'L', 'A', 'B'};
    pix = reshape(lab, [], 3);
    les = pix(M(:), :);
    for k = 1:3
        f.(['mean' names{k}]) = mean(les(:,k));
        f.(['std'  names{k}]) = std(les(:,k));
        f.(['skew' names{k}]) = skewness(les(:,k));
    end

    hsv = rgb2hsv(I); S = hsv(:,:,2);
    f.meanSat = mean(S(M));

    % Surrounding skin ring (10-40 px outside the border)
    ring = imdilate(M, strel('disk', 40)) & ~imdilate(M, strel('disk', 10));
    skin = mean(pix(ring(:), :), 1);
    f.contrastL = skin(1) - f.meanL;                   % how much darker than skin
    f.deltaESkin = norm(mean(les, 1) - skin);

    % Dermoscopic colors by rules in Lab (lightness L, chroma C, hue angle h).
    % Fixed reference colors from the literature did not fit after color
    % constancy (median Delta-E ~27 to the nearest reference), so the classes
    % are defined relative to this image's own skin. Thresholds are
    % heuristic - mention this in the report.
    Lp = les(:,1); ap = les(:,2); bp = les(:,3);
    C = hypot(ap, bp); h = atan2d(bp, ap);              % h ~ 0 red, ~60-70 brown/yellow
    refNames = {'White', 'Red', 'LightBrown', 'DarkBrown', 'BlueGray', 'Black'};
    cls = zeros(size(Lp));
    cls(Lp < 25)                                     = 6;   % black
    cls(cls == 0 & Lp >= skin(1) - 5 & C < 15)       = 1;   % white: as light as skin, unpigmented
    cls(cls == 0 & (bp < 3 | h > 100) & C < 15)      = 5;   % blue-gray: no yellow component
    cls(cls == 0 & h < 40 & ap > 15)                 = 2;   % red: red dominates yellow
    cls(cls == 0 & Lp < skin(1) - 25)                = 4;   % dark brown
    cls(cls == 0)                                    = 3;   % light brown (rest)
    share = accumarray(cls, 1, [6 1]) / numel(cls);
    for k = 1:6
        f.(['color' refNames{k}]) = share(k);
    end
    f.numColors = sum(share >= 0.05);

    % Pigmentation variation
    L = lab(:,:,1);
    f.entropyL = entropy(L(M) / 100);
    dist = bwdist(~M); dist = dist / max(dist(:));     % 0 at border, 1 at center
    inner = M & dist > 0.5; outer = M & dist <= 0.25;
    f.centerPeripheryL = mean(L(outer)) - mean(L(inner));   % >0 = darker center
end
