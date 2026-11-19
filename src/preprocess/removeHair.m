function [Iout, hairMask] = removeHair(I, lesionMask, p)
%REMOVEHAIR DullRazor-style hair and ruler-mark removal.
%   1. Black top-hat with line elements in 12 orientations picks up thin
%      dark structures narrower than the line length.
%   2. Threshold, then keep only elongated components (hairs are long and
%      thin; pigment dots/globules inside the lesion are compact, so they
%      survive).
%   3. Inpaint the hair pixels channel by channel with regionfill.

    G = rgb2gray(I);
    th = zeros(size(G));
    for ang = 0:15:165
        th = max(th, imbothat(G, strel('line', p.hairLineLen, ang)));
    end
    cand = th > p.hairThresh;
    cand = bwareaopen(cand, 20);

    % Keep only thin, long, UNBRANCHED curves. Pigment network inside the
    % lesion is also thin and dark, but it is a mesh (many branch points)
    % and is short relative to its area - it must NOT be inpainted, since it
    % is a key malignancy cue.
    cc = bwconncomp(cand);
    keep = false(1, cc.NumObjects);
    for k = 1:cc.NumObjects
        comp = false(size(G)); comp(cc.PixelIdxList{k}) = true;
        sk = bwskel(comp);
        len = nnz(sk);
        if len < 12, continue; end
        width = numel(cc.PixelIdxList{k}) / len;          % mean stroke width (px)
        branches = nnz(bwmorph(sk, 'branchpoints'));
        branchDensity = branches / len * 100;             % per 100 px of length
        insideFrac = 0;
        if nargin > 1 && ~isempty(lesionMask)
            insideFrac = mean(lesionMask(cc.PixelIdxList{k}));
        end
        % stricter inside the lesion, where pigment structures live
        if insideFrac > 0.5
            keep(k) = len >= 40 && width <= 5 && branchDensity < 1.5;
        else
            keep(k) = width <= 8 && branchDensity < 6;   % skin: be generous
        end
    end
    hairMask = false(size(G));
    hairMask(vertcat(cc.PixelIdxList{keep})) = true;
    hairMask = imdilate(hairMask, strel('disk', 2));

    % Don't let a huge "hair" mask wipe out lesion texture (safety cap)
    if nargin > 1 && ~isempty(lesionMask) && mean(hairMask(lesionMask)) > 0.25
        hairMask = hairMask & ~lesionMask;
    end

    Iout = I;
    if any(hairMask(:))
        for c = 1:3
            Iout(:,:,c) = regionfill(I(:,:,c), hairMask);
        end
    end
end
