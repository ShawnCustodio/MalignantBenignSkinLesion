function P = preprocessLesion(I, M, cfg)
%PREPROCESSLESION Full preprocessing chain for one image + mask.
%   P = preprocessLesion(I, M, cfg) with I RGB double [0,1], M logical.
%
%   Steps (each kept in P so they can be shown/compared):
%     1. resized      - shorter side = cfg.pre.shortSide, aspect kept
%     2. hairRemoved  - DullRazor hair/ruler removal + inpainting
%     3. illumCorrected - vignetting / uneven light removed (L* only)
%     4. colorNorm    - Shades-of-Gray color constancy
%     5. smoothed     - edge-preserving bilateral filter (noise)  <- USE FOR COLOR FEATURES
%     6. enhanced     - CLAHE on L* of smoothed                    <- USE FOR TEXTURE / GRAPH
%   P.mask is the cleaned, resized lesion mask. P.quality has raw-image
%   quality descriptors.
%
%   Why two outputs: CLAHE boosts local contrast (good for texture) but
%   distorts absolute color, so color/pigmentation features should come from
%   P.smoothed, not P.enhanced.

%   AUTOMATIC MODE: pass M = [] and no hand-drawn mask is used anywhere.
%   A rough automatic mask guides hair removal / illumination, and the
%   final P.mask comes from segmentLesion on the cleaned image.

    p = cfg.pre;
    scale = p.shortSide / min(size(I,1), size(I,2));
    P.resized = min(max(imresize(I, scale, 'bicubic'), 0), 1);
    P.autoMask = isempty(M);
    if P.autoMask
        % rough guide only (dark core, full Otsu threshold)
        P.mask = segmentLesion(P.resized, struct('refine', false, 'threshScale', 1.0));
    else
        P.mask = cleanMask(imresize(M, [size(P.resized,1) size(P.resized,2)], 'nearest'));
    end

    [P.hairRemoved, P.hairMask] = removeHair(P.resized, P.mask, p);
    [P.illumCorrected, P.illumField] = correctIllumination(P.hairRemoved, P.mask, p.illumSigma);

    skin = ~imdilate(P.mask, strel('disk', 10));
    P.colorNorm = shadesOfGray(P.illumCorrected, skin, p.grayWorldP);

    P.smoothed = imbilatfilt(P.colorNorm, 0.005, 2);

    if P.autoMask
        P.mask = segmentLesion(P.smoothed);    % final automatic mask on the clean image
    end

    lab = rgb2lab(P.smoothed);
    lab(:,:,1) = 100 * adapthisteq(lab(:,:,1) / 100, 'ClipLimit', p.clipLimit);
    P.enhanced = min(max(lab2rgb(lab), 0), 1);

    P.quality = imageQuality(P.resized, P.mask, P.hairMask);
end
