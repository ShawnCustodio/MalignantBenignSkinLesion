function q = imageQuality(Iraw, Mraw, hairMask)
%IMAGEQUALITY Simple quality descriptors of the RAW (resized) image.
%   Saved per image so later we can ask "do failures correlate with poor
%   image quality?" (proposal: effect of image quality on performance).
    lab = rgb2lab(Iraw); L = lab(:,:,1);
    skin = ~imdilate(Mraw, strel('disk', 10));
    % Vignetting: skin brightness near the image center vs. in the corners
    % (lesion pixels excluded, otherwise the dark lesion dominates "center")
    [h, w] = size(L);
    [X, Y] = meshgrid(1:w, 1:h);
    rad = hypot((X - w/2) / (w/2), (Y - h/2) / (h/2));   % 0 center, ~1.41 corner
    inner = skin & rad < 0.7;
    outer = skin & rad > 1.1;
    if ~any(inner(:)), inner = skin & rad < 1; end
    a = lab(:,:,2); b = lab(:,:,3);

    q.brightness    = mean(L(skin));                       % 0-100
    q.contrast      = std(L(:));
    q.lesionSkinDL  = mean(L(skin)) - mean(L(Mraw));       % how dark the lesion is vs skin
    q.sharpness     = var(reshape(del2(L), [], 1));        % variance of Laplacian
    q.vignetting    = mean(L(inner)) - mean(L(outer));     % >0 = darker corners
    q.hairFraction  = mean(hairMask(:));
    q.lesionFraction = mean(Mraw(:));
    q.colorCastAB   = hypot(mean(a(skin)), mean(b(skin))); % skin tint strength
end
