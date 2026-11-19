function [Iout, field] = correctIllumination(I, lesionMask, sigma)
%CORRECTILLUMINATION Remove uneven lighting / vignetting.
%   The illumination field is estimated from the SKIN only: the lesion is
%   filled in from its surroundings (regionfill) so the dark lesion does not
%   pull the estimate down, then the L* channel is blurred heavily. L* is
%   divided by this field and rescaled to the median skin brightness.
%   Chroma (a*, b*) is left untouched so lesion colors are preserved.

    lab = rgb2lab(I);
    L = lab(:,:,1);

    fillMask = imdilate(lesionMask, strel('disk', 15));
    Lskin = regionfill(L, fillMask);
    field = imgaussfilt(Lskin, sigma, 'Padding', 'symmetric');
    field = max(field, 5);

    target = median(field(~fillMask));
    lab(:,:,1) = min(L ./ field * target, 100);
    Iout = min(max(lab2rgb(lab), 0), 1);
end
