function [Iout, gains] = shadesOfGray(I, skinMask, p)
%SHADESOFGRAY Color constancy (Finlayson & Trezzi 2004), widely used on ISIC.
%   Estimates the light-source color as the Minkowski p-norm of each channel
%   over the skin pixels, then rescales channels so the skin looks neutral.
%   Removes the yellow/pink casts that differ between cameras/clinics.

    if nargin < 2 || isempty(skinMask), skinMask = true(size(I,1), size(I,2)); end
    illum = zeros(1, 3);
    for c = 1:3
        ch = I(:,:,c);
        illum(c) = mean(ch(skinMask).^p)^(1/p);
    end
    illum = illum / norm(illum);
    gains = 1 ./ (illum * sqrt(3));
    Iout = min(I .* reshape(gains, 1, 1, 3), 1);
end
