function p = mcnemarExact(correctA, correctB)
%MCNEMAREXACT Two-sided exact McNemar test for two classifiers on the same images.
%   Only images where exactly one model is right are informative:
%   b = A right & B wrong, c = A wrong & B right. Under H0 (equal accuracy)
%   b ~ Binomial(b + c, 0.5).
    b = sum(correctA & ~correctB);
    c = sum(~correctA & correctB);
    nd = b + c;
    if nd == 0, p = 1; return; end
    p = min(1, 2 * binocdf(min(b, c), nd, 0.5));
end
