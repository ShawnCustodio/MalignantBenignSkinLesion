function M = cleanMask(M)
%CLEANMASK Fill holes, smooth slightly and keep the largest lesion component.
    M = imfill(M, 'holes');
    M = imopen(M, strel('disk', 2));
    if any(M(:))
        M = bwareafilt(M, 1);
    end
end
