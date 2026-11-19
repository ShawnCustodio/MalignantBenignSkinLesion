function [I, M] = readSample(T, i)
%READSAMPLE Read image i (RGB double in [0,1]) and its logical lesion mask.
    I = im2double(imread(T.imagePath(i)));
    M = imread(T.maskPath(i));
    if size(M, 3) > 1, M = M(:,:,1); end
    M = M > 0;
    if ~isequal(size(M), [size(I,1) size(I,2)])
        M = imresize(M, [size(I,1) size(I,2)], 'nearest');
    end
end
