function f = extractFeatures(P)
%EXTRACTFEATURES All handcrafted lesion features for one preprocessed image.
%   P is the struct saved by s01_preprocess (fields smoothed, enhanced, mask).
%   Returns a flat struct of scalars; field-name prefixes tell the group:
%     shape: areaFraction compactness solidity extent eccentricity aspectRatio convexity
%     asym*  : asymmetry (shape and color)
%     border : radialCV fractalDim borderGrad*
%     color  : mean*/std*/skew* contrastL deltaESkin color* numColors entropyL centerPeripheryL
%     glcm*/lbp* : texture

    M = P.mask;
    lab = rgb2lab(P.smoothed);

    f = shapeFeatures(M);
    f = mergeStruct(f, asymmetryFeatures(M, lab));
    f = mergeStruct(f, borderFeatures(M, lab(:,:,1)));
    f = mergeStruct(f, colorFeatures(P.smoothed, M, lab));
    f = mergeStruct(f, textureFeatures(rgb2gray(P.enhanced), M));
end

function a = mergeStruct(a, b)
    for fn = fieldnames(b)'
        a.(fn{1}) = b.(fn{1});
    end
end
