function s = segmentationScores(Mauto, Mgt)
%SEGMENTATIONSCORES Overlap between automatic and hand-drawn masks.
%   Dice = 2|A&B| / (|A|+|B|), Jaccard = |A&B| / |A|B|  (1 = perfect).
    TP = nnz(Mauto & Mgt); FP = nnz(Mauto & ~Mgt); FN = nnz(~Mauto & Mgt); TN = nnz(~Mauto & ~Mgt);
    s.dice        = 2*TP / max(2*TP + FP + FN, 1);
    s.jaccard     = TP / max(TP + FP + FN, 1);
    s.sensitivity = TP / max(TP + FN, 1);     % share of true lesion found
    s.specificity = TN / max(TN + FP, 1);     % share of skin correctly left out
    s.areaRatio   = nnz(Mauto) / max(nnz(Mgt), 1);   % >1 = over-segmented
end
