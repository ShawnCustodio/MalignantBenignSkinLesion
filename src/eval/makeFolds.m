function cv = makeFolds(labels, cfg)
%MAKEFOLDS Stratified k-fold partition, reproducible via cfg.seed.
%   Use the SAME partition for Random Forest and GCN so results are comparable:
%       for f = 1:cv.NumTestSets
%           trainIdx = training(cv, f);  testIdx = test(cv, f);
%       end
    rng(cfg.seed);
    cv = cvpartition(labels, 'KFold', cfg.kFolds, 'Stratify', true);
end
