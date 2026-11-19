function [pMal, nodeScores] = gcnPredict(model, graphs)
%GCNPREDICT Malignancy probability per graph + per-region explanation scores.
%   nodeScores{i}(k) > 0 : region k pushes lesion i toward MALIGNANT
%   nodeScores{i}(k) < 0 : region k pushes toward BENIGN
%   The lesion's logit difference equals mean(nodeScores{i}) + bias.
    n = numel(graphs);
    pMal = zeros(n, 1); nodeScores = cell(n, 1);
    p = model.params;
    wDiff = p.Wfc(:, 2) - p.Wfc(:, 1);
    for i = 1:n
        [A, X, Pool] = gcnBatch(graphs(i), model);
        [logits, H2] = gcnForward(p, A, X, Pool, 0, false);
        z = double(extractdata(logits));
        pMal(i) = 1 / (1 + exp(z(1) - z(2)));
        nodeScores{i} = double(extractdata(H2 * wDiff));
    end
end
