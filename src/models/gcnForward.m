function [logits, H2] = gcnForward(params, A, X, Pool, dropoutRate, training)
%GCNFORWARD Two graph-convolution layers + mean pooling + linear classifier.
%   A    : normalized adjacency (block-diagonal for a mini-batch), N x N
%   X    : node features, N x d
%   Pool : B x N mean-pooling matrix (row b averages the nodes of graph b)
%   Returns logits (B x 2, columns = [benign malignant]) and node
%   embeddings H2 (N x hidden).
%
%   Graph convolution (Kipf & Welling 2017):  H' = ReLU( A * H * W + b )
%   Each layer mixes every region with its neighbors, so after 2 layers a
%   node "sees" regions up to 2 steps away - this is how differences between
%   neighboring regions enter the model.
%
%   Because pooling is a MEAN followed by a LINEAR layer, the malignant-vs-
%   benign logit is exactly the average of per-node scores
%       s_i = H2_i * (Wfc(:,2) - Wfc(:,1))
%   which we use as the region-level explanation (graph class activation map).

    H1 = relu(A * X * params.W1 + params.b1);
    if training, H1 = dropoutMask(H1, dropoutRate); end
    H2 = relu(A * H1 * params.W2 + params.b2);
    if training, H2 = dropoutMask(H2, dropoutRate); end
    logits = Pool * H2 * params.Wfc + params.bfc;
end

function H = dropoutMask(H, p)
    if p > 0
        H = H .* (rand(size(H), 'like', extractdata(H)) > p) / (1 - p);
    end
end
